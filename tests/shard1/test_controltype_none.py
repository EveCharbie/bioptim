"""
Test for file IO.
"""

from typing import Callable
from casadi import vertcat, SX, MX
import numpy as np
import pytest
from bioptim import (
    BoundsList,
    ConfigureProblem,
    ConstraintFcn,
    ConstraintList,
    ControlType,
    DynamicsEvaluation,
    DynamicsList,
    ObjectiveFcn,
    ObjectiveList,
    OdeSolver,
    OdeSolverBase,
    OptimalControlProgram,
    Node,
    NonLinearProgram,
    Solver,
    PhaseDynamics,
)


class NonControlledMethod:
    """
    This is a custom model that inherits from bioptim.CustomModel
    As CustomModel is an abstract class, some methods must be implemented.
    """

    def __init__(self, name: str = None):
        self.a = 0
        self.b = 0
        self.c = 0
        self._name = name

    def serialize(self) -> tuple[Callable, dict]:
        # This is where you can serialize your model
        # This is useful if you want to save your model and load it later
        return NonControlledMethod, dict()

    @property
    def name_dof(self) -> list[str]:
        return ["a", "b", "c"]

    @property
    def nb_state(self):
        return 3

    @property
    def name(self):
        return self._name

    def system_dynamics(
        self,
        a: MX | SX,
        b: MX | SX,
        c: MX | SX,
        t: MX | SX,
        t_phase: MX | SX,
    ) -> MX | SX:
        """
        The system dynamics is the function that describes the model.

        Returns
        -------
        The value of the derivative of each state dx/dt at the current time t
        """
        a_dot = 100 + b
        b_dot = a / (((t - t_phase) + 1) * 100)
        c_dot = a * b + c

        return vertcat(a_dot, b_dot, c_dot)

    def custom_dynamics(
        self,
        time: MX | SX,
        states: MX | SX,
        controls: MX | SX,
        parameters: MX | SX,
        stochastic_variables: MX | SX,
        nlp: NonLinearProgram,
        t_phase: MX | SX,
    ) -> DynamicsEvaluation:

        return DynamicsEvaluation(
            dxdt=self.system_dynamics(a=states[0], b=states[1], c=states[2], t=time, t_phase=t_phase),
            defects=None,
        )

    def declare_variables(self, ocp: OptimalControlProgram, nlp: NonLinearProgram):
        name = "a"
        name_a = [name]
        ConfigureProblem.configure_new_variable(
            name,
            name_a,
            ocp,
            nlp,
            as_states=True,
            as_controls=False,
            as_states_dot=False,
        )

        name = "b"
        name_b = [name]
        ConfigureProblem.configure_new_variable(
            name,
            name_b,
            ocp,
            nlp,
            as_states=True,
            as_controls=False,
            as_states_dot=False,
        )

        name = "c"
        name_c = [name]
        ConfigureProblem.configure_new_variable(
            name,
            name_c,
            ocp,
            nlp,
            as_states=True,
            as_controls=False,
            as_states_dot=False,
        )

        # t_phase = 0
        # for i in range(nlp.phase_idx):
        #     t_phase = nlp.dt
        t_phase = ocp.node_time(phase_idx=nlp.phase_idx, node_idx=0)
        # t_phase = ocp.nlp[nlp.phase_idx].dt
        ConfigureProblem.configure_dynamics_function(ocp, nlp, self.custom_dynamics, t_phase=t_phase, allow_free_variables=True)
        # ConfigureProblem.configure_dynamics_function(ocp, nlp, self.custom_dynamics, t_phase=t_phase)

def prepare_ocp(
    n_phase: int,
    time_min: list,
    time_max: list,
    use_sx: bool,
    ode_solver: OdeSolverBase = OdeSolver.RK4(n_integration_steps=5,allow_free_variables=True),
    # ode_solver: OdeSolverBase = OdeSolver.RK4(n_integration_steps=5),
    phase_dynamics: PhaseDynamics = PhaseDynamics.SHARED_DURING_THE_PHASE,
) -> OptimalControlProgram:
    """
    Prepare the ocp

    Parameters
    ----------
    n_phase: int
        Number of phase
    time_min: list
        The minimal time for each phase
    time_max: list
        The maximal time for each phase
    ode_solver: OdeSolverBase
        The ode solver to use
    use_sx: bool
        Callable Mx or Sx used for ocp
    phase_dynamics: PhaseDynamics
        If the dynamics equation within a phase is unique or changes at each node.
        PhaseDynamics.SHARED_DURING_THE_PHASE is much faster, but lacks the capability to have changing dynamics within
        a phase. A good example of when PhaseDynamics.ONE_PER_NODE should be used is when different external forces
        are applied at each node

    Returns
    -------
    The OptimalControlProgram ready to be solved
    """
    custom_model = NonControlledMethod()
    models = (
        NonControlledMethod(),
        NonControlledMethod(),
        NonControlledMethod(),
        NonControlledMethod(),
        NonControlledMethod(),
        NonControlledMethod(),
        NonControlledMethod(),
        NonControlledMethod(),
        NonControlledMethod(),
        NonControlledMethod(),
    )
    n_shooting = [5 for i in range(n_phase)]  # Gives m node shooting for my n phases problem
    final_time = [0.01 for i in range(n_phase)]  # Set the final time for all my n phases

    # Creates the system's dynamic for my n phases
    dynamics = DynamicsList()
    for i in range(n_phase):
        dynamics.add(
            custom_model.declare_variables,
            dynamic_function=custom_model.custom_dynamics,
            phase=i,
            expand_dynamics=True,
            phase_dynamics=phase_dynamics,
        )

    # Creates the constraint for my n phases
    constraints = ConstraintList()
    for i in range(n_phase):
        constraints.add(
            ConstraintFcn.TIME_CONSTRAINT, node=Node.END, min_bound=time_min[i], max_bound=time_max[i], phase=i
        )

    objective_functions = ObjectiveList()
    objective_functions.add(
        ObjectiveFcn.Mayer.MINIMIZE_STATE, target=5, key="c", node=Node.END, quadratic=True, weight=1, phase=9
    )
    objective_functions.add(
        ObjectiveFcn.Mayer.MINIMIZE_STATE, target=100, key="a", node=Node.END, quadratic=True, weight=0.001, phase=9
    )

    # Sets the bound for all the phases
    x_bounds = BoundsList()
    for i in range(n_phase):
        x_bounds.add("a", min_bound=[[0, 0, 0]], max_bound=[[0 if i == 0 else 1000, 1000, 1000]], phase=i)
        x_bounds.add("b", min_bound=[[0, 0, 0]], max_bound=[[0 if i == 0 else 1000, 1000, 1000]], phase=i)
        x_bounds.add("c", min_bound=[[0, 0, 0]], max_bound=[[0 if i == 0 else 1000, 1000, 1000]], phase=i)

    return OptimalControlProgram(
        models,
        dynamics,
        n_shooting,
        final_time,
        x_bounds=x_bounds,
        objective_functions=objective_functions,
        constraints=constraints,
        ode_solver=ode_solver,
        control_type=ControlType.NONE,
        use_sx=use_sx,
    )


@pytest.mark.parametrize("phase_dynamics", [PhaseDynamics.ONE_PER_NODE])
@pytest.mark.parametrize("use_sx", [False])  #, True
def test_main_control_type_none(use_sx, phase_dynamics):
    """
    Prepare and solve and animate a reaching task ocp
    """

    # number of stimulation corresponding to phases
    n = 10
    # minimum time between two phase
    time_min = [0.01 for _ in range(n)]
    # maximum time between two phase
    time_max = [0.1 for _ in range(n)]
    ocp = prepare_ocp(
        n_phase=n,
        time_min=time_min,
        time_max=time_max,
        use_sx=use_sx,
        phase_dynamics=phase_dynamics,
    )

    # --- Solve the program --- #
    sol = ocp.solve(Solver.IPOPT(show_online_optim=False),)

    # a = []
    # b = []
    # c = []
    # import matplotlib.pyplot as plt
    # time = []
    # for i in range(len(sol.times)):
    #     time.append(sol.times[i][:-1])
    #
    #     flatten_a = [item for sublist in sol.decision_states()[i]["a"] for item in sublist]
    #     flatten_b = [item for sublist in sol.decision_states()[i]["b"] for item in sublist]
    #     flatten_c = [item for sublist in sol.decision_states()[i]["c"] for item in sublist]
    #
    #     flatten_a1 = [item for sublist in flatten_a for item in sublist]
    #     flatten_b1 = [item for sublist in flatten_b for item in sublist]
    #     flatten_c1 = [item for sublist in flatten_c for item in sublist]
    #
    #     a.append(flatten_a1)
    #     b.append(flatten_b1)
    #     c.append(flatten_c1)
    # time = [j for sub in sol.times for j in sub]
    # time = list(set(time))
    # time = [sum(time[0:x:1]) for x in range(0, len(time))]
    # a = [j for sub in a for j in sub]
    # b = [j for sub in b for j in sub]
    # c = [j for sub in c for j in sub]
    # plt.plot(time, a)
    # plt.plot(time, b)
    # plt.plot(time, c)
    # plt.show()

    # Check objective function value
    f = np.array(sol.cost)
    np.testing.assert_equal(f.shape, (1, 1))
    np.testing.assert_almost_equal(f[0, 0], 0.2919065990591678)

    # Check finishing time
    np.testing.assert_almost_equal(np.cumsum([t[-1] for t in sol.times])[-1], 0.8299336018055604)

    # Check constraints
    g = np.array(sol.constraints)
    for i in range(n):
        np.testing.assert_almost_equal(g[i * 19 + 0 : i * 19 + 15], np.zeros((15, 1)))
    np.testing.assert_almost_equal(
        g[18:-1:19, 0],
        [0.09848005, 0.0974753, 0.09652673, 0.09540809, 0.0939693, 0.09197322, 0.08894771, 0.08377719, 0.07337567],
    )
    np.testing.assert_equal(g.shape, (187, 1))

    # Check some results
    # first phase
    np.testing.assert_almost_equal(
        sol.decision_states()[0]["a"], np.array([0.0, 1.96960231, 3.93921216, 5.90883684, 7.87848335, 9.84815843]), decimal=8
    )
    np.testing.assert_almost_equal(
        sol.decision_states()[0]["b"], np.array([0.0, 0.00019337, 0.00076352, 0.00169617, 0.00297785, 0.0045958]), decimal=8
    )
    np.testing.assert_almost_equal(
        sol.decision_states()[0]["c"],
        np.array([0.00000000e00, 1.88768128e-06, 3.00098595e-05, 1.50979104e-04, 4.74274962e-04, 1.15105831e-03]),
        decimal=8,
    )

    # intermediate phase
    np.testing.assert_almost_equal(
        sol.decision_states()[5]["a"],
        np.array([48.20121535, 50.04237763, 51.88365353, 53.72504579, 55.56655709, 57.40819004]),
        decimal=8,
    )
    np.testing.assert_almost_equal(
        sol.decision_states()[5]["b"],
        np.array([0.08926236, 0.0953631, 0.10161488, 0.10801404, 0.11455708, 0.1212406]),
        decimal=8,
    )
    np.testing.assert_almost_equal(
        sol.decision_states()[5]["c"],
        np.array([0.60374532, 0.69912979, 0.80528341, 0.92297482, 1.05299864, 1.19617563]),
        decimal=8,
    )

    # last phase
    np.testing.assert_almost_equal(
        sol.decision_states()[9]["a"],
        np.array([82.06013653, 82.2605896, 82.4610445, 82.6615012, 82.86195973, 83.06242009]),
        decimal=8,
    )
    np.testing.assert_almost_equal(
        sol.decision_states()[9]["b"],
        np.array([0.22271563, 0.22362304, 0.22453167, 0.2254415, 0.22635253, 0.22726477]),
        decimal=8,
    )
    np.testing.assert_almost_equal(
        sol.decision_states()[9]["c"],
        np.array([4.83559727, 4.88198772, 4.92871034, 4.97576671, 5.02315844, 5.07088713]),
        decimal=8,
    )
