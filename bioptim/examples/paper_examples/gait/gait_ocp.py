"""
This example is an updated version of the gait example from the original Bioptim paper.
This example is presented to introduce Bioptim’s ability to deal with a multiphase
locomotion estimation problem, including muscle actuation and contact forces.
The goal was to estimate muscle activations by tracking markers trajectories and ground
reaction forces. The gait cycle was defined from the first heel strike to the end of the swing phase.
"""

import platform

import numpy as np
from scipy.interpolate import interp1d
from casadi import vertcat, sum1, MX

from bioptim import (
    OptimalControlProgram,
    DynamicsFcn,
    BoundsList,
    InitialGuessList,
    ObjectiveFcn,
    OdeSolver,
    OdeSolverBase,
    Solver,
    BiorbdModel,
    ControlType,
    PhaseDynamics,
    Node,
    ObjectiveList,
    DynamicsList,
    ConstraintList,
    ConstraintFcn,
    Axis,
    PhaseTransitionList,
    PhaseTransitionFcn,
    InterpolationType,
    PenaltyController,
)

from load_experimental_data import LoadData


def get_contact_index(model, tag):
    force_names = [s for s in model.contact_names]
    return [i for i, t in enumerate([s[-1] == tag for s in force_names]) if t]

# --- track grf ---
def track_sum_contact_forces(controller: PenaltyController) -> MX:
    """
    Adds the objective that the mismatch between the
    sum of the contact forces and the reference ground reaction forces should be minimized.

    Parameters
    ----------
    controller: PenaltyController
        Thepenalty controller

    Returns
    -------
    The cost that should be minimized in the MX format.
    """

    q = controller.states["q"].mx
    qdot = controller.states["qdot"].mx
    tau = controller.controls["tau"].mx

    force_tp = controller.model.contact_forces(q, qdot, tau)

    force = vertcat(sum1(force_tp[get_contact_index(controller.model, "X"), :]),
                    sum1(force_tp[get_contact_index(controller.model, "Y"), :]),
                    sum1(force_tp[get_contact_index(controller.model, "Z"), :]))
    out = controller.mx_to_cx("grf", force, controller.states["q"],
                              controller.states["qdot"],
                              controller.controls["tau"])
    return out
#
def get_phase_time_shooting_numbers(data, dt):
    """
    Get the duration of the phases from the number of frames and frame rate from the c3d file.
    """
    phase_time = data.c3d_data.get_time()
    number_shooting_points = []
    for time in phase_time:
        number_shooting_points.append(int(time / dt))
    return phase_time, number_shooting_points

def get_experimental_data(data, number_shooting_points, phase_time):
    q_ref = data.dispatch_data(data=data.q, n_shooting=number_shooting_points, phase_time=phase_time)
    qdot_ref = data.dispatch_data(data=data.qdot, n_shooting=number_shooting_points, phase_time=phase_time)
    markers_ref = data.dispatch_data(data=data.c3d_data.trajectories, n_shooting=number_shooting_points, phase_time=phase_time)
    grf_ref = data.dispatch_data(data=data.c3d_data.forces, n_shooting=number_shooting_points, phase_time=phase_time)
    moments_ref = data.dispatch_data(data=data.c3d_data.moments, n_shooting=number_shooting_points, phase_time=phase_time)
    cop_ref = data.dispatch_data(data=data.c3d_data.cop, n_shooting=number_shooting_points, phase_time=phase_time)
    emg_ref = data.dispatch_data(data=data.c3d_data.emg, n_shooting=number_shooting_points, phase_time=phase_time)
    return q_ref, qdot_ref, markers_ref, grf_ref, moments_ref, cop_ref, emg_ref


def prepare_ocp(
    bio_models: list[BiorbdModel],
    final_time: list[float],
    n_shooting: list[int],
    markers_ref: list[np.ndarray],
    grf_ref: list[np.ndarray],
    q_ref: list[np.ndarray],
    qdot_ref: list[np.ndarray],
    activation_ref: list[np.ndarray],
    ode_solver: OdeSolverBase = OdeSolver.RK4(),
    use_sx: bool = True,
    n_threads: int = 1,
    phase_dynamics: PhaseDynamics = PhaseDynamics.SHARED_DURING_THE_PHASE,
    expand_dynamics: bool = True,
    control_type: ControlType = ControlType.CONSTANT,
) -> OptimalControlProgram:
    """
    The initialization of an ocp

    Parameters
    ----------
    bio_models: list[BiorbdModel]
        The path to the biorbd models
    final_time: list[float]
        The time of each phase of the movement
    n_shooting: list[int]
        The number of shooting points for each phase of the movement
    markers_ref: list[np.ndarray]
        The marker positions to track
    grf_ref: list[np.ndarray]
        The ground reaction forces to track
    q_ref: list[np.ndarray]
        The generalized coordinates to track
    qdot_ref: list[np.ndarray]
        The generalized velocities to track
    activation_ref: list[np.ndarray]
        The muscle activations to track
    ode_solver: OdeSolverBase = OdeSolver.RK4()
        Which type of OdeSolver to use
    use_sx: bool
        If the SX variable should be used instead of MX (can be extensive on RAM)
    n_threads: int
        The number of threads to use in the paralleling (1 = no parallel computing)
    phase_dynamics: PhaseDynamics
        If the dynamics equation within a phase is unique or changes at each node.
        PhaseDynamics.SHARED_DURING_THE_PHASE is much faster, but lacks the capability to have changing dynamics within
        a phase. A good example of when PhaseDynamics.ONE_PER_NODE should be used is when different external forces
        are applied at each node
    expand_dynamics: bool
        If the dynamics function should be expanded. Please note, this will solve the problem faster, but will slow down
        the declaration of the OCP, so it is a trade-off. Also depending on the solver, it may or may not work
        (for instance IRK is not compatible with expanded dynamics)
    control_type: ControlType
        The type of the controls

    Returns
    -------
    The OptimalControlProgram ready to be solved
    """

    # Problem parameters
    nb_phases = len(bio_models)
    nb_tau = bio_models[0].nb_tau
    nb_mus = bio_models[0].nb_muscles

    min_bound, max_bound = 0, np.inf
    torque_min, torque_max, torque_init = -1000, 1000, 0
    activation_min, activation_max, activation_init = 1e-3, 1.0, 0.1

    # Add objective functions
    markers_pelvis = [0, 1, 2, 3]  # ["L_IAS", "L_IPS", "R_IAS", "R_IPS"]
    markers_anat = [4, 9, 10, 11, 12, 17, 18]  # ["R_FTC", "R_FLE", "R_FME", "R_FAX", "R_TTC", "R_FAL", "R_TAM"]
    markers_tissus = [5, 6, 7, 8, 13, 14, 15, 16]  # ["R_Thigh_Top", "R_Thigh_Down", "R_Thigh_Front", "R_Thigh_Back", "R_Shank_Top", "R_Shank_Down", "R_Shank_Front", "R_Shank_Tibia"]
    markers_foot = [19, 20, 21, 22, 23, 24, 25]  # ["R_FCC", "R_FM1", "R_FMP1", "R_FM2", "R_FMP2", "R_FM5", "R_FMP5"]
    markers_index = (markers_pelvis, markers_anat, markers_foot, markers_tissus)
    weight = (10000, 1000, 10000, 100)
    objective_functions = ObjectiveList()
    for p in range(nb_phases):
        for (i, m_idx) in enumerate(markers_index):
            objective_functions.add(
                ObjectiveFcn.Lagrange.TRACK_MARKERS,
                node=Node.ALL,
                weight=weight[i],
                marker_index=m_idx,
                target=markers_ref[p][:, m_idx, :],
                quadratic=True,
                phase=p,
            )
        objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=0.001, index=(10, 12), quadratic=True, phase=p)
        objective_functions.add(
            ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=1, index=(6, 7, 8, 9, 11), phase=p, quadratic=True,
        )
        objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="muscles", weight=10, phase=p, quadratic=True)
        objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", derivative=True, weight=0.1, quadratic=True, phase=p)

    # --- track contact forces for the stance phase ---
    for p in range(nb_phases - 1):
        objective_functions.add(
            track_sum_contact_forces,
            custom_type=ObjectiveFcn.Lagrange,
            target=grf_ref[p],
            node=Node.ALL,
            weight=0.01,
            quadratic=True,
            phase=p,
        )

    # Dynamics
    dynamics = DynamicsList()
    for p in range(nb_phases - 1):
        dynamics.add(DynamicsFcn.MUSCLE_DRIVEN, phase=p, with_contact=True, with_residual_torque=True, expand_dynamics=expand_dynamics)
    dynamics.add(DynamicsFcn.MUSCLE_DRIVEN, phase=3, with_residual_torque=True, expand_dynamics=expand_dynamics)

    # Constraints
    m_heel, m_m1, m_m5, m_toes = 26, 27, 28, 29
    constraints = ConstraintList()
    # null speed for the first phase --> non-sliding contact point
    constraints.add(ConstraintFcn.TRACK_MARKERS_VELOCITY, node=Node.START, marker_index=m_heel, phase=0)
    # on the ground z=0
    constraints.add(ConstraintFcn.TRACK_MARKERS, node=Node.START, marker_index=m_heel, axes=Axis.Z, phase=0)

    # --- phase flatfoot ---
    Fz_heel, Fz_m1, Fx_m5, Fy_m5, Fz_m5 = 0, 1, 2, 3, 4
    # on the ground z=0
    constraints.add(ConstraintFcn.TRACK_MARKERS, node=Node.START, marker_index=[m_m1, m_m5], axes=Axis.Z, phase=1)
    constraints.add(  # positive vertical forces
        ConstraintFcn.TRACK_CONTACT_FORCES,
        min_bound=min_bound,
        max_bound=max_bound,
        node=Node.ALL,
        contact_index=(Fz_heel, Fz_m1, Fz_m5),
        phase=1,
    )
    constraints.add(  # non slipping
        ConstraintFcn.NON_SLIPPING,
        node=Node.ALL,
        tangential_component_idx=(Fx_m5, Fy_m5),
        normal_component_idx=(Fz_heel, Fz_m1, Fz_m5),
        static_friction_coefficient=0.5,
        phase=1,
    )
    constraints.add(  # forces heel at zeros at the end of the phase
        ConstraintFcn.TRACK_CONTACT_FORCES,
        node=Node.PENULTIMATE,
        contact_index=[i for i, name in enumerate(bio_models[1].contact_names) if "Heel_r" in name],
        phase=1,
    )

    # --- phase forefoot ---
    Fz_m1, Fx_m5, Fy_m5, Fz_m5, Fz_toe = 0, 1, 2, 3, 4
    constraints.add(  # positive vertical forces
        ConstraintFcn.TRACK_CONTACT_FORCES,
        min_bound=min_bound,
        max_bound=max_bound,
        node=Node.ALL,
        contact_index=(Fz_m1, Fz_m5, Fz_toe),
        phase=2,
    )
    constraints.add(  # non slipping x m1
        ConstraintFcn.NON_SLIPPING,
        node=Node.ALL,
        tangential_component_idx=(Fx_m5, Fy_m5),
        normal_component_idx=(Fz_m1, Fz_m5, Fz_toe),
        static_friction_coefficient=0.5,
        phase=2,
    )

    # Phase Transitions
    phase_transitions = PhaseTransitionList()
    phase_transitions.add(PhaseTransitionFcn.IMPACT, phase_pre_idx=0)
    phase_transitions.add(PhaseTransitionFcn.IMPACT, phase_pre_idx=1)
    
    # Path constraint
    x_bounds = BoundsList()
    u_bounds = BoundsList()
    for p in range(nb_phases):
        x_bounds.add("q", min_bound=bio_models[p].bounds_from_ranges("q").min,
                     max_bound=bio_models[p].bounds_from_ranges("q").max, phase=p)
        x_bounds.add("qdot", min_bound=bio_models[p].bounds_from_ranges("qdot").min,
                     max_bound=bio_models[p].bounds_from_ranges("qdot").max, phase=p)
        u_bounds.add("tau", min_bound=[torque_min] * nb_tau, max_bound=[torque_max] * nb_tau, phase=p)
        u_bounds.add("muscles", min_bound=[activation_min] * nb_mus, max_bound=[activation_max] * nb_mus, phase=p)

    # Initial guess
    x_init = InitialGuessList()
    u_init = InitialGuessList()

    if ode_solver.is_direct_collocation:
        n_degree = ode_solver.polynomial_degree
        for p in range(nb_phases):
            t_init = np.linspace(0, final_time[p], n_shooting[p] + 1)
            t_node = np.linspace(0, final_time[p], (n_shooting[p])*(n_degree + 1) + 1)
            fq = interp1d(t_init, q_ref[p], kind="cubic")
            fqdot = interp1d(t_init, qdot_ref[p], kind="cubic")
            factivation = interp1d(t_init, activation_ref[p], kind="cubic")

            x_init.add("q", initial_guess=fq(t_node), interpolation=InterpolationType.EACH_FRAME, phase=p)
            x_init.add("qdot", initial_guess=fqdot(t_node), interpolation=InterpolationType.EACH_FRAME, phase=p)

            u_init.add("tau", initial_guess=np.zeros((nb_tau, (n_shooting[p]) * (n_degree + 1))), interpolation=InterpolationType.EACH_FRAME, phase=p)
            u_init.add("muscles", initial_guess=factivation(t_node)[:, :-1], interpolation=InterpolationType.EACH_FRAME, phase=p)
    else:
        for p in range(nb_phases):
            x_init.add("q", initial_guess=q_ref[p], interpolation=InterpolationType.EACH_FRAME, phase=p)
            x_init.add("qdot", initial_guess=qdot_ref[p], interpolation=InterpolationType.EACH_FRAME, phase=p)

            u_init.add("tau", initial_guess=np.zeros((nb_tau, n_shooting[p])), interpolation=InterpolationType.EACH_FRAME, phase=p)
            u_init.add("muscles", initial_guess=activation_ref[p][:, :-1], interpolation=InterpolationType.EACH_FRAME, phase=p)

    return OptimalControlProgram(
        bio_models,
        dynamics,
        n_shooting,
        final_time,
        x_init=x_init,
        u_init=u_init,
        x_bounds=x_bounds,
        u_bounds=u_bounds,
        objective_functions=objective_functions,
        ode_solver=ode_solver,
        use_sx=use_sx,
        n_threads=n_threads,
        control_type=control_type,
    )


def main():
    """
    Solve the OCP and animate the solution.
    """

    biorbd_model_paths = ["models/Gait_1leg_12dof_heel.bioMod",
                          "models/Gait_1leg_12dof_flatfoot.bioMod",
                          "models/Gait_1leg_12dof_forefoot.bioMod",
                          "models/Gait_1leg_12dof_0contact.bioMod"
                          ]
    bio_models = [BiorbdModel(path) for path in biorbd_model_paths]

    # --- files path ---
    c3d_file = "data/normal01_out.c3d"
    q_kalman_filter_file = "data/normal01_q_KalmanFilter.txt"
    qdot_kalman_filter_file = "data/normal01_qdot_KalmanFilter.txt"
    data = LoadData(bio_models[0], c3d_file, q_kalman_filter_file, qdot_kalman_filter_file)

    # --- phase time and number of shooting ---
    phase_time, number_shooting_points = get_phase_time_shooting_numbers(data, 0.01)

    # --- get experimental data ---
    q_ref, qdot_ref, markers_ref, grf_ref, moments_ref, cop_ref, emg_ref = get_experimental_data(data, number_shooting_points, phase_time)

    solver = Solver.IPOPT(show_online_optim=platform.system() == "Linux")
    # solver.set_convergence_tolerance(1e-3)
    # solver.set_hessian_approximation("exact")
    # solver.set_maximum_iterations(3000)

    # --- Prepare the ocp --- #
    ocp = prepare_ocp(
        bio_models,
        phase_time,
        number_shooting_points,
        markers_ref,
        grf_ref,
        q_ref,
        qdot_ref,
        emg_ref,
        n_threads=4,
        ode_solver=OdeSolver.RK4(),
        use_sx=True,
        control_type=ControlType.CONSTANT,
    )

    # --- Solve the ocp --- #
    sol = ocp.solve(solver)
    sol.print_cost()

    # --- Show the results --- #
    # sol.graphs(show_bounds=True)
    sol.animate()


if __name__ == "__main__":
    main()
