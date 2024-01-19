"""
This example is an updated version of the jumper example from the original Bioptim paper.
This example was designed to introduce Bioptim’s ability to reduce the number of DoFs of a
model via the BiMapping feature, to account for nonlinear boundaries on the controls,
and to solve complex multiphase OCP. Two phases were used to describe the dynamics of
the push-off phase of the jump: flat foot (two floor contacts) and then toe only (one contact).
A pseudo-2-D full-body symmetric model consisting of 3 DoFs at the pelvis (forward and
upward translations and transverse rotation), 1 DoF at the upper limb (shoulder flexion),
and 3 DoFs at the lower limb (hip, knee, and ankle flexion) was used. Since this is a full-body model,
the root segment (i.e., the pelvis) was left uncontrolled, reducing the number of control variables to four,
namely, the shoulder, hip, knee, and ankle flexions. The goal of the model was to jump as high as possible.
"""

import platform

import numpy as np

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
    PhaseTransitionList,
    PhaseTransitionFcn,
    BiMappingList,
)

from jumper_utils import find_initial_root_pose

def prepare_ocp(
        bio_models: list[BiorbdModel],
        final_time: list[float],
        time_min: list[float],
        time_max: list[float],
        n_shooting: list[int],
        ode_solver: OdeSolverBase = OdeSolver.RK4(),
        use_sx: bool = False,
        n_threads: int = 1,
        phase_dynamics: PhaseDynamics = PhaseDynamics.SHARED_DURING_THE_PHASE,
        expand_dynamics: bool = False,
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
    time_min: list[float]
        The minimum time of each phase of the movement
    time_max: list[float]
        The maximum time of each phase of the movement
    n_shooting: list[int]
        The number of shooting points for each phase of the movement
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

    # Mapping
    dof_mappings = BiMappingList()
    dof_mappings.add("q", to_second=[0, 1, 2, 3, 3, 4, 5, 6, 4, 5, 6], to_first=[0, 1, 2, 3, 5, 6, 7])
    dof_mappings.add("qdot", to_second=[0, 1, 2, 3, 3, 4, 5, 6, 4, 5, 6], to_first=[0, 1, 2, 3, 5, 6, 7])
    dof_mappings.add("tau", to_second=[None, None, None, 0, 0, 1, 2, 3, 1, 2, 3], to_first=[3, 5, 6, 7])

    # Problem parameters
    nb_phases = len(bio_models)
    nb_q = len(dof_mappings["q"].to_first.map_idx)
    nb_tau = len(dof_mappings["tau"].to_first.map_idx)
    n_root = bio_models[0].nb_root

    tau_constant_bound = 500
    body_at_first_node = [0, 0, 0, 2.10, 1.15, 0.80, 0.20]
    initial_velocity = [0, 0, 0, 0, 0, 0, 0]
    initial_pose = np.array([body_at_first_node]).T
    initial_velocity = np.array([initial_velocity]).T
    initial_pose[:n_root, 0] = find_initial_root_pose(bio_models[0], dof_mappings["q"].to_second.map(initial_pose))
    initial_states = np.concatenate((initial_pose, initial_velocity))

    tau_min = 20  # Tau minimal bound despite the torque activation
    arm_dof = 3
    heel_dof = 6
    heel_marker_idx = 85
    toe_marker_idx = 86

    flat_foot_phases = 0, 4  # The indices of flat foot phases
    toe_only_phases = 1, 3  # The indices of toe only phases

    flatfoot_contact_x_idx = ()  # Contacts indices of heel and toe in bioMod 2 contacts
    flatfoot_contact_y_idx = (1, 4)  # Contacts indices of heel and toe in bioMod 2 contacts
    flatfoot_contact_z_idx = (0, 2, 3, 5)  # Contacts indices of heel and toe in bioMod 2 contacts
    flatfoot_non_slipping = ((1,), (0, 2))  # (X-Y components), Z components

    toe_contact_x_idx = ()  # Contacts indices of toe in bioMod 1 contact
    toe_contact_y_idx = (0, 2)  # Contacts indices of toe in bioMod 1 contact
    toe_contact_z_idx = (1, 2)  # Contacts indices of toe in bioMod 1 contact
    toe_non_slipping = ((0,), 1)  # (X-Y components), Z components
    static_friction_coefficient = 0.5

    control_nodes = Node.ALL if control_type == ControlType.LINEAR_CONTINUOUS else Node.ALL_SHOOTING

    # Dynamics
    dynamics = DynamicsList()
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN, with_contact=True, phase_dynamics=phase_dynamics, expand_dynamics=expand_dynamics)  # Flat foot
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN, with_contact=True, phase_dynamics=phase_dynamics, expand_dynamics=expand_dynamics)  # Toe only
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN, phase_dynamics=phase_dynamics, expand_dynamics=expand_dynamics)  # Aerial phase
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN, with_contact=True, phase_dynamics=phase_dynamics, expand_dynamics=expand_dynamics)  # Toe only
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN, with_contact=True, phase_dynamics=phase_dynamics, expand_dynamics=expand_dynamics)  # Flat foot

    # Constraints
    constraints = ConstraintList()
    # Torque constrained to torqueMax
    for p in range(nb_phases):
        constraints.add(ConstraintFcn.TORQUE_MAX_FROM_Q_AND_QDOT, phase=p, node=control_nodes, min_torque=tau_min)

    # Positivity of CoM_dot on z axis prior the take-off (to make sure it jumps upward)
    constraints.add(ConstraintFcn.TRACK_COM_VELOCITY, phase=1, node=Node.END, min_bound=0, max_bound=np.inf)

    # Floor constraints for flat foot phases
    for p in flat_foot_phases:

        # Do not pull on floor
        for i in flatfoot_contact_z_idx:
            constraints.add(
                ConstraintFcn.TRACK_CONTACT_FORCES, phase=p, node=control_nodes, contact_index=i, min_bound=0, max_bound=np.inf
            )

        # Non-slipping constraints
        constraints.add(  # On only one of the feet
            ConstraintFcn.NON_SLIPPING,
            phase=p,
            node=control_nodes,
            tangential_component_idx=flatfoot_non_slipping[0],
            normal_component_idx=flatfoot_non_slipping[1],
            static_friction_coefficient=static_friction_coefficient,
        )

    # Floor constraints for toe only phases
    for p in toe_only_phases:

        # Do not pull on floor
        for i in toe_contact_z_idx:
            constraints.add(
                ConstraintFcn.TRACK_CONTACT_FORCES, phase=p, node=control_nodes, contact_index=i, min_bound=0, max_bound=np.inf
            )

        # The heel must remain over floor
        constraints.add(
            ConstraintFcn.TRACK_MARKERS,
            phase=p,
            node=Node.ALL,
            index=2,
            min_bound=-0.0001,
            max_bound=np.inf,
            marker_index=heel_marker_idx,
            target=0,
        )

        # Non-slipping constraints
        constraints.add(  # On only one of the feet
            ConstraintFcn.NON_SLIPPING,
            phase=p,
            node=control_nodes,
            tangential_component_idx=toe_non_slipping[0],
            normal_component_idx=toe_non_slipping[1],
            static_friction_coefficient=static_friction_coefficient,
        )

    constraints.add(
        ConstraintFcn.TRACK_MARKERS,
        phase=2,
        index=2,
        node=Node.END,
        min_bound=-0.001,
        max_bound=0.001,
        marker_index=toe_marker_idx,
        target=0,
    )
    constraints.add(
            ConstraintFcn.TRACK_MARKERS,
            phase=3,
            index=2,
            node=Node.END,
            min_bound=-0.001,
            max_bound=0.001,
            marker_index=heel_marker_idx,
            target=0,
        )

    # Target the final pose (except for translation)
    trans_root = bio_models[nb_phases - 1].model.segment(0).nbDofTrans()
    constraints.add(
        ConstraintFcn.TRACK_STATE,
        key="q",
        node=Node.END,
        phase=nb_phases - 1,
        index=range(trans_root, nb_q),
        target=initial_states[trans_root:nb_q, :],
        min_bound=-0.1,
        max_bound=0.1,
    )
    constraints.add(
        ConstraintFcn.TRACK_STATE,
        key="qdot",
        node=Node.END,
        phase=nb_phases - 1,
        target=initial_velocity,
        min_bound=-0.1,
        max_bound=0.1,
    )

    # Objectives
    objective_functions = ObjectiveList()

    # Maximize the jump height
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_PREDICTED_COM_HEIGHT, weight=-100, phase=1)

    # Minimize the tau on root if present
    for p in range(nb_phases):
        root = [i for i in dof_mappings["tau"].to_second.map_idx[:n_root] if i is not None]
        if root:
            objective_functions.add(
                ObjectiveFcn.Lagrange.MINIMIZE_CONTROL,
                key="tau",
                weight=0.1,
                phase=p,
                index=root,
            )

    # Minimize unnecessary acceleration during for the aerial and reception phases
    for p in range(nb_phases):
        if control_type == ControlType.LINEAR_CONTINUOUS:
            objective_functions.add(
                ObjectiveFcn.Lagrange.MINIMIZE_CONTROL,
                key="tau",
                weight=0.1,
                derivative=True,
                phase=p,
            )
    for p in range(2, nb_phases):
        objective_functions.add(
            ObjectiveFcn.Lagrange.MINIMIZE_STATE,
            key="qdot",
            weight=0.1,
            derivative=True,
            phase=p,
        )

    # Minimize time of the phase
    for i in range(nb_phases):
        if time_min[i] != time_max[i]:
            objective_functions.add(
                ObjectiveFcn.Mayer.MINIMIZE_TIME,
                weight=0.1,
                phase=i,
                min_bound=time_min[i],
                max_bound=time_max[i],
            )

    # Path constraints
    x_bounds = BoundsList()
    u_bounds = BoundsList()
    for p in range(nb_phases):
        x_bounds.add("q", min_bound=bio_models[0].bounds_from_ranges("q", mapping=dof_mappings).min,
                        max_bound=bio_models[0].bounds_from_ranges("q", mapping=dof_mappings).max,
                     phase=p)
        x_bounds.add("qdot", min_bound=bio_models[0].bounds_from_ranges("qdot", mapping=dof_mappings).min,
                        max_bound=bio_models[0].bounds_from_ranges("qdot", mapping=dof_mappings).max,
                     phase=p)
        if p == 3 or p == 4:
            # Allow greater speed in passive reception
            x_bounds[p]["qdot"].max[heel_dof, :] *= 2
        u_bounds.add("tau", min_bound=[-tau_constant_bound] * nb_tau, max_bound=[tau_constant_bound] * nb_tau, phase=p)

    # Enforce the initial pose and velocity
    x_bounds[0]["q"].min[:, 0] = initial_pose[:, 0]
    x_bounds[0]["qdot"][:, 0] = initial_velocity[:, 0]

    # Allow for passive velocity at reception
    if nb_phases >= 4:
        x_bounds[3]["qdot"].min[:, 0] = 2 * x_bounds[3]["qdot"].min[:, 0]
        x_bounds[3]["qdot"].max[:, 0] = 2 * x_bounds[3]["qdot"].max[:, 0]
        x_bounds[4]["qdot"].min[:, 0] = 2 * x_bounds[4]["qdot"].min[:, 0]
        x_bounds[4]["qdot"].max[:, 0] = 2 * x_bounds[4]["qdot"].max[:, 0]

    # Phase transitions
    phase_transitions = PhaseTransitionList()
    phase_transitions.add(PhaseTransitionFcn.CONTINUOUS, phase_pre_idx=0)
    phase_transitions.add(PhaseTransitionFcn.CONTINUOUS, phase_pre_idx=1)
    phase_transitions.add(PhaseTransitionFcn.IMPACT, phase_pre_idx=2)
    phase_transitions.add(PhaseTransitionFcn.IMPACT, phase_pre_idx=3)

    # Initial guesses
    x_init = InitialGuessList()
    u_init = InitialGuessList()
    for i in range(nb_phases):
        x_init.add("q", initial_guess=initial_states)
        u_init.add("tau", initial_guess=[0] * nb_tau)

    return OptimalControlProgram(
        bio_models,
        dynamics,
        n_shooting,
        phase_time=final_time,
        x_init=x_init,
        x_bounds=x_bounds,
        u_init=u_init,
        u_bounds=u_bounds,
        objective_functions=objective_functions,
        constraints=constraints,
        variable_mappings=dof_mappings,
        phase_transitions=phase_transitions,
        n_threads=n_threads,
        control_type=control_type,
        ode_solver=ode_solver,
        use_sx=use_sx,
    )


def main():
    """
    Solve the OCP and animate the solution.
    """

    biorbd_model_paths = ["models/jumper2contacts.bioMod",
                          "models/jumper1contacts.bioMod",
                          "models/jumper1contacts.bioMod",
                          "models/jumper1contacts.bioMod",
                          "models/jumper2contacts.bioMod"
                          ]
    bio_models = [BiorbdModel(path) for path in biorbd_model_paths]

    solver = Solver.IPOPT(show_online_optim=platform.system() == "Linux")

    phase_time = [0.3, 0.2, 0.6, 0.2, 0.2]
    time_min = [0.2, 0.05, 0.6, 0.05, 0.1]
    time_max = [0.5, 0.5, 2.0, 0.5, 0.5]
    number_shooting_points = [30, 15, 20, 30, 30]

    # --- Prepare the ocp --- #
    ocp = prepare_ocp(
        bio_models,
        phase_time,
        time_min,
        time_max,
        number_shooting_points,
        n_threads=4,
        ode_solver=OdeSolver.RK4(),
        use_sx=False,
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
