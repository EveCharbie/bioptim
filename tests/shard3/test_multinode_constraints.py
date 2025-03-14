import re

import pytest
from bioptim import (
    BiorbdModel,
    MultinodeConstraintList,
    MultinodeConstraintFcn,
    Node,
    OdeSolver,
    OptimalControlProgram,
    DynamicsList,
    DynamicsFcn,
    ObjectiveList,
    BoundsList,
    PhaseDynamics,
)
from tests.utils import TestUtils


def prepare_ocp(biorbd_model_path, phase_1, phase_2, phase_dynamics) -> OptimalControlProgram:
    bio_model = (BiorbdModel(biorbd_model_path), BiorbdModel(biorbd_model_path), BiorbdModel(biorbd_model_path))

    # Problem parameters
    n_shooting = (100, 300, 100)
    final_time = (2, 5, 4)
    tau_min, tau_max, tau_init = -100, 100, 0

    # Add objective functions
    objective_functions = ObjectiveList()

    # Dynamics
    dynamics = DynamicsList()
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN, expand_dynamics=True, phase_dynamics=phase_dynamics)
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN, expand_dynamics=True, phase_dynamics=phase_dynamics)
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN, expand_dynamics=True, phase_dynamics=phase_dynamics)

    multinode_constraints = MultinodeConstraintList()
    # hard constraint
    multinode_constraints.add(
        MultinodeConstraintFcn.STATES_EQUALITY,
        nodes_phase=(phase_1, phase_2),
        nodes=(Node.START, Node.START),
        sub_nodes=(0, 0),
    )
    multinode_constraints.add(
        MultinodeConstraintFcn.COM_EQUALITY,
        nodes_phase=(phase_1, phase_2),
        nodes=(Node.START, Node.START),
        sub_nodes=(0, 0),
    )
    multinode_constraints.add(
        MultinodeConstraintFcn.COM_VELOCITY_EQUALITY,
        nodes_phase=(phase_1, phase_2),
        nodes=(Node.START, Node.START),
        sub_nodes=(0, 0),
    )

    # Path constraint
    x_bounds = BoundsList()
    x_bounds.add("q", bounds=bio_model[0].bounds_from_ranges("q"), phase=0)
    x_bounds.add("qdot", bounds=bio_model[0].bounds_from_ranges("qdot"), phase=0)
    x_bounds.add("q", bounds=bio_model[1].bounds_from_ranges("q"), phase=1)
    x_bounds.add("qdot", bounds=bio_model[1].bounds_from_ranges("qdot"), phase=1)
    x_bounds.add("q", bounds=bio_model[2].bounds_from_ranges("q"), phase=2)
    x_bounds.add("qdot", bounds=bio_model[2].bounds_from_ranges("qdot"), phase=2)

    for bounds in x_bounds:
        bounds["q"][1, [0, -1]] = 0
        bounds["qdot"][:, [0, -1]] = 0
    x_bounds[0]["q"][2, 0] = 0.0
    x_bounds[2]["q"][2, [0, -1]] = [0.0, 1.57]

    # Define control path constraint
    u_bounds = BoundsList()
    u_bounds.add("tau", min_bound=[tau_min] * bio_model[0].nb_tau, max_bound=[tau_max] * bio_model[0].nb_tau, phase=0)
    u_bounds.add("tau", min_bound=[tau_min] * bio_model[1].nb_tau, max_bound=[tau_max] * bio_model[1].nb_tau, phase=1)
    u_bounds.add("tau", min_bound=[tau_min] * bio_model[2].nb_tau, max_bound=[tau_max] * bio_model[2].nb_tau, phase=2)

    return OptimalControlProgram(
        bio_model,
        dynamics,
        n_shooting,
        final_time,
        x_bounds=x_bounds,
        u_bounds=u_bounds,
        objective_functions=objective_functions,
        multinode_constraints=multinode_constraints,
        ode_solver=OdeSolver.RK4(),
    )


@pytest.mark.parametrize("node", [*Node, 0])
def test_multinode_fail_first_node(node):
    # Constraints
    multinode_constraints = MultinodeConstraintList()
    # hard constraint
    if node in [Node.START, Node.MID, Node.PENULTIMATE, Node.END, 0]:
        multinode_constraints.add(
            MultinodeConstraintFcn.STATES_EQUALITY,
            nodes_phase=(0, 2),
            nodes=(node, Node.START),
            sub_nodes=(0, 0),
        )
    else:
        with pytest.raises(
            ValueError,
            match=re.escape(
                "Multinode penalties only works with Node.START, Node.MID, Node.PENULTIMATE, Node.END or a node index (int)."
            ),
        ):
            multinode_constraints.add(
                MultinodeConstraintFcn.STATES_EQUALITY,
                nodes_phase=(0, 2),
                nodes=(node, Node.START),
                sub_nodes=(0, 0),
            )


@pytest.mark.parametrize("node", [*Node, 0])
def test_multinode_fail_second_node(node):
    # Constraints
    multinode_constraints = MultinodeConstraintList()
    # hard constraint
    if node in [Node.START, Node.MID, Node.PENULTIMATE, Node.END, 0]:
        multinode_constraints.add(
            MultinodeConstraintFcn.STATES_EQUALITY,
            nodes_phase=(0, 2),
            nodes=(node, Node.START),
            sub_nodes=(0, 0),
        )
    else:
        with pytest.raises(
            ValueError,
            match=re.escape(
                "Multinode penalties only works with Node.START, Node.MID, Node.PENULTIMATE, Node.END or a node index (int)."
            ),
        ):
            multinode_constraints.add(
                MultinodeConstraintFcn.STATES_EQUALITY,
                nodes_phase=(0, 2),
                nodes=(Node.START, node),
                sub_nodes=(0, 0),
            )
