"""
Probleme temporaire pour montrer le ROCP sur des mouvements de barre fixe (Tkatchev).
À ajouter un bon modèle de gymnaste.
Idéalement pouvoir comparer notre méthode avec celle de https://www.sciencedirect.com/science/article/pii/S0167945711001680?casa_token=76oAyGt3Zs4AAAAA:5Xo1zlcjC8hXwz80JuF7E5vfmx-nUxjKIkXjVixY50nE0LmwU7f3Q5ywiJWU6MbkyVw6qoaJ
En temre de techniques et de temps de calcul.
"""

from casadi import DM, MX, vertcat, sum2
import numpy as np
import biorbd_casadi as biorbd
from bioptim import (
    OptimalControlProgram,
    DynamicsList,
    DynamicsFcn,
    ObjectiveList,
    ConstraintList,
    ConstraintFcn,
    BoundsList,
    InitialGuessList,
    Node,
    QAndQDotBounds,
    ObjectiveFcn,
    BiMappingList,
    Axis,
    PhaseTransitionList,
    PhaseTransitionFcn,
    BiMapping,
    CostType,
    Solver,
    DynamicsFunctions,
    ConfigureProblem,
    BiorbdInterface,
)


def custom_robust_objective_MINIMIZE_COM_POSITION(all_pn, axes, nb_random, size_random):
    """
    Maximiser la hauteur moyenne en présence de bruit  sur les U
    """
    # voir si on tient a connecter les phases entre elles
    # voir si on veut permettre du bruit en x et comment

    # Est-ce qu'il faut proceder en MS ou SS ici?
    # - si MS, en Mayer on peut calculer seulement les noeuds pertinents, mais ca enleve toute la pertinance d'accumuler les erreurs :(
    # - si SS, depuis le refactor, on n'a plus acces a tous les noeuds, donc pas vraiment possible (choix logique puisque en MS on ne veut pas faire de lien entre les shooting nodes)

    for i in range(len(all_pn)):

        q_optim = all_pn[i].nlp.states["q"].cx  # [0]
        qdot_optim = all_pn[i].nlp.states["qdot"].cx  # [0]
        controls_optim = all_pn[i].nlp.controls["tau"].cx  # [0]

        states_perturbed = [vertcat(q_optim, qdot_optim)]
        controls_perturbed = [controls_optim]

        states_rand = DM.rand(len(all_pn[i].nlp.states["all"]), nb_random) * \
                      size_random[all_pn[i].nlp.phase_idx]["states"] - \
                      size_random[all_pn[i].nlp.phase_idx]["states"] / 2
        controls_rand = DM.rand(len(all_pn[i].nlp.controls["all"]), nb_random) * \
                        size_random[all_pn[i].nlp.phase_idx]["controls"] - \
                        size_random[all_pn[i].nlp.phase_idx]["controls"]/2

        next_x, _ = all_pn[i].nlp.dynamics[0].function(vertcat(q_optim, qdot_optim) + states_rand,
                                                    controls_optim + controls_rand,
                                                    all_pn[i].nlp.parameters.cx)

        CoM = MX.zeros(3, nb_random)
        mean_com = MX.zeros(3, all_pn[i].nlp.ns)
        std_com = MX.zeros(3, all_pn[i].nlp.ns)

        for j in range(1, all_pn[i].nlp.ns):

            states_optim = next_x
            controls_optim = all_pn[i].nlp.controls["tau"].cx  # [j] !!!!!!

            states_perturbed += [states_optim]
            controls_perturbed += [controls_optim]

            states_rand = DM.rand(len(all_pn[i].nlp.states["all"]), nb_random) * \
                          size_random[all_pn[i].nlp.phase_idx]["states"] - \
                          size_random[all_pn[i].nlp.phase_idx]["states"] / 2
            controls_rand = DM.rand(len(all_pn[i].nlp.controls["all"]), nb_random) * \
                            size_random[all_pn[i].nlp.phase_idx]["controls"] - \
                            size_random[all_pn[i].nlp.phase_idx]["controls"] / 2

            next_x, _ = all_pn[i].nlp.dynamics[j].function(states_optim + states_rand, controls_optim + controls_rand, all_pn[i].nlp.parameters.cx)

            for k in range(nb_random):
                CoM[:, k] = all_pn[i].nlp.model.CoM(next_x[:all_pn[i].nlp.model.nbQ(), :]).to_mx()

            mean_com[:, j] = sum2(CoM) / nb_random
            std_com[:, j] = sum2(CoM - mean_com)**2/ np.sqrt(nb_random)  # discrepency metric, not really std to save a sqrt

    # comme dans notre cas c'est un mayer, je vais sélectionner le dernier noeud
    mean_com_cx = BiorbdInterface.mx_to_cx("mean_com", mean_com[axes, -1], all_pn.nlp.states["q"], all_pn.nlp.states["qdot"], all_pn.nlp.controls["tau"])
    std_com_cx = BiorbdInterface.mx_to_cx("std_com", std_com[axes, -1], all_pn.nlp.states["q"], all_pn.nlp.states["qdot"], all_pn.nlp.controls["tau"])

    return mean_com_cx, std_com_cx


def prepare_rocp(
    biorbd_model_path: str = "models/double_pendulum.bioMod",
    biorbd_model_path_withTranslations: str = "models/double_pendulum_with_translations.bioMod",
) -> OptimalControlProgram:

    biorbd_model = (biorbd.Model(biorbd_model_path), biorbd.Model(biorbd_model_path_withTranslations), biorbd.Model(biorbd_model_path_withTranslations))

    # Problem parameters
    n_shooting = (40, 40, 40)
    final_time = (1, 1, 1)
    tau_min, tau_max, tau_init = -200, 200, 0

    # Mapping
    tau_mappings = BiMappingList()
    tau_mappings.add("tau", [None, 0], [1], phase=0)
    tau_mappings.add("tau", [None, None, None, 0], [3], phase=1)
    tau_mappings.add("tau", [None, None, None, 0], [3], phase=2)

    # Phase transition
    phase_transitions = PhaseTransitionList()
    phase_transitions.add(
        PhaseTransitionFcn.CONTINUOUS, phase_pre_idx=0, states_mapping=BiMapping([0, 1, 2, 3], [2, 3, 6, 7])
    )

    # Random-Robust specifications
    size_random = [{
        "states": np.ones(biorbd_model[i].nbQ() + biorbd_model[i].nbQdot()) * 0, # * 5 * np.pi/180
        "controls": np.ones(len(tau_mappings[i]["tau"].to_first.map_idx)) * 1,
    } for i in range(len(biorbd_model))]

    # Add objective functions
    objective_functions = ObjectiveList()
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=1, phase=0)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=1, phase=1)
    # objective_functions.add(
    #     ObjectiveFcn.Mayer.MINIMIZE_COM_POSITION, node=Node.END, weight=-1000, axes=Axis.Z, phase=1, quadratic=False
    # ) #robustifier
    objective_functions.add(
        custom_robust_objective_MINIMIZE_COM_POSITION,
        custom_type=ObjectiveFcn.Mayer,
        node=Node.END,
        quadratic=False,
        weight=-1000,
        phase=1,
        axes=Axis.Z,
        nb_random=100,
        size_random=size_random,
    )

    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_COM_POSITION, node=Node.END, weight=-100, axes=Axis.Y, phase=1) #robustifier

    # Constraints
    constraints = ConstraintList()
    constraints.add(ConstraintFcn.TIME_CONSTRAINT, node=Node.END, min_bound=0.1, max_bound=2, phase=0)
    constraints.add(ConstraintFcn.TIME_CONSTRAINT, node=Node.END, min_bound=0.1, max_bound=2, phase=1)
    constraints.add(ConstraintFcn.TIME_CONSTRAINT, node=Node.END, min_bound=0.1, max_bound=2, phase=2)
    constraints.add(ConstraintFcn.TRACK_COM_POSITION, node=Node.END, axes=Axis.Z, phase=1, quadratic=False, min_bound=0, max_bound=2)
    constraints.add(ConstraintFcn.SUPERIMPOSE_MARKERS, node=Node.END, first_marker="bar", second_marker="marker_1", phase=2) #robustifier

    # Dynamics
    dynamics = DynamicsList()
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN, with_contact=False)
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN, with_contact=False)
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN, with_contact=False)

    # Path constraint
    x_bounds = BoundsList()
    x_bounds.add(bounds=QAndQDotBounds(biorbd_model[0]))
    x_bounds.add(bounds=QAndQDotBounds(biorbd_model[1]))
    x_bounds.add(bounds=QAndQDotBounds(biorbd_model[2]))

    # Phase 0
    x_bounds[0].min[0, :] = np.pi
    x_bounds[0][1, 0] = 0
    x_bounds[0][0, 0] = np.pi
    x_bounds[0].min[0, -1] = 2 * np.pi

    # Phase 1
    x_bounds[1].min[2, :] = np.pi
    x_bounds[1][[0, 1, 4, 5], 0] = 0
    # x_bounds[1].min[2, -1] = np.pi
    # x_bounds[1].max[2, -1] = 2 * np.pi

    # Phase 2
    x_bounds[2].min[2, -1] = 5/4 * np.pi
    x_bounds[2].max[2, -1] = 2 * np.pi
    # x_bounds[2].min[3, -1] = 0
    # je pourrais ajouter des contraintes pour les mains sur la barre ici a la place au besoin

    # Initial guess
    x_init = InitialGuessList()
    x_init.add([0] * (biorbd_model[0].nbQ() + biorbd_model[0].nbQdot()))
    x_init.add([0] * (biorbd_model[1].nbQ() + biorbd_model[1].nbQdot()))
    x_init.add([0] * (biorbd_model[2].nbQ() + biorbd_model[2].nbQdot()))

    # Define control path constraint
    u_bounds = BoundsList()
    u_bounds.add([tau_min] * len(tau_mappings[0]["tau"].to_first), [tau_max] * len(tau_mappings[0]["tau"].to_first))
    u_bounds.add([tau_min] * len(tau_mappings[1]["tau"].to_first), [tau_max] * len(tau_mappings[1]["tau"].to_first))
    u_bounds.add([tau_min] * len(tau_mappings[2]["tau"].to_first), [tau_max] * len(tau_mappings[2]["tau"].to_first))

    # Control initial guess
    u_init = InitialGuessList()
    u_init.add([tau_init] * len(tau_mappings[0]["tau"].to_first))
    u_init.add([tau_init] * len(tau_mappings[1]["tau"].to_first))
    u_init.add([tau_init] * len(tau_mappings[2]["tau"].to_first))

    return OptimalControlProgram(
        biorbd_model,
        dynamics,
        n_shooting,
        final_time,
        x_init=x_init,
        u_init=u_init,
        x_bounds=x_bounds,
        u_bounds=u_bounds,
        objective_functions=objective_functions,
        constraints=constraints,
        variable_mappings=tau_mappings,
        phase_transitions=phase_transitions,
    )


def prepare_ocp(
    biorbd_model_path: str = "models/double_pendulum.bioMod",
    biorbd_model_path_withTranslations: str = "models/double_pendulum_with_translations.bioMod",
) -> OptimalControlProgram:

    biorbd_model = (biorbd.Model(biorbd_model_path), biorbd.Model(biorbd_model_path_withTranslations), biorbd.Model(biorbd_model_path_withTranslations))

    # Problem parameters
    n_shooting = (40, 40, 40)
    final_time = (1, 1, 1)
    tau_min, tau_max, tau_init = -200, 200, 0

    # Mapping
    tau_mappings = BiMappingList()
    tau_mappings.add("tau", [None, 0], [1], phase=0)
    tau_mappings.add("tau", [None, None, None, 0], [3], phase=1)
    tau_mappings.add("tau", [None, None, None, 0], [3], phase=2)

    # Add objective functions
    objective_functions = ObjectiveList()
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=1, phase=0)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=1, phase=1)
    objective_functions.add(
        ObjectiveFcn.Mayer.MINIMIZE_COM_POSITION, node=Node.END, weight=-1000, axes=Axis.Z, phase=1, quadratic=False
    )
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_COM_POSITION, node=Node.END, weight=-100, axes=Axis.Y, phase=1)

    # Constraints
    constraints = ConstraintList()
    constraints.add(ConstraintFcn.TIME_CONSTRAINT, node=Node.END, min_bound=0.1, max_bound=2, phase=0)
    constraints.add(ConstraintFcn.TIME_CONSTRAINT, node=Node.END, min_bound=0.1, max_bound=2, phase=1)
    constraints.add(ConstraintFcn.TIME_CONSTRAINT, node=Node.END, min_bound=0.1, max_bound=2, phase=2)
    constraints.add(ConstraintFcn.TRACK_COM_POSITION, node=Node.END, axes=Axis.Z, phase=1, quadratic=False, min_bound=0, max_bound=2)
    constraints.add(ConstraintFcn.SUPERIMPOSE_MARKERS, node=Node.END, first_marker="bar", second_marker="marker_1", phase=2)

    # Dynamics
    dynamics = DynamicsList()
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN, with_contact=False)
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN, with_contact=False)
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN, with_contact=False)

    # Path constraint
    x_bounds = BoundsList()
    x_bounds.add(bounds=QAndQDotBounds(biorbd_model[0]))
    x_bounds.add(bounds=QAndQDotBounds(biorbd_model[1]))
    x_bounds.add(bounds=QAndQDotBounds(biorbd_model[2]))

    # Phase 0
    x_bounds[0].min[0, :] = np.pi
    x_bounds[0][1, 0] = 0
    x_bounds[0][0, 0] = np.pi
    x_bounds[0].min[0, -1] = 2 * np.pi

    # Phase 1
    x_bounds[1].min[2, :] = np.pi
    x_bounds[1][[0, 1, 4, 5], 0] = 0
    # x_bounds[1].min[2, -1] = np.pi
    # x_bounds[1].max[2, -1] = 2 * np.pi

    # Phase 2
    x_bounds[2].min[2, -1] = 5/4 * np.pi
    x_bounds[2].max[2, -1] = 2 * np.pi
    # x_bounds[2].min[3, -1] = 0
    # je pourrais ajouter des contraintes pour les mains sur la barre ici a la place au besoin

    # Initial guess
    x_init = InitialGuessList()
    x_init.add([0] * (biorbd_model[0].nbQ() + biorbd_model[0].nbQdot()))
    x_init.add([0] * (biorbd_model[1].nbQ() + biorbd_model[1].nbQdot()))
    x_init.add([0] * (biorbd_model[2].nbQ() + biorbd_model[2].nbQdot()))

    # Define control path constraint
    u_bounds = BoundsList()
    u_bounds.add([tau_min] * len(tau_mappings[0]["tau"].to_first), [tau_max] * len(tau_mappings[0]["tau"].to_first))
    u_bounds.add([tau_min] * len(tau_mappings[1]["tau"].to_first), [tau_max] * len(tau_mappings[1]["tau"].to_first))
    u_bounds.add([tau_min] * len(tau_mappings[2]["tau"].to_first), [tau_max] * len(tau_mappings[2]["tau"].to_first))

    # Control initial guess
    u_init = InitialGuessList()
    u_init.add([tau_init] * len(tau_mappings[0]["tau"].to_first))
    u_init.add([tau_init] * len(tau_mappings[1]["tau"].to_first))
    u_init.add([tau_init] * len(tau_mappings[2]["tau"].to_first))

    phase_transitions = PhaseTransitionList()
    phase_transitions.add(
        PhaseTransitionFcn.CONTINUOUS, phase_pre_idx=0, states_mapping=BiMapping([0, 1, 2, 3], [2, 3, 6, 7])
    )

    return OptimalControlProgram(
        biorbd_model,
        dynamics,
        n_shooting,
        final_time,
        x_init=x_init,
        u_init=u_init,
        x_bounds=x_bounds,
        u_bounds=u_bounds,
        objective_functions=objective_functions,
        constraints=constraints,
        variable_mappings=tau_mappings,
        phase_transitions=phase_transitions,
    )

def main():

    # --- Prepare the ocp --- #
    ocp = prepare_ocp()
    ocp.add_plot_penalty(CostType.ALL)
    sol = ocp.solve(Solver.IPOPT()) # show_online_optim=True
    sol.animate()

    # --- Prepare the rocp --- #
    rocp = prepare_rocp()
    rocp.add_plot_penalty(CostType.ALL)
    sol = rocp.solve(Solver.IPOPT()) # show_online_optim=True
    sol.animate()

if __name__ == "__main__":
    main()
