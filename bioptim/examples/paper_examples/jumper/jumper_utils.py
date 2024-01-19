import biorbd_casadi as biorbd
from scipy import optimize
import numpy as np
from casadi import MX
from bioptim import BiMapping, OdeSolver


def find_initial_root_pose(model, initial_pose):
    flatfoot_contact_z_idx = (0, 2, 3, 5)  # Contacts indices of heel and toe in bioMod 2 contacts

    # This method finds a root pose such that the body of a given pose has its CoM centered to the feet
    n_root = model.nb_root

    body_pose_no_root = initial_pose[n_root:, 0]
    bimap = BiMapping(list(range(n_root)) + [None] * body_pose_no_root.shape[0], list(range(n_root)))

    bound_min = []
    bound_max = []
    for i in range(model.nb_segments):
        seg = model.model.segment(i)
        for r in seg.QRanges():
            bound_min.append(r.min())
            bound_max.append(r.max())
    bound_min = bimap.to_first.map(np.array(bound_min)[:, np.newaxis])
    bound_max = bimap.to_first.map(np.array(bound_max)[:, np.newaxis])
    root_bounds = (list(bound_min[:, 0]), list(bound_max[:, 0]))

    q_sym = MX.sym("Q", model.nb_q, 1)
    com_func = biorbd.to_casadi_func("com", model.model.CoM, q_sym)
    contacts_func = biorbd.to_casadi_func("contacts", model.model.constraintsInGlobal, q_sym, True)
    shoulder_jcs_func = biorbd.to_casadi_func("shoulder_jcs", model.model.globalJCS, q_sym, 3)
    hand_marker_func = biorbd.to_casadi_func("hand_marker", model.model.marker, q_sym, 32)

    def objective_function(q_root, *args, **kwargs):
        # Center of mass
        q = bimap.to_second.map(q_root[:, np.newaxis])[:, 0]
        q[model.nb_root:] = body_pose_no_root
        com = np.array(com_func(q))
        contacts = np.array(contacts_func(q))
        mean_contacts = np.mean(contacts, axis=1)
        shoulder_jcs = np.array(shoulder_jcs_func(q))
        hand = np.array(hand_marker_func(q))

        # Prepare output
        out = np.ndarray((0,))

        # The center of contact points should be at 0
        out = np.concatenate((out, mean_contacts[0:2]))
        out = np.concatenate((out, contacts[2, flatfoot_contact_z_idx]))

        # The projection of the center of mass should be at 0 and at 0.95 meter high
        out = np.concatenate((out, (com + np.array([[0, 0, -0.95]]).T)[:, 0]))

        # Keep the arms horizontal
        out = np.concatenate((out, (shoulder_jcs[2, 3] - hand[2])))

        return out

    q_root0 = np.mean(root_bounds, axis=0)
    pos = optimize.least_squares(objective_function, x0=q_root0, bounds=root_bounds)
    root = np.zeros(n_root)
    root[bimap.to_first.map_idx] = pos.x
    return root
