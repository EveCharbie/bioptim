import pytest

from casadi import DM, Function
import numpy as np
import biorbd
from bioptim import (
    OptimalControlProgram,
    DynamicsList,
    DynamicsFcn,
    Bounds,
    InitialGuess,
    Objective,
    ObjectiveFcn,
    Axis,
    ConstraintFcn,
    Constraint,
    Node,
)
from bioptim.interfaces.ipopt_interface import IpoptInterface
from bioptim.limits.penalty_node import PenaltyNodeList
from bioptim.limits.penalty import PenaltyFunctionAbstract, PenaltyOption
from bioptim.optimization.non_linear_program import NonLinearProgram as nlp
from .utils import TestUtils


def prepare_test_ocp(with_muscles=False, with_contact=False, with_actuator=False):
    bioptim_folder = TestUtils.bioptim_folder()
    if with_muscles and with_contact or with_muscles and with_actuator or with_contact and with_actuator:
        raise RuntimeError("With muscles and with contact and with_actuator together is not defined")
    elif with_muscles:
        biorbd_model = biorbd.Model(bioptim_folder + "/examples/muscle_driven_ocp/arm26.bioMod")
        dynamics = DynamicsList()
        dynamics.add(DynamicsFcn.MUSCLE_DRIVEN, with_residual_torque=True)
        nx = biorbd_model.nbQ() + biorbd_model.nbQdot()
        nu = biorbd_model.nbGeneralizedTorque() + biorbd_model.nbMuscles()
    elif with_contact:
        biorbd_model = biorbd.Model(
            bioptim_folder + "/examples/muscle_driven_with_contact/2segments_4dof_2contacts_1muscle.bioMod"
        )
        dynamics = DynamicsList()
        dynamics.add(DynamicsFcn.TORQUE_DRIVEN, with_contact=True)
        nx = biorbd_model.nbQ() + biorbd_model.nbQdot()
        nu = biorbd_model.nbGeneralizedTorque()
    elif with_actuator:
        biorbd_model = biorbd.Model(bioptim_folder + "/examples/torque_driven_ocp/cube.bioMod")
        dynamics = DynamicsList()
        dynamics.add(DynamicsFcn.TORQUE_DRIVEN)
        nx = biorbd_model.nbQ() + biorbd_model.nbQdot()
        nu = biorbd_model.nbGeneralizedTorque()
    else:
        biorbd_model = biorbd.Model(bioptim_folder + "/examples/track/cube_and_line.bioMod")
        dynamics = DynamicsList()
        dynamics.add(DynamicsFcn.TORQUE_DRIVEN)
        nx = biorbd_model.nbQ() + biorbd_model.nbQdot()
        nu = biorbd_model.nbGeneralizedTorque()
    x_init = InitialGuess(np.zeros((nx, 1)))
    u_init = InitialGuess(np.zeros((nu, 1)))
    x_bounds = Bounds(np.zeros((nx, 1)), np.zeros((nx, 1)))
    u_bounds = Bounds(np.zeros((nu, 1)), np.zeros((nu, 1)))
    ocp = OptimalControlProgram(biorbd_model, dynamics, 10, 1.0, x_init, u_init, x_bounds, u_bounds, use_sx=True)
    ocp.nlp[0].J = [[]]
    ocp.nlp[0].g = [[]]
    return ocp


def test_penalty_targets_shapes():
    p = ObjectiveFcn.Parameter
    np.testing.assert_equal(Objective([], custom_type=p, target=1).target.shape, (1, 1))
    np.testing.assert_equal(Objective([], custom_type=p, target=np.array(1)).target.shape, (1, 1))
    np.testing.assert_equal(Objective([], custom_type=p, target=[1]).target.shape, (1, 1))
    np.testing.assert_equal(Objective([], custom_type=p, target=[1, 2]).target.shape, (2, 1))
    np.testing.assert_equal(Objective([], custom_type=p, target=[[1], [2]]).target.shape, (2, 1))
    np.testing.assert_equal(Objective([], custom_type=p, target=[[1, 2]]).target.shape, (1, 2))
    np.testing.assert_equal(Objective([], custom_type=p, target=np.array([[1, 2]])).target.shape, (1, 2))


@pytest.mark.parametrize("penalty_origin", [ObjectiveFcn.Lagrange, ObjectiveFcn.Mayer])
@pytest.mark.parametrize("value", [0.1, -10])
def test_penalty_minimize_time(penalty_origin, value):
    ocp = prepare_test_ocp()
    penalty_type = penalty_origin.MINIMIZE_TIME
    penalty = Objective(penalty_type)
    penalty_type.value[0](penalty, PenaltyNodeList(ocp, ocp.nlp[0], [], [], [], []))

    np.testing.assert_almost_equal(
        ocp.nlp[0].J[0][0]["val"],
        np.array(1),
    )


@pytest.mark.parametrize("penalty_origin", [ObjectiveFcn.Lagrange, ObjectiveFcn.Mayer])
@pytest.mark.parametrize("value", [0.1, -10])
def test_penalty_minimize_state(penalty_origin, value):
    ocp = prepare_test_ocp()
    t = [0]
    x = [DM.ones((12, 1)) * value]
    u = [0]
    penalty_type = penalty_origin.MINIMIZE_STATE
    penalty = Objective(penalty_type)
    penalty_type.value[0](penalty, PenaltyNodeList(ocp, ocp.nlp[0], t, x, u, []))

    np.testing.assert_almost_equal(
        ocp.nlp[0].J[0][0]["val"],
        np.array([[value]] * 8),
    )


@pytest.mark.parametrize("penalty_origin", [ObjectiveFcn.Lagrange, ObjectiveFcn.Mayer, ConstraintFcn])
@pytest.mark.parametrize("value", [0.1, -10])
def test_penalty_minimize_qddot(penalty_origin, value):
    ocp = prepare_test_ocp()
    t = [0, 1]
    x = [DM.ones((8, 1)) * value, DM.ones((8, 1)) * value]
    u = [DM.ones((4, 1)) * value]
    if penalty_origin == ObjectiveFcn.Mayer or penalty_origin == ConstraintFcn:
        with pytest.raises(AttributeError, match="MINIMIZE_QDDOT"):
            _ = penalty_origin.MINIMIZE_QDDOT
        return
    else:
        penalty_type = penalty_origin.MINIMIZE_QDDOT
    penalty = Objective(penalty_type)
    penalty_type.value[0](penalty, PenaltyNodeList(ocp, ocp.nlp[0], t, x, u, []))

    np.testing.assert_almost_equal(
        ocp.nlp[0].J[0][0]["val"].T,
        [[value, -9.81 + value, value, value]],
    )


@pytest.mark.parametrize("penalty_origin", [ObjectiveFcn.Lagrange, ObjectiveFcn.Mayer, ConstraintFcn])
@pytest.mark.parametrize("value", [0.1, -10])
def test_penalty_track_state(penalty_origin, value):
    ocp = prepare_test_ocp()
    x = [DM.ones((12, 1)) * value]
    penalty_type = penalty_origin.TRACK_STATE
    if isinstance(penalty_type, (ObjectiveFcn.Lagrange, ObjectiveFcn.Mayer)):
        penalty = Objective(penalty_type, target=np.ones((8, 1)) * value)
    else:
        penalty = Constraint(penalty_type, target=np.ones((8, 1)) * value)
    penalty_type.value[0](penalty, PenaltyNodeList(ocp, ocp.nlp[0], [1], x, [], []))

    if isinstance(penalty_type, (ObjectiveFcn.Lagrange, ObjectiveFcn.Mayer)):
        res = ocp.nlp[0].J[0][0]["val"]
    else:
        res = ocp.nlp[0].g[0][0]["val"]

    expected = np.array([[value]] * 8)

    np.testing.assert_almost_equal(
        res,
        expected,
    )

    if isinstance(penalty_type, ConstraintFcn):
        np.testing.assert_almost_equal(ocp.nlp[0].g[0][0]["bounds"].min, np.array([[0]] * 8))
        np.testing.assert_almost_equal(
            ocp.nlp[0].g[0][0]["bounds"].max,
            np.array([[0]] * 8),
        )


@pytest.mark.parametrize("penalty_origin", [ObjectiveFcn.Lagrange, ObjectiveFcn.Mayer])
@pytest.mark.parametrize("value", [0.1, -10])
def test_penalty_minimize_markers(penalty_origin, value):
    ocp = prepare_test_ocp()
    t = [0]
    x = [DM.ones((8, 1)) * value]
    u = [0]
    penalty_type = penalty_origin.MINIMIZE_MARKERS
    penalty = Objective(penalty_type)
    penalty_type.value[0](penalty, PenaltyNodeList(ocp, ocp.nlp[0], t, x, u, []))

    res = np.array(
        [
            [0.1, 0.99517075, 1.9901749, 1.0950042, 1, 2, 0.49750208],
            [0, 0, 0, 0, 0, 0, 0],
            [0.1, -0.9948376, -1.094671, 0.000166583, 0, 0, -0.0499167],
        ]
    )
    if value == -10:
        res = np.array(
            [
                [-10, -11.3830926, -12.2221642, -10.8390715, 1.0, 2.0, -0.4195358],
                [0, 0, 0, 0, 0, 0, 0],
                [-10, -9.7049496, -10.2489707, -10.5440211, 0, 0, -0.2720106],
            ]
        )

    np.testing.assert_almost_equal(
        penalty.weighted_function(x[0], u[0], [], 1, [], 1),
        res,
    )


@pytest.mark.parametrize("penalty_origin", [ObjectiveFcn.Lagrange, ObjectiveFcn.Mayer, ConstraintFcn])
@pytest.mark.parametrize("value", [0.1, -10])
def test_penalty_track_markers(penalty_origin, value):
    ocp = prepare_test_ocp()
    x = [DM.ones((12, 1)) * value]
    penalty_type = penalty_origin.TRACK_MARKERS

    if isinstance(penalty_type, (ObjectiveFcn.Lagrange, ObjectiveFcn.Mayer)):
        penalty = Objective(penalty_type, target=np.ones((3, 7, 1)) * value)
    else:
        penalty = Constraint(penalty_type, target=np.ones((3, 7, 1)) * value)

    penalty_type.value[0](penalty, ocp, ocp.nlp[0], [2], x, [], [])

    if isinstance(penalty_type, (ObjectiveFcn.Lagrange, ObjectiveFcn.Mayer)):
        res = ocp.nlp[0].J[0][0]["val"]
    else:
        res = ocp.nlp[0].g[0][0]["val"]

    expected = np.array(
        [
            [0.1, 0.99517075, 1.9901749, 1.0950042, 1, 2, 0.49750208],
            [0, 0, 0, 0, 0, 0, 0],
            [0.1, -0.9948376, -1.094671, 0.000166583, 0, 0, -0.0499167],
        ]
    )
    if value == -10:
        expected = np.array(
            [
                [-10, -11.3830926, -12.2221642, -10.8390715, 1.0, 2.0, -0.4195358],
                [0, 0, 0, 0, 0, 0, 0],
                [-10, -9.7049496, -10.2489707, -10.5440211, 0, 0, -0.2720106],
            ]
        )

    np.testing.assert_almost_equal(
        res,
        expected,
    )

    if isinstance(penalty_type, ConstraintFcn):
        np.testing.assert_almost_equal(
            ocp.nlp[0].g[0][0]["bounds"].min,
            np.array([[0]] * 3),
        )
        np.testing.assert_almost_equal(
            ocp.nlp[0].g[0][0]["bounds"].max,
            np.array([[0]] * 3),
        )


@pytest.mark.parametrize("penalty_origin", [ObjectiveFcn.Lagrange])
@pytest.mark.parametrize("value", [0.1, -10])
def test_penalty_minimize_markers_displacement(penalty_origin, value):
    ocp = prepare_test_ocp()
    x = [DM.ones((12, 1)) * value]
    penalty_type = penalty_origin.MINIMIZE_MARKERS_DISPLACEMENT
    penalty = Objective(penalty_type)
    penalty_type.value[0](penalty, PenaltyNodeList(ocp, ocp.nlp[0], [], x, [], []))

    np.testing.assert_almost_equal(
        ocp.nlp[0].J[0],
        np.array([]),
    )


@pytest.mark.parametrize("penalty_origin", [ObjectiveFcn.Lagrange, ObjectiveFcn.Mayer])
@pytest.mark.parametrize("value", [0.1, -10])
def test_penalty_minimize_markers_velocity(penalty_origin, value):
    ocp = prepare_test_ocp()
    t = [0]
    x = [DM.ones((12, 1)) * value]
    u = [0]
    penalty_type = penalty_origin.MINIMIZE_MARKERS_VELOCITY
    penalty = Objective(penalty_type)
    penalty_type.value[0](penalty, PenaltyNodeList(ocp, ocp.nlp[0], t, x, u, []))

    if value == 0.1:
        np.testing.assert_almost_equal(
            ocp.nlp[0].J[0][6]["val"],
            np.array(
                [
                    [-0.00499167],
                    [0],
                    [-0.0497502],
                ]
            ),
        )
    else:
        np.testing.assert_almost_equal(
            ocp.nlp[0].J[0][6]["val"],
            np.array(
                [
                    [2.7201056],
                    [0],
                    [-4.1953576],
                ]
            ),
        )


@pytest.mark.parametrize("penalty_origin", [ObjectiveFcn.Lagrange, ObjectiveFcn.Mayer, ConstraintFcn])
@pytest.mark.parametrize("value", [0.1, -10])
def test_penalty_track_markers_velocity(penalty_origin, value):
    ocp = prepare_test_ocp()
    x = [DM.ones((12, 1)) * value]
    penalty_type = penalty_origin.TRACK_MARKERS_VELOCITY

    if isinstance(penalty_type, (ObjectiveFcn.Lagrange, ObjectiveFcn.Mayer)):
        penalty = Objective(penalty_type, target=np.ones((3, 7, 1)) * value)
    else:
        penalty = Constraint(penalty_type, target=np.ones((3, 7, 1)) * value)

    penalty_type.value[0](penalty, PenaltyNodeList(ocp, ocp.nlp[0], [3], x, [], []))

    if isinstance(penalty_type, (ObjectiveFcn.Lagrange, ObjectiveFcn.Mayer)):
        res = ocp.nlp[0].J[0][6]["val"]
    else:
        res = ocp.nlp[0].g[0][6]["val"]

    if value == 0.1:
        np.testing.assert_almost_equal(
            res,
            np.array(
                [
                    [-0.00499167],
                    [0],
                    [-0.0497502],
                ]
            ),
        )
    else:
        np.testing.assert_almost_equal(
            res,
            np.array(
                [
                    [2.7201056],
                    [0],
                    [-4.1953576],
                ]
            ),
        )

    if isinstance(penalty_type, ConstraintFcn):
        np.testing.assert_almost_equal(
            ocp.nlp[0].g[0][0]["bounds"].min,
            np.array([[0]] * 3),
        )
        np.testing.assert_almost_equal(
            ocp.nlp[0].g[0][0]["bounds"].max,
            np.array([[0]] * 3),
        )


@pytest.mark.parametrize("penalty_origin", [ObjectiveFcn.Lagrange, ObjectiveFcn.Mayer, ConstraintFcn])
@pytest.mark.parametrize("value", [0.1, -10])
def test_penalty_track_markers(penalty_origin, value):
    ocp = prepare_test_ocp()
    t = [0]
    x = [DM.ones((12, 1)) * value]
    u = [0]
    penalty_type = penalty_origin.SUPERIMPOSE_MARKERS

    if isinstance(penalty_type, (ObjectiveFcn.Lagrange, ObjectiveFcn.Mayer)):
        penalty = Objective(penalty_type)
    else:
        penalty = Constraint(penalty_type)

    penalty_type.value[0](penalty, PenaltyNodeList(ocp, ocp.nlp[0], t, x, u, []), first_marker="m0", second_marker="m1")

    expected = np.array(
        [
            [-0.8951707],
            [0],
            [1.0948376],
        ]
    )
    if value == -10:
        expected = np.array(
            [
                [1.3830926],
                [0],
                [-0.2950504],
            ]
        )

    if isinstance(penalty_type, (ObjectiveFcn.Lagrange, ObjectiveFcn.Mayer)):
        res = ocp.nlp[0].J[0][0]["val"]
    else:
        res = ocp.nlp[0].g[0][0]["val"]

    np.testing.assert_almost_equal(
        res,
        expected,
    )

    if isinstance(penalty_type, ConstraintFcn):
        np.testing.assert_almost_equal(
            ocp.nlp[0].g[0][0]["bounds"].min,
            np.array([[0]] * 3),
        )
        np.testing.assert_almost_equal(
            ocp.nlp[0].g[0][0]["bounds"].max,
            np.array([[0]] * 3),
        )


@pytest.mark.parametrize("penalty_origin", [ObjectiveFcn.Lagrange, ObjectiveFcn.Mayer, ConstraintFcn])
@pytest.mark.parametrize("value", [0.1, -10])
def test_penalty_proportional_state(penalty_origin, value):
    ocp = prepare_test_ocp()
    x = [DM.ones((12, 1)) * value]
    penalty_type = penalty_origin.PROPORTIONAL_STATE

    if isinstance(penalty_type, (ObjectiveFcn.Lagrange, ObjectiveFcn.Mayer)):
        penalty = Objective(penalty_type)
    else:
        penalty = Constraint(penalty_type)

    penalty_type.value[0](
        penalty, PenaltyNodeList(ocp, ocp.nlp[0], [], x, [], []), which_var="states", first_dof=0, second_dof=1, coef=2
    )

    if isinstance(penalty_type, (ObjectiveFcn.Lagrange, ObjectiveFcn.Mayer)):
        res = ocp.nlp[0].J[0][0]["val"]
    else:
        res = ocp.nlp[0].g[0][0]["val"]

    np.testing.assert_almost_equal(
        res,
        np.array([[-value]]),
    )

    if isinstance(penalty_type, ConstraintFcn):
        np.testing.assert_almost_equal(ocp.nlp[0].g[0][0]["bounds"].min, np.array([[0]]))
        np.testing.assert_almost_equal(ocp.nlp[0].g[0][0]["bounds"].max, np.array([[0]]))


@pytest.mark.parametrize("penalty_origin", [ObjectiveFcn.Lagrange, ConstraintFcn])
@pytest.mark.parametrize("value", [0.1, -10])
def test_penalty_proportional_control(penalty_origin, value):
    ocp = prepare_test_ocp()
    u = [DM.ones((12, 1)) * value]
    penalty_type = penalty_origin.PROPORTIONAL_CONTROL

    if isinstance(penalty_type, (ObjectiveFcn.Lagrange, ObjectiveFcn.Mayer)):
        penalty = Objective(penalty_type)
    else:
        penalty = Constraint(penalty_type)

    first = 0
    second = 1
    coef = 2
    penalty_type.value[0](
        penalty,
        PenaltyNodeList(ocp, ocp.nlp[0], [], [], u, []),
        which_var="controls",
        first_dof=first,
        second_dof=second,
        coef=coef,
    )

    if isinstance(penalty_type, (ObjectiveFcn.Lagrange, ObjectiveFcn.Mayer)):
        res = ocp.nlp[0].J[0][0]["val"]
    else:
        res = ocp.nlp[0].g[0][0]["val"]

    np.testing.assert_almost_equal(
        res,
        np.array(u[0][first] - coef * u[0][second]),
    )

    if isinstance(penalty_type, ConstraintFcn):
        np.testing.assert_almost_equal(ocp.nlp[0].g[0][0]["bounds"].min, np.array([[0.0]]))
        np.testing.assert_almost_equal(ocp.nlp[0].g[0][0]["bounds"].max, np.array([[0.0]]))


@pytest.mark.parametrize("penalty_origin", [ObjectiveFcn.Lagrange])
@pytest.mark.parametrize("value", [0.1, -10])
def test_penalty_minimize_torque(penalty_origin, value):
    ocp = prepare_test_ocp()
    t = [0, 1]
    x = [0]
    u = [DM.ones((12, 1)) * value]
    penalty_type = penalty_origin.MINIMIZE_CONTROL, name="tau"
    penalty = Objective(penalty_type)
    penalty_type.value[0](penalty, PenaltyNodeList(ocp, ocp.nlp[0], t, x, u, []))

    if isinstance(penalty_type, (ObjectiveFcn.Lagrange, ObjectiveFcn.Mayer)):
        res = ocp.nlp[0].J[0][0]["val"]
    else:
        res = ocp.nlp[0].g[0][0]["val"]

    np.testing.assert_almost_equal(
        res,
        np.array([[value, value, value, value]]).T,
    )

    if isinstance(penalty_type, ConstraintFcn):
        np.testing.assert_almost_equal(ocp.nlp[0].g[0][0]["bounds"].min, np.array([[0.0, 0, 0, 0]]))
        np.testing.assert_almost_equal(ocp.nlp[0].g[0][0]["bounds"].max, np.array([[0.0, 0, 0, 0]]))


@pytest.mark.parametrize("penalty_origin", [ObjectiveFcn.Lagrange, ConstraintFcn])
@pytest.mark.parametrize("value", [0.1, -10])
def test_penalty_track_torque(penalty_origin, value):
    ocp = prepare_test_ocp()
    t = [0, 1]
    x = [0]
    u = [DM.ones((12, 1)) * value]
    penalty_type = penalty_origin.TRACK_TORQUE

    if isinstance(penalty_type, (ObjectiveFcn.Lagrange, ObjectiveFcn.Mayer)):
        penalty = Objective(penalty_type, target=np.ones((4, 1)) * value)
    else:
        penalty = Constraint(penalty_type, target=np.ones((4, 1)) * value)

    penalty_type.value[0](penalty, PenaltyNodeList(ocp, ocp.nlp[0], t, x, u, []))

    if isinstance(penalty_type, (ObjectiveFcn.Lagrange, ObjectiveFcn.Mayer)):
        res = ocp.nlp[0].J[0][0]["val"]
    else:
        res = ocp.nlp[0].g[0][0]["val"]

    np.testing.assert_almost_equal(
        res,
        np.array([[value, value, value, value]]).T,
    )

    if isinstance(penalty_type, ConstraintFcn):
        np.testing.assert_almost_equal(ocp.nlp[0].g[0][0]["bounds"].min, np.zeros((4, 1)))
        np.testing.assert_almost_equal(ocp.nlp[0].g[0][0]["bounds"].max, np.zeros((4, 1)))


@pytest.mark.parametrize("value", [0.1, -10])
def test_penalty_minimize_state_derivative(value):
    ocp = prepare_test_ocp()
    x = [DM.ones((12, 1)) * value, DM.ones((12, 1)) * value * 3]
    penalty_type = ObjectiveFcn.Lagrange.MINIMIZE_STATE_DERIVATIVE
    penalty = Objective(penalty_type)
    penalty_type.value[0](penalty, PenaltyNodeList(ocp, ocp.nlp[0], [], x, [], []))

    if isinstance(penalty_type, (ObjectiveFcn.Lagrange, ObjectiveFcn.Mayer)):
        res = ocp.nlp[0].J[0][0]["val"]
    else:
        res = ocp.nlp[0].g[0][0]["val"]

    np.testing.assert_almost_equal(
        res,
        np.array([[value * 2] * 8]).T,
    )

    if isinstance(penalty_type, ConstraintFcn):
        np.testing.assert_almost_equal(ocp.nlp[0].g[0][0]["bounds"].min, np.array([[0.0, 0, 0, 0]]))
        np.testing.assert_almost_equal(ocp.nlp[0].g[0][0]["bounds"].max, np.array([[0.0, 0, 0, 0]]))


@pytest.mark.parametrize("value", [0.1, -10])
def test_penalty_minimize_torque_derivative(value):
    ocp = prepare_test_ocp()
    u = [DM.ones((12, 1)) * value, DM.ones((12, 1)) * value * 3]
    penalty = Objective(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, name="tau", derivative=True)
    penalty.type(penalty, PenaltyNodeList(ocp, ocp.nlp[0], [], [], u, []))

    if isinstance(penalty.type, (ObjectiveFcn.Lagrange, ObjectiveFcn.Mayer)):
        res = ocp.nlp[0].J[0][0]["val"]
    else:
        res = ocp.nlp[0].g[0][0]["val"]

    np.testing.assert_almost_equal(
        res,
        np.array([[value * 2, value * 2, value * 2, value * 2]]).T,
    )

    if isinstance(penalty.type, ConstraintFcn):
        np.testing.assert_almost_equal(ocp.nlp[0].g[0][0]["bounds"].min, np.array([[0.0, 0, 0, 0]]))
        np.testing.assert_almost_equal(ocp.nlp[0].g[0][0]["bounds"].max, np.array([[0.0, 0, 0, 0]]))


@pytest.mark.parametrize("penalty_origin", [ObjectiveFcn.Lagrange])
@pytest.mark.parametrize("value", [0.1, -10])
def test_penalty_minimize_muscles_control(penalty_origin, value):
    ocp = prepare_test_ocp(with_muscles=True)
    t = [0, 1]
    x = [0]
    u = [DM.ones((12, 1)) * value]
    penalty_type = penalty_origin.MINIMIZE_MUSCLES_CONTROL
    penalty = Objective(penalty_type)
    penalty_type.value[0](penalty, PenaltyNodeList(ocp, ocp.nlp[0], t, x, u, []))

    if isinstance(penalty_type, (ObjectiveFcn.Lagrange, ObjectiveFcn.Mayer)):
        res = ocp.nlp[0].J[0][0]["val"]
    else:
        res = ocp.nlp[0].g[0][0]["val"]

    np.testing.assert_almost_equal(
        res,
        np.array([[value, value, value, value, value, value]]).T,
    )

    if isinstance(penalty_type, ConstraintFcn):
        np.testing.assert_almost_equal(ocp.nlp[0].g[0][0]["bounds"].min, np.array([[0.0, 0, 0, 0, 0, 0]]))
        np.testing.assert_almost_equal(ocp.nlp[0].g[0][0]["bounds"].max, np.array([[0.0, 0, 0, 0, 0, 0]]))


@pytest.mark.parametrize("penalty_origin", [ObjectiveFcn.Lagrange, ConstraintFcn])
@pytest.mark.parametrize("value", [0.1, -10])
def test_penalty_track_muscles_control(penalty_origin, value):
    ocp = prepare_test_ocp(with_muscles=True)
    t = [0, 1]
    x = [0]
    u = [DM.ones((12, 1)) * value]
    penalty_type = penalty_origin.TRACK_MUSCLES_CONTROL

    if isinstance(penalty_type, (ObjectiveFcn.Lagrange, ObjectiveFcn.Mayer)):
        penalty = Objective(penalty_type, target=np.ones((1, 1)) * value, index=0)
    else:
        penalty = Constraint(penalty_type, target=np.ones((1, 1)) * value, index=0)

    penalty_type.value[0](penalty, PenaltyNodeList(ocp, ocp.nlp[0], t, x, u, []))

    if isinstance(penalty_type, (ObjectiveFcn.Lagrange, ObjectiveFcn.Mayer)):
        res = ocp.nlp[0].J[0][0]["val"]
    else:
        res = ocp.nlp[0].g[0][0]["val"]

    np.testing.assert_almost_equal(
        res,
        np.array([[value]]),
    )

    if isinstance(penalty_type, ConstraintFcn):
        np.testing.assert_almost_equal(ocp.nlp[0].g[0][0]["bounds"].min, np.array([[0.0]]))
        np.testing.assert_almost_equal(ocp.nlp[0].g[0][0]["bounds"].max, np.array([[0.0]]))


@pytest.mark.parametrize("penalty_origin", [ObjectiveFcn.Lagrange])
@pytest.mark.parametrize("value", [0.1, -10])
def test_penalty_minimize_all_controls(penalty_origin, value):
    ocp = prepare_test_ocp(with_muscles=True)
    t = [0, 1]
    u = [DM.ones((12, 1)) * value]
    x = [0, 1]
    penalty_type = penalty_origin.MINIMIZE_ALL_CONTROLS
    penalty = Objective(penalty_type)
    penalty_type.value[0](penalty, PenaltyNodeList(ocp, ocp.nlp[0], t, x, u, []))

    if isinstance(penalty_type, (ObjectiveFcn.Lagrange, ObjectiveFcn.Mayer)):
        res = ocp.nlp[0].J[0][0]["val"]
    else:
        res = ocp.nlp[0].g[0][0]["val"]

    np.testing.assert_almost_equal(
        res,
        np.array([[value, value, value, value, value, value, value, value]]).T,
    )

    if isinstance(penalty_type, ConstraintFcn):
        np.testing.assert_almost_equal(ocp.nlp[0].g[0][0]["bounds"].min, np.array([[0.0, 0, 0, 0, 0, 0, 0, 0]]))
        np.testing.assert_almost_equal(ocp.nlp[0].g[0][0]["bounds"].max, np.array([[0.0, 0, 0, 0, 0, 0, 0, 0]]))


@pytest.mark.parametrize("penalty_origin", [ObjectiveFcn.Lagrange, ConstraintFcn])
@pytest.mark.parametrize("value", [0.1, -10])
def test_penalty_track_all_controls(penalty_origin, value):
    ocp = prepare_test_ocp(with_muscles=True)
    t = [0, 1]
    u = [DM.ones((12, 1)) * value]
    x = [0, 1]
    penalty_type = penalty_origin.TRACK_ALL_CONTROLS

    if isinstance(penalty_type, (ObjectiveFcn.Lagrange, ObjectiveFcn.Mayer)):
        penalty = Objective(penalty_type, target=np.ones((8, 1)) * value)
    else:
        penalty = Constraint(penalty_type, target=np.ones((8, 1)) * value)

    penalty_type.value[0](penalty, PenaltyNodeList(ocp, ocp.nlp[0], t, x, u, []))

    if isinstance(penalty_type, (ObjectiveFcn.Lagrange, ObjectiveFcn.Mayer)):
        res = ocp.nlp[0].J[0][0]["val"]
    else:
        res = ocp.nlp[0].g[0][0]["val"]

    np.testing.assert_almost_equal(
        res,
        np.array([[value, value, value, value, value, value, value, value]]).T,
    )

    if isinstance(penalty_type, ConstraintFcn):
        np.testing.assert_almost_equal(ocp.nlp[0].g[0][0]["bounds"].min, np.array([[0.0, 0, 0, 0, 0, 0, 0, 0]]).T)
        np.testing.assert_almost_equal(ocp.nlp[0].g[0][0]["bounds"].max, np.array([[0.0, 0, 0, 0, 0, 0, 0, 0]]).T)


@pytest.mark.parametrize("penalty_origin", [ObjectiveFcn.Lagrange])
@pytest.mark.parametrize("value", [0.1, -10])
def test_penalty_minimize_contact_forces(penalty_origin, value):
    ocp = prepare_test_ocp(with_contact=True)
    x = [DM.ones((8, 1)) * value]
    u = [DM.ones((4, 1)) * value]
    penalty_type = penalty_origin.MINIMIZE_CONTACT_FORCES
    penalty = Objective(penalty_type)
    penalty_type.value[0](penalty, PenaltyNodeList(ocp, ocp.nlp[0], [], x, u, []))

    if isinstance(penalty_type, (ObjectiveFcn.Lagrange, ObjectiveFcn.Mayer)):
        res = ocp.nlp[0].J[0][0]["val"]
    else:
        res = ocp.nlp[0].g[0][0]["val"]

    if value == 0.1:
        np.testing.assert_almost_equal(
            res,
            np.array([[-9.6680105, 127.2360329, 5.0905995]]).T,
        )
    else:
        np.testing.assert_almost_equal(
            res,
            np.array([[25.6627161, 462.7973306, -94.0182191]]).T,
        )

    if isinstance(penalty_type, ConstraintFcn):
        np.testing.assert_almost_equal(ocp.nlp[0].g[0][0]["bounds"].min, np.array([[0.0]]).T)
        np.testing.assert_almost_equal(ocp.nlp[0].g[0][0]["bounds"].max, np.array([[0.0]]).T)


@pytest.mark.parametrize("penalty_origin", [ObjectiveFcn.Lagrange, ConstraintFcn])
@pytest.mark.parametrize("value", [0.1, -10])
def test_penalty_track_contact_forces(penalty_origin, value):
    ocp = prepare_test_ocp(with_contact=True)
    x = [DM.ones((8, 1)) * value]
    u = [DM.ones((4, 1)) * value]
    penalty_type = penalty_origin.TRACK_CONTACT_FORCES

    if isinstance(penalty_type, (ObjectiveFcn.Lagrange, ObjectiveFcn.Mayer)):
        penalty = Objective(penalty_type, target=np.ones((1, 1)) * value, index=0)
    else:
        penalty = Constraint(penalty_type, target=np.ones((1, 1)) * value, index=0)

    penalty_type.value[0](penalty, PenaltyNodeList(ocp, ocp.nlp[0], [7], x, u, []))

    if isinstance(penalty_type, (ObjectiveFcn.Lagrange, ObjectiveFcn.Mayer)):
        res = ocp.nlp[0].J[0][0]["val"]
    else:
        res = ocp.nlp[0].g[0][0]["val"]

    if value == 0.1:
        np.testing.assert_almost_equal(
            res,
            np.array([[-9.6680105]]),
        )
    else:
        np.testing.assert_almost_equal(
            res,
            np.array([[25.6627161]]),
        )

    if isinstance(penalty_type, ConstraintFcn):
        np.testing.assert_almost_equal(ocp.nlp[0].g[0][0]["bounds"].min, np.array([[0.0]]).T)
        np.testing.assert_almost_equal(ocp.nlp[0].g[0][0]["bounds"].max, np.array([[0.0]]).T)


@pytest.mark.parametrize("value", [0.1, -10])
def test_penalty_minimize_predicted_com_height(value):
    ocp = prepare_test_ocp()
    t = [0]
    x = [DM.ones((12, 1)) * value]
    u = [0]
    penalty_type = ObjectiveFcn.Mayer.MINIMIZE_PREDICTED_COM_HEIGHT
    penalty = Objective(penalty_type)
    penalty_type.value[0](penalty, PenaltyNodeList(ocp, ocp.nlp[0], t, x, u, []))

    res = np.array(0.0501274 if value == 0.1 else -3.72579)
    np.testing.assert_almost_equal(ocp.nlp[0].J[0][0]["val"], res)


@pytest.mark.parametrize("penalty_origin", [ObjectiveFcn.Lagrange, ObjectiveFcn.Mayer, ConstraintFcn])
@pytest.mark.parametrize("value", [0.1, -10])
def test_penalty_minimize_com_position(value, penalty_origin):
    ocp = prepare_test_ocp()
    t = [0]
    x = [DM.ones((12, 1)) * value]
    u = [0]
    if "TRACK_COM_POSITION" in penalty_origin._member_names_:
        penalty_type = penalty_origin.TRACK_COM_POSITION
    else:
        penalty_type = penalty_origin.MINIMIZE_COM_POSITION

    if isinstance(penalty_type, (ObjectiveFcn.Lagrange, ObjectiveFcn.Mayer)):
        penalty = Objective(penalty_type)
    else:
        penalty = Constraint(penalty_type)

    penalty_type.value[0](penalty, PenaltyNodeList(ocp, ocp.nlp[0], t, x, u, []))

    if isinstance(penalty_type, (ObjectiveFcn.Lagrange, ObjectiveFcn.Mayer)):
        res = ocp.nlp[0].J[0][0]["val"]
    else:
        res = ocp.nlp[0].g[0][0]["val"]

    expected = np.array([[0.05], [0.05], [0.05]])
    if value == -10:
        expected = np.array([[-5], [0.05], [-5]])

    np.testing.assert_almost_equal(res, expected)

    if isinstance(penalty_type, ConstraintFcn):
        np.testing.assert_almost_equal(ocp.nlp[0].g[0][0]["bounds"].min, np.array(0))
        np.testing.assert_almost_equal(ocp.nlp[0].g[0][0]["bounds"].max, np.array(0))


@pytest.mark.parametrize("penalty_origin", [ObjectiveFcn.Lagrange, ObjectiveFcn.Mayer, ConstraintFcn])
@pytest.mark.parametrize("value", [0.1, -10])
def test_penalty_track_segment_with_custom_rt(penalty_origin, value):
    ocp = prepare_test_ocp()
    t = [0]
    x = [DM.ones((12, 1)) * value]
    u = [0]
    penalty_type = penalty_origin.TRACK_SEGMENT_WITH_CUSTOM_RT

    if isinstance(penalty_type, (ObjectiveFcn.Lagrange, ObjectiveFcn.Mayer)):
        penalty = Objective(penalty_type)
    else:
        penalty = Constraint(penalty_type)

    penalty_type.value[0](penalty, PenaltyNodeList(ocp, ocp.nlp[0], t, x, u, []), segment="ground", rt_idx=0)

    if isinstance(penalty_type, (ObjectiveFcn.Lagrange, ObjectiveFcn.Mayer)):
        res = ocp.nlp[0].J[0][0]["val"]
    else:
        res = ocp.nlp[0].g[0][0]["val"]

    expected = np.array([[0], [0.1], [0]])
    if value == -10:
        expected = np.array([[3.1415927], [0.575222], [3.1415927]])

    np.testing.assert_almost_equal(
        res,
        expected,
    )

    if isinstance(penalty_type, ConstraintFcn):
        np.testing.assert_almost_equal(ocp.nlp[0].g[0][0]["bounds"].min, np.array([[0], [0], [0]]))
        np.testing.assert_almost_equal(ocp.nlp[0].g[0][0]["bounds"].max, np.array([[0], [0], [0]]))


@pytest.mark.parametrize("penalty_origin", [ObjectiveFcn.Lagrange, ObjectiveFcn.Mayer, ConstraintFcn])
@pytest.mark.parametrize("value", [0.1, -10])
def test_penalty_track_marker_with_segment_axis(penalty_origin, value):
    ocp = prepare_test_ocp()
    t = [0]
    x = [DM.ones((12, 1)) * value]
    u = [0]
    penalty_type = penalty_origin.TRACK_MARKER_WITH_SEGMENT_AXIS

    if isinstance(penalty_type, (ObjectiveFcn.Lagrange, ObjectiveFcn.Mayer)):
        penalty = Objective(penalty_type)
    else:
        penalty = Constraint(penalty_type)

    penalty_type.value[0](
        penalty, PenaltyNodeList(ocp, ocp.nlp[0], t, x, u, []), marker="m0", segment="ground", axis=Axis.X
    )

    if isinstance(penalty_type, (ObjectiveFcn.Lagrange, ObjectiveFcn.Mayer)):
        res = ocp.nlp[0].J[0][0]["val"]
    else:
        res = ocp.nlp[0].g[0][0]["val"]

    np.testing.assert_almost_equal(
        res,
        np.array([[0]]),
    )

    if isinstance(penalty_type, ConstraintFcn):
        np.testing.assert_almost_equal(ocp.nlp[0].g[0][0]["bounds"].min, np.array([[0]]))
        np.testing.assert_almost_equal(ocp.nlp[0].g[0][0]["bounds"].max, np.array([[0]]))


@pytest.mark.parametrize("penalty_origin", [ConstraintFcn])
@pytest.mark.parametrize("value", [0.1, -10])
@pytest.mark.parametrize("direction", ["GREATER_THAN", "LESSER_THAN"])
def test_penalty_contact_force_inequality(penalty_origin, value, direction):
    ocp = prepare_test_ocp(with_contact=True)
    x = [DM.ones((8, 1)) * value]
    u = [DM.ones((4, 1)) * value]

    if direction == "GREATER_THAN":
        min_bound = 1
        max_bound = np.inf
        if value == 0.1:
            expected = [-9.6680105, 1.0, np.inf]
        elif value == -10:
            expected = [25.6627161, 1.0, np.inf]
        else:
            raise RuntimeError("Wrong test")
    elif direction == "LESSER_THAN":
        min_bound = -np.inf
        max_bound = 1
        if value == 0.1:
            expected = [-9.6680105, -np.inf, 1.0]
        elif value == -10:
            expected = [25.6627161, -np.inf, 1.0]
        else:
            raise RuntimeError("Wrong test")
    else:
        raise RuntimeError("Wrong test")

    penalty_type = penalty_origin.CONTACT_FORCE
    penalty = Constraint(penalty_type, min_bound=min_bound, max_bound=max_bound)
    penalty_type.value[0](
        penalty,
        PenaltyNodeList(ocp, ocp.nlp[0], [], x, u, []),
        contact_force_idx=0,
    )
    res = ocp.nlp[0].g[0][0]["val"]

    np.testing.assert_almost_equal(res, np.array([[expected[0]]]))

    if isinstance(penalty_type, ConstraintFcn):
        np.testing.assert_almost_equal(ocp.nlp[0].g[0][0]["bounds"].min, np.array([[expected[1]]]))
        np.testing.assert_almost_equal(ocp.nlp[0].g[0][0]["bounds"].max, np.array([[expected[2]]]))


@pytest.mark.parametrize("value", [0.1, -10])
def test_penalty_non_slipping(value):
    ocp = prepare_test_ocp(with_contact=True)
    t = [0]
    x = [DM.ones((8, 1)) * value]
    u = [DM.ones((4, 1)) * value]
    penalty_type = ConstraintFcn.NON_SLIPPING
    penalty = Constraint(penalty_type)
    penalty_type.value[0](
        penalty,
        PenaltyNodeList(ocp, ocp.nlp[0], t, x, u, []),
        tangential_component_idx=0,
        normal_component_idx=1,
        static_friction_coefficient=2,
    )

    res = []
    for i in range(len(ocp.nlp[0].g[0])):
        res.append(ocp.nlp[0].g[0][i]["val"])

    if value == 0.1:
        expected = [[64662.56185612, 64849.5027121], [0, 0], [np.inf, np.inf]]
    elif value == -10:
        expected = [[856066.90177734, 857384.05177395], [0, 0], [np.inf, np.inf]]
    else:
        raise RuntimeError("Test not ready")

    np.testing.assert_almost_equal(np.concatenate(res)[:, 0], np.array(expected[0]))

    if isinstance(penalty_type, ConstraintFcn):
        np.testing.assert_almost_equal(ocp.nlp[0].g[0][0]["bounds"].min, np.array([expected[1]]).T)
        np.testing.assert_almost_equal(ocp.nlp[0].g[0][0]["bounds"].max, np.array([expected[2]]).T)


@pytest.mark.parametrize("value", [2])
@pytest.mark.parametrize("threshold", [None, 15, -15])
def test_tau_max_from_actuators(value, threshold):
    ocp = prepare_test_ocp(with_actuator=True)
    x = [DM.zeros((6, 1)), DM.zeros((6, 1))]
    u = [DM.ones((3, 1)) * value, DM.ones((3, 1)) * value]
    penalty_type = ConstraintFcn.TORQUE_MAX_FROM_ACTUATORS
    penalty = Constraint(penalty_type)
    if threshold and threshold < 0:
        with pytest.raises(ValueError, match="min_torque cannot be negative in tau_max_from_actuators"):
            penalty_type.value[0](penalty, PenaltyNodeList(ocp, ocp.nlp[0], [], x, u, []), min_torque=threshold),
    else:
        penalty_type.value[0](penalty, PenaltyNodeList(ocp, ocp.nlp[0], [], x, u, []), min_torque=threshold)

    val = []
    for i in range(len(ocp.nlp[0].g[0])):
        val.append(ocp.nlp[0].g[0][i]["val"])
    for res in val:
        if threshold:
            np.testing.assert_almost_equal(res, np.repeat([value + threshold, value - threshold], 3)[:, np.newaxis])
        else:
            np.testing.assert_almost_equal(res, np.repeat([value + 5, value - 10], 3)[:, np.newaxis])


@pytest.mark.parametrize("value", [0.1, -10])
def test_penalty_time_constraint(value):
    ocp = prepare_test_ocp()
    penalty_type = ConstraintFcn.TIME_CONSTRAINT
    penalty = Constraint(penalty_type)
    penalty_type.value[0](penalty, PenaltyNodeList(ocp, ocp.nlp[0], [], [], [], []))
    res = ocp.nlp[0].g[0]

    np.testing.assert_almost_equal(
        res,
        np.array([]),
    )


@pytest.mark.parametrize("penalty_origin", [ObjectiveFcn.Lagrange, ObjectiveFcn.Mayer, ConstraintFcn])
@pytest.mark.parametrize("value", [0.1, -10])
def test_penalty_custom(penalty_origin, value):
    def custom(pn, mult):
        my_values = DM.zeros((12, 1)) + pn.x[0] * mult
        return my_values

    ocp = prepare_test_ocp()
    t = [0]
    x = [DM.ones((12, 1)) * value]
    penalty_type = penalty_origin.CUSTOM

    if isinstance(penalty_type, (ObjectiveFcn.Lagrange, ObjectiveFcn.Mayer)):
        penalty = Objective(penalty_type, index=0)
    else:
        penalty = Constraint(penalty_type, index=0)

    penalty.custom_function = custom
    mult = 2
    penalty_type.value[0](penalty, PenaltyNodeList(ocp, ocp.nlp[0], t, x, [], []), mult=mult)

    if isinstance(penalty_type, (ObjectiveFcn.Lagrange, ObjectiveFcn.Mayer)):
        res = ocp.nlp[0].J[0][0]["val"]
    else:
        res = ocp.nlp[0].g[0][0]["val"]

    np.testing.assert_almost_equal(res, np.array([[value * mult]] * 12))

    if isinstance(penalty_type, ConstraintFcn):
        np.testing.assert_almost_equal(ocp.nlp[0].g[0][0]["bounds"].min, np.array([[0]] * 12))
        np.testing.assert_almost_equal(ocp.nlp[0].g[0][0]["bounds"].max, np.array([[0]] * 12))


@pytest.mark.parametrize("penalty_origin", [ObjectiveFcn.Lagrange, ObjectiveFcn.Mayer, ConstraintFcn])
@pytest.mark.parametrize("value", [0.1, -10])
def test_penalty_custom_fail(penalty_origin, value):
    def custom_no_mult(ocp, nlp, t, x, u, p):
        my_values = DM.zeros((12, 1)) + x[0]
        return my_values

    def custom_with_mult(ocp, nlp, t, x, u, p, mult):
        my_values = DM.zeros((12, 1)) + x[0] * mult
        return my_values

    ocp = prepare_test_ocp()
    x = [DM.ones((12, 1)) * value]
    penalty_type = penalty_origin.CUSTOM

    if isinstance(penalty_type, (ObjectiveFcn.Lagrange, ObjectiveFcn.Mayer)):
        penalty = Objective(penalty_type)
    else:
        penalty = Constraint(penalty_type)

    with pytest.raises(TypeError):
        penalty.custom_function = custom_no_mult
        penalty_type.value[0](penalty, ocp, ocp.nlp[0], [], x, [], [], mult=2)

    with pytest.raises(TypeError):
        penalty.custom_function = custom_with_mult
        penalty_type.value[0](penalty, ocp, ocp.nlp[0], [], x, [], [])

    with pytest.raises(TypeError):
        keywords = [
            "phase",
            "list_index",
            "name",
            "type",
            "params",
            "node",
            "quadratic",
            "index",
            "target",
            "min_bound",
            "max_bound",
            "custom_function",
            "weight",
        ]
        for keyword in keywords:
            exec(
                f"""def custom_with_keyword(ocp, nlp, t, x, u, p, {keyword}):
                            my_values = DM.zeros((12, 1)) + x[index]
                            return my_values"""
            )
            exec("""penalty.custom_function = custom_with_keyword""")
            exec(f"""penalty_type.value[0](penalty, ocp, ocp.nlp[0], [], x, [], [], {keyword}=0)""")


@pytest.mark.parametrize("value", [0.1, -10])
def test_penalty_custom_with_bounds(value):
    def custom_with_bounds(pn):
        my_values = DM.zeros((12, 1)) + pn.x[0]
        return -10, my_values, 10

    ocp = prepare_test_ocp()
    t = [0]
    x = [DM.ones((12, 1)) * value]

    penalty_type = ConstraintFcn.CUSTOM
    penalty = Constraint(penalty_type)

    penalty.custom_function = custom_with_bounds
    penalty_type.value[0](penalty, PenaltyNodeList(ocp, ocp.nlp[0], t, x, [], []))

    res = ocp.nlp[0].g[0][0]["val"]

    np.testing.assert_almost_equal(res, np.array([[value]] * 12))
    np.testing.assert_almost_equal(ocp.nlp[0].g[0][0]["bounds"].min, np.array([[-10]] * 12))
    np.testing.assert_almost_equal(ocp.nlp[0].g[0][0]["bounds"].max, np.array([[10]] * 12))


@pytest.mark.parametrize("value", [0.1, -10])
def test_penalty_custom_with_bounds_failing_min_bound(value):
    def custom_with_bounds(pn):
        my_values = DM.zeros((12, 1)) + pn.x[0]
        return -10, my_values, 10

    ocp = prepare_test_ocp()
    t = [0]
    x = [DM.ones((12, 1)) * value]

    penalty_type = ConstraintFcn.CUSTOM
    penalty = Constraint(penalty_type)

    penalty.min_bound = 0
    penalty.custom_function = custom_with_bounds

    with pytest.raises(RuntimeError):
        penalty_type.value[0](penalty, PenaltyNodeList(ocp, ocp.nlp[0], t, x, [], []))


@pytest.mark.parametrize("value", [0.1, -10])
def test_penalty_custom_with_bounds_failing_max_bound(value):
    def custom_with_bounds(pn):
        my_values = DM.zeros((12, 1)) + x[0]
        return -10, my_values, 10

    ocp = prepare_test_ocp()
    t = [0]
    x = [DM.ones((12, 1)) * value]

    penalty_type = ConstraintFcn.CUSTOM
    penalty = Constraint(penalty_type)

    penalty.max_bound = 0
    penalty.custom_function = custom_with_bounds

    with pytest.raises(
        RuntimeError,
        match="You cannot have non linear bounds for custom constraints and min_bound or max_bound defined",
    ):
        penalty_type.value[0](penalty, PenaltyNodeList(ocp, ocp.nlp[0], t, x, [], []))


@pytest.mark.parametrize("penalty_origin", [ObjectiveFcn.Lagrange, ObjectiveFcn.Mayer])
@pytest.mark.parametrize("value", [0.1, -10])
def test_penalty_track_markers_with_nan(penalty_origin, value):
    ocp = prepare_test_ocp()
    penalty_type = penalty_origin.TRACK_MARKERS

    target = np.ones((3, 7, 11)) * value
    target[:, -2, [0, -1]] = np.nan

    if isinstance(penalty_type, ObjectiveFcn.Lagrange):
        penalty = Objective(penalty_type, node=Node.ALL, target=target)
        X = ocp.nlp[0].X[0]
    elif isinstance(penalty_type, ObjectiveFcn.Mayer):
        penalty = Objective(penalty_type, node=Node.END, target=target[:, :, -1:])
        X = ocp.nlp[0].X[10]
    else:
        raise RuntimeError("Test not ready")
    ocp.update_objectives(penalty)
    res = Function("res", [X], [IpoptInterface.finalize_objective_value(ocp.nlp[0].J[0][0])]).expand()()["o0"]

    if value == 0.1:
        expected = 8.73 * ocp.nlp[0].J[0][0]["dt"]
    else:
        expected = 1879.25 * ocp.nlp[0].J[0][0]["dt"]

    np.testing.assert_almost_equal(
        np.array(res),
        expected,
    )


@pytest.mark.parametrize(
    "node", [Node.ALL, Node.INTERMEDIATES, Node.START, Node.MID, Node.PENULTIMATE, Node.END, Node.TRANSITION]
)
@pytest.mark.parametrize("ns", [1, 10, 11])
def test_PenaltyFunctionAbstract_get_node(node, ns):

    NLP = nlp()
    NLP.ns = ns
    NLP.X = np.linspace(0, -10, ns + 1)
    NLP.U = np.linspace(10, 19, ns)

    pn = []
    penalty = PenaltyOption(pn)
    penalty.node = node

    if node == Node.MID and ns % 2 != 0:
        with pytest.raises(ValueError, match="Number of shooting points must be even to use MID"):
            t, x, u = PenaltyFunctionAbstract._get_node(NLP, penalty)
        return
    elif node == Node.TRANSITION:
        with pytest.raises(RuntimeError, match=" is not a valid node"):
            t, x, u = PenaltyFunctionAbstract._get_node(NLP, penalty)
        return
    elif ns == 1 and node == Node.PENULTIMATE:
        with pytest.raises(ValueError, match="Number of shooting points must be greater than 1"):
            t, x, u = PenaltyFunctionAbstract._get_node(NLP, penalty)
        return
    else:
        t, x, u = PenaltyFunctionAbstract._get_node(NLP, penalty)

    x_expected = NLP.X
    u_expected = NLP.U

    if node == Node.ALL:
        np.testing.assert_almost_equal(t, [i for i in range(ns + 1)])
        np.testing.assert_almost_equal(np.array(x), np.linspace(0, -10, ns + 1))
        np.testing.assert_almost_equal(np.array(u), np.linspace(10, 19, ns))
    elif node == Node.INTERMEDIATES:
        np.testing.assert_almost_equal(t, [i for i in range(1, ns - 1)])
        np.testing.assert_almost_equal(np.array(x), x_expected[1 : ns - 1])
        np.testing.assert_almost_equal(np.array(u), u_expected[1 : ns - 1])
    elif node == Node.START:
        np.testing.assert_almost_equal(t, [0])
        np.testing.assert_almost_equal(np.array(x), x_expected[0])
        np.testing.assert_almost_equal(np.array(u), u_expected[0])
    elif node == Node.MID:
        np.testing.assert_almost_equal(t, [ns // 2])
        np.testing.assert_almost_equal(np.array(x), x_expected[ns // 2])
        np.testing.assert_almost_equal(np.array(u), u_expected[ns // 2])
    elif node == Node.PENULTIMATE:
        np.testing.assert_almost_equal(t, [ns - 1])
        np.testing.assert_almost_equal(np.array(x), x_expected[-2])
        np.testing.assert_almost_equal(np.array(u), u_expected[-1])
    elif node == Node.END:
        np.testing.assert_almost_equal(t, [ns])
        np.testing.assert_almost_equal(np.array(x), x_expected[ns])
        np.testing.assert_almost_equal(u, [])
    else:
        raise RuntimeError("Something went wrong")
