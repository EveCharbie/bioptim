from casadi import Function, vertcat, horzcat, collocation_points, tangent, rootfinder, DM, MX, SX, linspace
import numpy as np

from ..misc.enums import ControlType, DefectType
from ..models.protocols.biomodel import BioModel


class Integrator:
    """
    Abstract class for CasADi-based integrator

    Attributes
    ----------
    model: BioModel
        The biorbd model to integrate
    time_integration_grid = tuple[float, ...]
        The time integration grid
    idx: int
        The index of the degrees of freedom to integrate
    cx: MX | SX
        The CasADi type the integration should be built from
    x_sym: MX | SX
        The state variables
    u_sym: MX | SX
        The control variables
    param_sym: MX | SX
        The parameters variables
    s_sym: MX | SX
        The stochastic variables
    fun: Callable
        The dynamic function which provides the derivative of the states
    implicit_fun: Callable
        The implicit dynamic function which provides the defects of the dynamics
    control_type: ControlType
        The type of the controls
    function = casadi.Function
        The CasADi graph of the integration

    Methods
    -------
    __call__(self, *args, **kwargs)
        Interface to self.function
    map(self, *args, **kwargs) -> Function
        Get the multithreaded CasADi graph of the integration
    get_u(self, u: np.ndarray, t: float) -> np.ndarray
        Get the control at a given time
    dxdt(self, h: float, time: float | MX | SX, states: MX | SX, controls: MX | SX, params: MX | SX, stochastic_variables: MX | SX) -> tuple[SX, list[SX]]
        The dynamics of the system
    """

    # Todo change ode and ode_opt into class
    def __init__(self, ode: dict, ode_opt: dict):
        """
        Parameters
        ----------
        ode: dict
            The ode description
        ode_opt: dict
            The ode options
        """

        self.model = ode_opt["model"]
        self.ode_idx = ode_opt["ode_index"]
        self.cx = ode_opt["cx"]
        self.t_span_sym = ode["t"]
        self.x_sym = ode["x"]
        self.u_sym = ode["p"]
        self.param_sym = ode["param"]
        self.param_scaling = ode_opt["param_scaling"]
        self.s_sym = ode["s"]
        self.fun = ode["ode"]
        self.implicit_fun = ode["implicit_ode"]
        self.defects_type = ode_opt["defects_type"]
        self.control_type = ode_opt["control_type"]
        self.function = None
        self.allow_free_variables = ode_opt["allow_free_variables"]

        # Initialize is expected to set step_time
        self._initialize(ode, ode_opt)

        self.step_times_from_dt = self._time_xall_from_dt_func
        self.function = Function(
            "integrator",
            [
                self.t_span_sym,
                self._x_sym_modified,
                self.u_sym,
                self.param_sym,
                self.s_sym,
            ],
            self.dxdt(
                states=self.x_sym,
                controls=self.u_sym,
                params=self.param_sym * self.param_scaling,
                stochastic_variables=self.s_sym,
            ),
            self._input_names,
            self._output_names,
            {"allow_free": self.allow_free_variables},
        )

        self.symbolic_dxdt = self.dxdt(
                states=self.x_sym,
                controls=self.u_sym,
                params=self.param_sym * self.param_scaling,
                stochastic_variables=self.s_sym,
            )

    @property
    def shape_xf(self) -> tuple[int, int]:
        """
        Returns the expected shape of xf
        """
        raise NotImplementedError("This method should be implemented for a given integrator")
    
    @property
    def shape_xall(self) -> tuple[int, int]:
        """
        Returns the expected shape of xall
        """
        raise NotImplementedError("This method should be implemented for a given integrator")
    
    @property
    def time_xall(self) -> DM:
        """
        Returns the time vector of xall
        """
        raise NotImplementedError("This method should be implemented for a given integrator")

    @property
    def _time_xall_from_dt_func(self) -> Function:
        raise NotImplementedError("This method should be implemented for a given integrator")

    @property
    def _x_sym_modified(self):
        return self.x_sym
    
    @property
    def _input_names(self):
        return ["t_span", "x0", "u", "p", "s"]

    @property
    def _output_names(self):
        return ["xf", "xall"]

    def _initialize(self, ode: dict, ode_opt: dict):
        """
        This method is called by the constructor to initialize the integrator right before 
        creating the CasADi function from dxdt
        """
        pass

    def __call__(self, *args, **kwargs):
        """
        Interface to self.function
        """

        return self.function(*args, **kwargs)

    def map(self, *args) -> Function:
        """
        Get the multithreaded CasADi graph of the integration

        Returns
        -------
        The multithreaded CasADi graph of the integration
        """
        return self.function.map(*args)

    @property
    def _integration_time(self):
        raise NotImplementedError("This method should be implemented for a given integrator")

    def get_u(self, u: np.ndarray, t: float) -> np.ndarray:
        """
        Get the control at a given time

        Parameters
        ----------
        u: np.ndarray
            The control matrix
        t: float
            The time a which control should be computed

        Returns
        -------
        The control at a given time
        """

        if self.control_type == ControlType.CONSTANT or self.control_type == ControlType.CONSTANT_WITH_LAST_NODE:
            return u
        elif self.control_type == ControlType.LINEAR_CONTINUOUS:
            dt_norm = (t - self.t_span_sym[0]) / self.t_span_sym[1]
            return u[:, 0] + (u[:, 1] - u[:, 0]) * dt_norm
        else:
            raise RuntimeError(f"{self.control_type} ControlType not implemented yet")

    def dxdt(
        self,
        states: MX | SX,
        controls: MX | SX,
        params: MX | SX,
        stochastic_variables: MX | SX,
    ) -> tuple:
        """
        The dynamics of the system

        Parameters
        ----------
        states: MX | SX
            The states of the system
        controls: MX | SX
            The controls of the system
        params: MX | SX
            The parameters of the system
        stochastic_variables: MX | SX
            The stochastic variables of the system

        Returns
        -------
        The derivative of the states
        """

        raise RuntimeError("Integrator is abstract, please specify a proper one")


class RK(Integrator):
    """
    Abstract class for Runge-Kutta integrators

    Attributes
    ----------
    _n_step: int
        Number of finite element during the integration
    """

    def __init__(self, ode: dict, ode_opt: dict):
        """
        Parameters
        ----------
        ode: dict
            The ode description
        ode_opt: dict
            The ode options
        """
        self._n_step = ode_opt["number_of_finite_elements"]
        super(RK, self).__init__(ode, ode_opt)
        
    @property
    def _integration_time(self):
        return self.t_span_sym[1] / self._n_step

    @property
    def shape_xf(self):
        return [self.x_sym.shape[0], 1]

    @property
    def shape_xall(self):
        return [self.x_sym.shape[0], self._n_step + 1]

    @property
    def _time_xall_from_dt_func(self) -> Function:
        return Function(
            "step_time", 
            [self.t_span_sym], 
            [linspace(self.t_span_sym[0], self.t_span_sym[0] + self.t_span_sym[1], self.shape_xall[1])]
        )

    @property
    def h(self):
        return (self.t_span_sym[1] - self.t_span_sym[0]) / self._n_step

    def next_x(self, t0: float | MX | SX, x_prev: MX | SX, u: MX | SX, p: MX | SX, s: MX | SX) -> MX | SX:
        """
        Compute the next integrated state (abstract)

        Parameters
        ----------
        t0: float | MX | SX
            The initial time of the integration
        x_prev: MX | SX
            The current state of the system
        u: MX | SX
            The control of the system
        p: MX | SX
            The parameters of the system
        s: MX | SX
            The stochastic variables of the system

        Returns
        -------
        The next integrate states
        """

        raise RuntimeError("RK is abstract, please select a specific RK")

    def dxdt(
        self,
        states: MX | SX,
        controls: MX | SX,
        params: MX | SX,
        stochastic_variables: MX | SX,
    ) -> tuple:

        u = controls
        x = self.cx(states.shape[0], self._n_step + 1)
        p = params
        x[:, 0] = states
        s = stochastic_variables

        for i in range(1, self._n_step + 1):
            t = self.t_span_sym[0] + self._integration_time * (i - 1)
            x[:, i] = self.next_x(t, x[:, i - 1], u, p, s)
            if self.model.nb_quaternions > 0:
                x[:, i] = self.model.normalize_state_quaternions(x[:, i])

        return x[:, -1], x


class RK1(RK):
    """
    Numerical integration using first order Runge-Kutta 1 Method (Forward Euler Method).
    """

    def next_x(self, t0: float | MX | SX, x_prev: MX | SX, u: MX | SX, p: MX | SX, s: MX | SX) -> MX | SX:
        return x_prev + self.h * self.fun(t0, x_prev, self.get_u(u, t0), p, s)[:, self.ode_idx]


class RK2(RK):
    """
    Numerical integration using second order Runge-Kutta Method (Midpoint Method).
    """

    def next_x(self, t0: float | MX | SX, x_prev: MX | SX, u: MX | SX, p: MX | SX, s: MX | SX):
        h = self.h

        k1 = self.fun(vertcat(t0, h), x_prev, self.get_u(u, t0), p, s)[:, self.ode_idx]
        return x_prev + h * self.fun(t0, x_prev + h / 2 * k1, self.get_u(u, t0 + h / 2), p, s)[:, self.ode_idx]


class RK4(RK):
    """
    Numerical integration using fourth order Runge-Kutta method.
    """

    def next_x(self, t0: float | MX | SX, x_prev: MX | SX, u: MX | SX, p: MX | SX, s: MX | SX):
        h = self.h
        t = vertcat(t0, h)

        k1 = self.fun(t, x_prev, self.get_u(u, t0), p, s)[:, self.ode_idx]
        k2 = self.fun(t, x_prev + h / 2 * k1, self.get_u(u, t0 + h / 2), p, s)[:, self.ode_idx]
        k3 = self.fun(t, x_prev + h / 2 * k2, self.get_u(u, t0 + h / 2), p, s)[:, self.ode_idx]
        k4 = self.fun(t, x_prev + h * k3, self.get_u(u, t0 + h), p, s)[:, self.ode_idx]
        return x_prev + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


class RK8(RK4):
    """
    Numerical integration using eighth order Runge-Kutta method.
    """

    def next_x(self, t0: float | MX | SX, x_prev: MX | SX, u: MX | SX, p: MX | SX, s: MX | SX):
        h = self.h
        t = vertcat(t0, h)

        k1 = self.fun(t, x_prev, self.get_u(u, t0), p, s)[:, self.ode_idx]
        k2 = self.fun(t, x_prev + (h * 4 / 27) * k1, self.get_u(u, t0 + h * (4 / 27)), p, s)[:, self.ode_idx]
        k3 = self.fun(t, x_prev + (h / 18) * (k1 + 3 * k2), self.get_u(u, t0 + h * (2 / 9)), p, s)[:, self.ode_idx]
        k4 = self.fun(t, x_prev + (h / 12) * (k1 + 3 * k3), self.get_u(u, t0 + h * (1 / 3)), p, s)[:, self.ode_idx]
        k5 = self.fun(t, x_prev + (h / 8) * (k1 + 3 * k4), self.get_u(u, t0 + h * (1 / 2)), p, s)[:, self.ode_idx]
        k6 = self.fun(
            t, x_prev + (h / 54) * (13 * k1 - 27 * k3 + 42 * k4 + 8 * k5), self.get_u(u, t0 + h * (2 / 3)), p, s
        )[:, self.ode_idx]
        k7 = self.fun(
            t,
            x_prev + (h / 4320) * (389 * k1 - 54 * k3 + 966 * k4 - 824 * k5 + 243 * k6),
            self.get_u(u, t0 + h * (1 / 6)),
            p,
            s,
        )[:, self.ode_idx]
        k8 = self.fun(
            t,
            x_prev + (h / 20) * (-234 * k1 + 81 * k3 - 1164 * k4 + 656 * k5 - 122 * k6 + 800 * k7),
            self.get_u(u, t0 + h),
            p,
            s,
        )[:, self.ode_idx]
        k9 = self.fun(
            t,
            x_prev + (h / 288) * (-127 * k1 + 18 * k3 - 678 * k4 + 456 * k5 - 9 * k6 + 576 * k7 + 4 * k8),
            self.get_u(u, t0 + h * (5 / 6)),
            p,
            s,
        )[:, self.ode_idx]
        k10 = self.fun(
            t,
            x_prev
            + (h / 820) * (1481 * k1 - 81 * k3 + 7104 * k4 - 3376 * k5 + 72 * k6 - 5040 * k7 - 60 * k8 + 720 * k9),
            self.get_u(u, t0 + h),
            p,
            s,
        )[:, self.ode_idx]
        return x_prev + h / 840 * (41 * k1 + 27 * k4 + 272 * k5 + 27 * k6 + 216 * k7 + 216 * k9 + 41 * k10)


class TRAPEZOIDAL(Integrator):
    """
    Numerical integration using trapezoidal method.
    Not that it is only possible to have one step using trapezoidal.
    It behaves like an order 1 collocation method meaning that the integration is implicit (but since the polynomial is
    of order 1, it is not possible to put a constraint on the slopes).
    """

    def __init__(self, ode: dict, ode_opt: dict):
        self._n_step = 1
        super().__init__(ode, ode_opt)

    def next_x(
        self,
        t0: float | MX | SX,
        x_prev: MX | SX,
        x_next: MX | SX,
        u_prev: MX | SX,
        u_next: MX | SX,
        p: MX | SX,
        s_prev: MX | SX,
        s_next: MX | SX,
    ):
        dx = self.fun(t0, x_prev, u_prev, p, s_prev)[:, self.ode_idx]
        dx_next = self.fun(t0, x_next, u_next, p, s_next)[:, self.ode_idx]
        return x_prev + (dx + dx_next) * self.h / 2

    @property
    def _time_xall_from_dt_func(self) -> Function:
        return Function(
            "step_time", 
            [self.t_span_sym], 
            [linspace(self.t_span_sym[0], self.t_span_sym[0] + self.t_span_sym[1], self.shape_xall[1])]
        )
    
    @property
    def shape_xf(self):
        return [self.x_sym.shape[0], 1]
    
    @property
    def shape_xall(self):
        return [self.x_sym.shape[0], self._n_step + 1]

    @property
    def h(self):
        return self.t_span_sym[1] - self.t_span_sym[0]

    def dxdt(
        self,
        states: MX | SX,
        controls: MX | SX,
        params: MX | SX,
        stochastic_variables: MX | SX,
    ) -> tuple:

        x_prev = self.cx(states.shape[0], 2)

        states_next = states[:, 1]
        controls_prev = controls[:, 0]
        controls_next = controls[:, 1]
        if stochastic_variables.shape != (0, 0):
            stochastic_variables_prev = stochastic_variables[:, 0]
            stochastic_variables_next = stochastic_variables[:, 1]
        else:
            stochastic_variables_prev = stochastic_variables
            stochastic_variables_next = stochastic_variables

        x_prev[:, 0] = states[:, 0]

        x_prev[:, 1] = self.next_x(
            self.t_span_sym[0],
            x_prev[:, 0],
            states_next,
            controls_prev,
            controls_next,
            params,
            stochastic_variables_prev,
            stochastic_variables_next,
        )

        if self.model.nb_quaternions > 0:
            x_prev[:, 1] = self.model.normalize_state_quaternions(x_prev[:, 1])

        return x_prev[:, 1], x_prev


class COLLOCATION(Integrator):
    """
    Numerical integration using implicit Runge-Kutta method.

    Attributes
    ----------
    degree: int
        The interpolation order of the polynomial approximation

    Methods
    -------
    get_u(self, u: np.ndarray, t: float | MX | SX) -> np.ndarray
        Get the control at a given time
    """

    def _initialize(self, ode: dict, ode_opt: dict):
        """
        Parameters
        ----------
        ode: dict
            The ode description
        ode_opt: dict
            The ode options
        """
        self.method = ode_opt["method"]
        self.degree = ode_opt["irk_polynomial_interpolation_degree"]
        self.duplicate_collocation_starting_point = ode_opt["duplicate_collocation_starting_point"]
        self.allow_free_variables = ode_opt["allow_free_variables"]

        # Coefficients of the collocation equation
        self._c = self.cx.zeros((self.degree + 1, self.degree + 1))

        # Coefficients of the continuity equation
        self._d = self.cx.zeros(self.degree + 1)

        # Dimensionless time inside one control interval
        time_control_interval = self.cx.sym("time_control_interval")

        # For all collocation points
        for j in range(self.degree + 1):
            # Construct Lagrange polynomials to get the polynomial basis at the collocation point
            _l = 1
            for r in range(self.degree + 1):
                if r != j:
                    _l *= (time_control_interval - self._integration_time[r]) / (self._integration_time[j] - self._integration_time[r])

            # Evaluate the polynomial at the final time to get the coefficients of the continuity equation
            if self.method == "radau":
                self._d[j] = 1 if j == self.degree else 0
            else:
                lfcn = Function("lfcn", [time_control_interval], [_l])
                self._d[j] = lfcn(1.0)

            # Construct Lagrange polynomials to get the polynomial basis at the collocation point
            _l = 1
            for r in range(self.degree + 1):
                if r != j:
                    _l *= (time_control_interval - self._integration_time[r]) / (self._integration_time[j] - self._integration_time[r])

            # Evaluate the time derivative of the polynomial at all collocation points to get
            # the coefficients of the continuity equation
            tfcn = Function("tfcn", [time_control_interval], [tangent(_l, time_control_interval)])
            for r in range(self.degree + 1):
                self._c[j, r] = tfcn(self._integration_time[r])

    @property
    def _x_sym_modified(self):
        return horzcat(*self.x_sym) if self.duplicate_collocation_starting_point else horzcat(*self.x_sym[1:])

    @property
    def _output_names(self):
        return ["xf", "xall", "defects"]
    
    @property
    def h(self):
        return self.t_span_sym[1] - self.t_span_sym[0]

    @property
    def _integration_time(self):
        return [0] + collocation_points(self.degree, self.method)
    
    @property
    def shape_xf(self):
        return [self._x_sym_modified.shape[0], self.degree + 1]
    
    @property
    def shape_xall(self):
        return [self._x_sym_modified.shape[0], self.degree + 2]

    @property
    def _time_xall_from_dt_func(self) -> Function:
        return Function(
            "step_time", [self.t_span_sym], [self.t_span_sym[0] + (self._integration_time + [1]) * self.t_span_sym[1]]
        )

    def get_u(self, u: np.ndarray, t: float | MX | SX) -> np.ndarray:
        """
        Get the control at a given time

        Parameters
        ----------
        u: np.ndarray
            The control matrix
        t: float | MX | SX
            The time a which control should be computed

        Returns
        -------
        The control at a given time
        """

        if self.control_type == ControlType.CONSTANT or self.control_type == ControlType.CONSTANT_WITH_LAST_NODE:
            return super(COLLOCATION, self).get_u(u, t)
        else:
            raise NotImplementedError(f"{self.control_type} ControlType not implemented yet with COLLOCATION")

    def dxdt(
        self,
        states: MX | SX,
        controls: MX | SX,
        params: MX | SX,
        stochastic_variables: MX | SX,
    ) -> tuple:

        # Total number of variables for one finite element
        states_end = self._d[0] * states[1]
        defects = []
        for j in range(1, self.degree + 1):
            t = vertcat(self.t_span_sym[0] + self._integration_time[j-1] * self.h, self.h)

            # Expression for the state derivative at the collocation point
            xp_j = 0
            for r in range(self.degree + 2):
                if r == 1:
                    # We skip r=1 because this collocation point is the same as the initial point
                    continue
                xp_j += self._c[r - 1 if r > 0 else r, j] * states[r]

            if self.defects_type == DefectType.EXPLICIT:
                f_j = self.fun(
                    t,
                    states[j + 1],
                    self.get_u(controls, self._integration_time[j]),
                    params,
                    stochastic_variables,
                )[:, self.ode_idx]
                defects.append(xp_j - self.h * f_j)
            elif self.defects_type == DefectType.IMPLICIT:
                defects.append(
                    self.implicit_fun(
                        t,
                        states[j + 1],
                        self.get_u(controls, self._integration_time[j]),
                        params,
                        stochastic_variables,
                        xp_j / self.h,
                    )
                )
            else:
                raise ValueError("Unknown defects type. Please use 'explicit' or 'implicit'")

            # Add contribution to the end state
            states_end += self._d[j] * states[j + 1]

        # Concatenate constraints
        defects = vertcat(*defects)
        return states_end, horzcat(*states), defects


class IRK(COLLOCATION):
    """
    Numerical integration using implicit Runge-Kutta method.

    Methods
    -------
    get_u(self, u: np.ndarray, t: float) -> np.ndarray
        Get the control at a given time
    """

    @property
    def _x_sym_modified(self):
        return self.x_sym[0]

    @property
    def _output_names(self):
        return ["xf", "xall"]

    @property
    def shape_xf(self):
        return [self._x_sym_modified.shape[0], 1]
    
    @property
    def shape_xall(self):
        return [self._x_sym_modified.shape[0], 2]

    @property
    def _time_xall_from_dt_func(self) -> Function:
        return Function("step_time", [self.t_span_sym], [vertcat(*[self.t_span_sym[0], self.t_span_sym[0] + self.t_span_sym[1]])])

    def dxdt(
        self,
        states: MX | SX,
        controls: MX | SX,
        params: MX | SX,
        stochastic_variables: MX | SX,
    ) -> tuple:

        nx = states[0].shape[0]
        _, _, defect = super(IRK, self).dxdt(
            states=states,
            controls=controls,
            params=params,
            stochastic_variables=stochastic_variables,
        )

        # Root-finding function, implicitly defines x_collocation_points as a function of x0 and p
        collocation_states = vertcat(*states[1:]) if self.duplicate_collocation_starting_point else vertcat(*states[2:])
        vfcn = Function(
            "vfcn",
            [collocation_states, self.t_span_sym, states[0], controls, params, stochastic_variables],
            [defect]
        ).expand()

        # Create an implicit function instance to solve the system of equations
        ifcn = rootfinder("ifcn", "newton", vfcn, {"error_on_fail": False})
        t = vertcat(self.t_span_sym)  # We should not subtract here as it is already formally done in COLLOCATION
        x_irk_points = ifcn(self.cx(), t, states[0], controls, params, stochastic_variables)
        x = [states[0] if r == 0 else x_irk_points[(r - 1) * nx : r * nx] for r in range(self.degree + 1)]

        # Get an expression for the state at the end of the finite element
        xf = self.cx.zeros(nx, self.degree + 1)  # 0 #
        for r in range(self.degree + 1):
            xf[:, r] = xf[:, r - 1] + self._d[r] * x[r]

        return xf[:, -1], horzcat(states[0], xf[:, -1])


class CVODES(Integrator):
    """
    Class for CVODES integrators

    """
