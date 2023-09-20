import casadi as cas

class Model:
    def __init__(self):

        # States
        x = cas.SX.sym("x")
        y = cas.SX.sym("y")
        dx = cas.SX.sym("dx")
        dy = cas.SX.sym("dy")
        p = cas.vertcat(x, y)
        v = cas.vertcat(dx, dy)
        states_sym = cas.MX.sym("states", 4, 1)
        states = {"x": x, "y": y, "dx": dx, "dy": dy, "states": cas.vertcat(p, v)}

        # Controls
        u = cas.SX.sym("u")
        v = cas.SX.sym("v")
        q = cas.vertcat(u, v)
        controls_sym = cas.MX.sym("controls", 2, 1)
        controls = {"u": u, "v": v, "controls": q}

        # Motor noise
        wx = cas.SX.sym("wx")
        wy = cas.SX.sym("wy")
        w = cas.vertcat(wx, wy)
        dist_sym = cas.MX.sym("dist", 2, 1)
        dist = {"wx": wx, "wy": wy, "dist": w}

        # Parmeters
        k = cas.SX.sym("k")
        c = cas.SX.sym("c")
        beta = cas.SX.sym("beta")
        param = cas.vertcat(k, c, beta)
        parameters = {"k": k, "c": c, "beta": beta, "parameters": param}

        k = 10
        c = 1
        beta = 1
        parameters_num = {"k": k, "c": c, "beta": beta}

        # Dynamincs
        F = -k * (states_sym[:2] - controls_sym) - beta * states_sym[2:] * cas.sqrt(cas.sum1(states_sym[2:]**2) + c**2) + dist_sym
        derivative_x = states_sym[2]
        derivative_y = states_sym[3]
        derivative_dx = F[0]
        derivative_dy = F[1]
        derivative_fun = cas.Function("derivative", [states_sym, controls_sym, dist_sym], [cas.vertcat(derivative_x, derivative_y, derivative_dx, derivative_dy)])
        sys = {"derivative_x": derivative_x, "derivative_y": derivative_y, "derivative_dx": derivative_dx, "derivative_dy": derivative_dy, "derivative_fun": derivative_fun}

        self.rhs = sys
        self.states = states
        self.controls = controls
        self.dist = dist
        self.parameters = parameters
        self.parameters_num = parameters_num


if __name__ == "__main__":
    m = Model()
    print(m.rhs)
