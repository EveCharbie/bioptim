import casadi as cas

class Model:
    def __init__(self):

        # States
        x = cas.SX.sym("x")
        y = cas.SX.sym("y")
        dx = cas.SX.sym("dx")
        dy = cas.SX.sym("dy")
        states = {"x": x, "y": y, "dx": dx, "dy": dy}

        # Controls
        u = cas.SX.sym("u")
        v = cas.SX.sym("v")
        controls = {"u": u, "v": v}

        # Motor noise
        wx = cas.SX.sym("wx")
        wy = cas.SX.sym("wy")
        noise = {"wx": wx, "wy": wy}

        # Parmeters
        k = cas.SX.sym("k")
        c = cas.SX.sym("c")
        beta = cas.SX.sym("beta")
        parameters = {"k": k, "c": c, "beta": beta}

        k = 10
        c = 1
        beta = 1
        parameters_num = {"k": k, "c": c, "beta": beta}

        # Vertcats
        p = cas.vertcat([x, y])
        q = cas.vertcat([u, v])
        w = cas.vertcat([wx, wy])
        v = cas.vertcat([dx, dy])

        # Dynamincs
        F = -k * (p - q) - beta * v * cas.sqrt(cas.sum1(v**2) + c**2) + w
        derivative_x = dx
        derivative_y = dy
        derivative_dx = F[0]
        derivative_dy = F[1]
        sys = {"derivative_x": derivative_x, "derivative_y": derivative_y, "derivative_dx": derivative_dx, "derivative_dy": derivative_dy}

        self.rhs = sys
        self.states = states
        self.controls = controls
        self.dist = noise
        self.parameters = parameters
        self.parameters_num = parameters_num


if __name__ == "__main__":
    m = Model()
    print(m.rhs)
