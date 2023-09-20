from formulation import *


class Collocation(Formulation):
    def __init__(
        self,
        with_cov=False,
        N=40,
        nlp_solver="ipopt",
        qp_solver="IPOPT",
        with_plotting=True,
        plot_hold=True,
        init=None,
        hot_start=False,
        with_gamma=False,
        gamma=0,
        hyper=[{"a": 1, "b": 1, "x0": 0, "y0": 0, "n": 2}],
        regularisation=1e-2,
        ubound=0,
        log_results=None,
        plot_clear=False,
        plot_initial=False,
    ):
        gamma_ = gamma
        if with_plotting:
            matplotlib.interactive(True)
        model = Model()

        # superellipse   ((x-x0)/a)^n + ((y-y0)/b)^n = 1
        hyper = [{"a": 1, "b": 1, "x0": 0, "y0": 0, "n": 4}, {"a": 0.5, "b": 2, "x0": 1, "y0": 0.5, "n": 4}]

        if log_results is not None and not os.path.exists("results"):
            os.makedirs("results")

        with_cov = with_cov or with_gamma

        # Sanity checks on superellipse defintions
        for h in hyper:
            if not (isinstance(h["n"], int)):
                raise Exception("superellipse: n must be integer")
            if not (h["n"] % 2 == 0):
                raise Exception("superellipse: n must be even")

        states, controls = model.states, model.controls

        n = states.shape[0]
        m = controls.shape[0]
        p = model.dist.shape[0]
        q = len(hyper)

        x, u, w = states[...], controls[...], model.dist[...]

        param = Collection()
        param.m = model.parameters
        if with_cov:
            Sigma_ww = param.Sigma_ww = SX.sym("Sigma_ww", p, p)
        T = param.T = SX.sym("T")
        if with_gamma:
            gamma = param.gamma = SX.sym("gamma")

        param.freeze()
        pp = param[...]

        # ============= Collocation on one control interval =============

        # Non-dimensional time-base running from 0 to 1 on collocation interval
        rho = SX.sym("rho")

        # Choose collocation points
        rho_root = [0, 0.046910, 0.230765, 0.500000, 0.769235, 0.953090]

        # Degree of interpolating polynomial + 1
        d = len(rho_root) - 1

        # The helper state sample used by collocation
        z = SX.sym("z", n, d + 1)

        # Collocation polynomial
        Le = vertcat(
            [
                numpy.prod([(rho - rho_root[r]) / (rho_root[j] - rho_root[r]) for r in range(d + 1) if not (r == j)])
                for j in range(d + 1)
            ]
        )
        Pi = SXFunction([rho, z], [mul(z, Le)])
        Pi.init()

        dPi = Pi.jacobian(0, 0)
        dPi.init()

        # System dynamics
        f = SXFunction([x, u, w, pp], [model.rhs[...]])
        f.init()

        # The function G in 0 = G(x_k,z_k,u_k,w_k)
        G_argout = []
        G_argout += [Pi.eval([0, z])[0] - x]
        G_argout += [
            dPi.eval([rho_i, z])[0] - T / N * f.eval([Pi.eval([rho_i, z])[0], u, w, pp])[0] for rho_i in rho_root[1:]
        ]

        G = SXFunction([x, z, u, w, pp], [vertcat(G_argout)])
        G.init()

        # The function F in x_{k+1} = F(z_k)
        F = SXFunction([z], Pi.eval([1, z]))
        F.init()

        # ======== Covariance-related function on one control interval ========
        if with_cov:
            P = SX.sym("P", n, n)
            M = SX.sym("M", n, n * (d + 1))

            Gdx, Gdz, Gdw = G.jac(0, 0), G.jac(1, 0), G.jac(2, 0)
            Fdz = F.jac(0, 0)

            # Covariance propagation rule
            Pf = SXFunction([x, z, u, w, pp, P, M], [mul([M, mul([Gdx, P, Gdx.T]) + mul([Gdw, Sigma_ww, Gdw.T]), M.T])])
            Pf.init()

            # Equality defining M
            Mf = SXFunction([x, z, u, w, pp, P, M], [Fdz.T - mul(Gdz.T, M.T)])
            Mf.init()

        # ==== Superellipsoid constraints on one control interval =====

        # The superellipsoid constraints

        # h = SXFunction([x],[vertcat([sqrt((x[states.i_x]-h["x0"])**2+(x[states.i_y]-h["y0"])**2)*(1 - (((x[states.i_x]-h["x0"])/h["a"])**h["n"] +  ((x[states.i_y]-h["y0"])/h["b"])**h["n"])**(-1.0/h["n"])) for h in hyper ])])
        # h.init()

        # h = SXFunction([x],[vertcat([(((x[states.i_x]-h["x0"])/h["a"])**h["n"] +  ((x[states.i_y]-h["y0"])/h["b"])**h["n"])**(1/h["n"]) - 1 for h in hyper ])])

        h = SXFunction(
            [x],
            [
                vertcat(
                    [
                        ((x[states.i_x] - h["x0"]) / h["a"]) ** h["n"]
                        + ((x[states.i_y] - h["y0"]) / h["b"]) ** h["n"]
                        - 1
                        for h in hyper
                    ]
                )
            ],
        )
        h.init()

        hdx = h.jac(0, 0)
        hx = h.eval([x])[0]

        if with_gamma:
            # The robustified superellipsoid constraints
            hr = SXFunction(
                [x, pp, P], [vertcat([hx[i, 0] - gamma * sqrt(mul([hdx[i, :], P, hdx[i, :].T])) for i in range(q)])]
            )
            hr.init()

        # ============= OCP Formulation =============

        V = Collection()  # Collection of decision variables
        V.Z = [MX.sym("Z[%d]" % i, n, d + 1) for i in range(N)]
        V.X = [MX.sym("X[%d]" % i, n) for i in range(N + 1)]
        V.U = [MX.sym("U[%d]" % i, m, 1) for i in range(N)]
        if with_cov:
            V.P = [MX.sym("P[%d]" % i, n, n) for i in range(N + 1)]
            V.M = [MX.sym("M[%d]" % i, n, n * (d + 1)) for i in range(N)]

        PP = Collection()
        PP.m = MX.sym("p", param.m.shape)
        if with_cov:
            PP.Sigma_ww = MX.sym("Sigma_ww", param.Sigma_ww.shape)
        PP.T = MX.sym("T", param.T.shape)
        if with_gamma:
            PP.gamma = MX.sym("gamma", param.gamma.shape)
        PP.freeze()
        V.PP = PP

        if with_cov:
            V.setOrder(["PP", ("X", "Z", "P", "M", "U")])
        else:
            V.setOrder(["PP", ("X", "Z", "U")])

        V.freeze()

        # Zero nominal disturbance
        w0 = DMatrix.zeros(w.shape)

        g = Collection()  # Collection of constraints
        g.F = [F.call([V.Z[k]])[0] - V.X[k + 1] for k in range(N)]
        g.G = [G.call([V.X[k], V.Z[k], V.U[k], w0, V.PP[...]])[0] for k in range(N)]
        if with_cov:
            g.P = [Pf.call([V.X[k], V.Z[k], V.U[k], w0, V.PP[...], V.P[k], V.M[k]])[0] - V.P[k] for k in range(N)]
            g.M = [Mf.call([V.X[k], V.Z[k], V.U[k], w0, V.PP[...], V.P[k], V.M[k]])[0] for k in range(N)]
        g.periodic = V.X[0] - V.X[-1]
        if with_cov:
            g.periodic_P = V.P[0] - V.P[-1]
        g.fix = V.X[0][states.i_x]
        if with_gamma:
            g.hyper = [hr.call([V.X[k], V.PP[...], V.P[k]])[0] for k in range(N)]
        else:
            g.hyper = [h.call([V.X[k]])[0] for k in range(N)]
        g.freeze()

        gf = MXFunction([V[...]], [g[...]])
        gf.init()

        ff = MXFunction([V[...]], [V.PP.T + regularisation * sumAll(vertcat(V.U) ** 2) / 2 / N])
        ff.init()

        # ===========================================
        # ===========================================

        class NLPSolutionInspector:
            def __init__(self):
                self.iter = 0
                self.log = numpy.zeros((4, 1000))
                self.colors = list("bgrcmyk" * 5)

                if plot_clear:
                    figure(1)
                    clf()
                    figure(2)
                    clf()
                figure(1)

                subplot(111)
                theta = linspace(0, 2 * pi, 1000)
                for h in hyper:
                    fill(
                        h["a"] * abs(cos(theta)) ** (2.0 / h["n"]) * sign(cos(theta)) + h["x0"],
                        h["b"] * abs(sin(theta)) ** (2.0 / h["n"]) * sign(sin(theta)) + h["y0"],
                        "r",
                    )
                title("X(t)")

                # A sampled circle, used to draw uncertainty ellipsoid
                self.circle = array([[sin(x), cos(x)] for x in linspace(0, 2 * pi, 100)]).T

            def __call__(self, f, *args):
                sol = f.input(NLP_X_OPT)

                X_opt = horzcat([sol[i] for i in V.i_X])
                Z_opt = horzcat([sol[i] for i in V.i_Z])
                U_opt = horzcat([sol[i] for i in V.i_U])

                if hasattr(self, "xlines"):
                    self.xlines[0].set_xdata(X_opt[states.i_x, :].T)
                    self.xlines[0].set_ydata(X_opt[states.i_y, :].T)
                    self.xlines[1].set_xdata(X_opt[states.i_x, 0].T)
                    self.xlines[1].set_ydata(X_opt[states.i_y, 0].T)

                    for k, v in enumerate(fabs(f.input(NLP_LAMBDA_G)[veccat(g.i_hyper)]).data()):
                        j = k % len(hyper)
                        i = (k - j) / len(hyper)
                        if v > 1e-6:
                            self.glines[i][j].set_xdata(DMatrix([hyper[j]["x0"], X_opt[states.i_x, i]]))
                            self.glines[i][j].set_ydata(DMatrix([hyper[j]["y0"], X_opt[states.i_y, i]]))
                        else:
                            self.glines[i][j].set_xdata(DMatrix.nan(2))
                            self.glines[i][j].set_ydata(DMatrix.nan(2))
                    self.zlines[0].set_xdata(Z_opt[states.i_x, :].T)
                    self.zlines[0].set_ydata(Z_opt[states.i_y, :].T)
                    self.zlines[1].set_xdata(Z_opt[states.i_x, 0].T)
                    self.zlines[1].set_ydata(Z_opt[states.i_y, 0].T)
                    for k in range(N):
                        Pi.input(1).set(sol[V.i_Z[k]])
                        s_ = numSample1D(Pi, DMatrix(linspace(0.0, 1, 20)).T)
                        self.smoothlines[k].set_xdata(s_[states.i_x, :].T)
                        self.smoothlines[k].set_ydata(s_[states.i_y, :].T)
                    self.ulines[0].set_xdata(U_opt[0, IMatrix(range(N) + [0])].T)
                    self.ulines[0].set_ydata(U_opt[1, IMatrix(range(N) + [0])].T)
                    self.xlines2[0].set_xdata(X_opt[states.i_x, :].T)
                    self.xlines2[0].set_ydata(X_opt[states.i_y, :].T)
                    for k in range(N):
                        self.strings[k].set_xdata(horzcat([X_opt[states.i_x, k], U_opt[0, k]]))
                        self.strings[k].set_ydata(horzcat([X_opt[states.i_y, k], U_opt[1, k]]))
                    if with_cov:
                        for k in range(N):
                            Pxy = sol[V.i_P[k]][states.iv_x + states.iv_y, states.iv_x + states.iv_y]
                            w, v = numpy.linalg.eig(Pxy)
                            W = mul(v, diag(sqrt(w)))
                            e = mul(W, self.circle)
                            self.plines[k].set_xdata(e[0, :].T + X_opt[states.i_x, k])
                            self.plines[k].set_ydata(e[1, :].T + X_opt[states.i_y, k])
                else:
                    figure(1)
                    subplot(111)
                    if plot_initial:
                        # Draw the initial guess
                        for k in range(N):
                            Pi.input(1).set(sol[V.i_Z[k]])
                            s_ = numSample1D(Pi, DMatrix(linspace(0.0, 1, 20)).T)
                            plot(s_[states.i_x, :].T, s_[states.i_y, :].T, "g")[0]
                    self.xlines = []
                    self.xlines.append(plot(X_opt[states.i_x, :].T, X_opt[states.i_y, :].T, "ko")[0])
                    self.xlines.append(plot(X_opt[states.i_x, 0].T, X_opt[states.i_y, 0].T, "ko", markersize=10)[0])
                    self.zlines = []
                    self.zlines.append(plot(Z_opt[states.i_x, :].T, Z_opt[states.i_y, :].T, "b.")[0])
                    self.zlines.append(plot(Z_opt[states.i_x, 0], Z_opt[states.i_y, 0], "bo")[0])
                    self.smoothlines = []
                    for k in range(N):
                        Pi.input(1).set(sol[V.i_Z[k]])
                        s_ = numSample1D(Pi, DMatrix(linspace(0.0, 1, 20)).T)
                        self.smoothlines.append(plot(s_[states.i_x, :].T, s_[states.i_y, :].T, "g")[0])
                    if with_cov:
                        self.plines = []
                        for k in range(N):
                            Pxy = sol[V.i_P[k]][states.iv_x + states.iv_y, states.iv_x + states.iv_y]
                            w, v = numpy.linalg.eig(Pxy)
                            W = mul(v, diag(sqrt(w)))
                            e = mul(W, self.circle)
                            self.plines.append(
                                plot(e[0, :].T + X_opt[states.i_x, k], e[1, :].T + X_opt[states.i_y, k], "k")[0]
                            )
                    self.glines = []
                    for i in range(N):
                        self.glines.append([])
                        for j in range(len(hyper)):
                            self.glines[i].append(plot(DMatrix.nan(2), DMatrix.nan(2), "k")[0])
                    figure(2)
                    subplot(111)
                    self.ulines = []
                    self.ulines.append(
                        plot(U_opt[0, IMatrix(range(N) + [0])].T, U_opt[1, IMatrix(range(N) + [0])].T, "bo-")[0]
                    )
                    self.xlines2 = []
                    self.xlines2.append(plot(X_opt[states.i_x, :].T, X_opt[states.i_y, :].T, "ko")[0])
                    self.strings = []
                    for k in range(N):
                        self.strings.append(
                            plot(
                                horzcat([X_opt[states.i_x, k], U_opt[0, k]]),
                                horzcat([X_opt[states.i_y, k], U_opt[1, k]]),
                                "k",
                            )[0]
                        )

                figure(1)
                subplot(111).relim()
                subplot(111).autoscale_view()
                if log_results is not None:
                    gcf().savefig("results/%s_x_%03d.eps" % (log_results, self.iter), format="eps")
                figure(2)
                subplot(111).relim()
                subplot(111).autoscale_view()
                if log_results is not None:
                    gcf().savefig("results/%s_u_%03d.eps" % (log_results, self.iter), format="eps")
                draw()

                # reportBounds(solver.output(NLP_G),solver.input(NLP_LBG),solver.output(NLP_UBG),g.getLabels())
                # raw_input()

                # print f.input(NLP_LAMBDA_G)[veccat(g.i_hyper)]
                # print fabs(f.input(NLP_LAMBDA_G)[veccat(g.i_hyper)])>1e-8

                self.iter += 1

        iterationInspector = NLPSolutionInspector()

        #! We wrap the logging instance in a PyFunction
        c = PyFunction(
            iterationInspector,
            nlpsolverOut(
                x_opt=sp_dense(V.shape),
                cost=sp_dense(1, 1),
                lambda_x=sp_dense(V.shape),
                lambda_g=sp_dense(g.shape),
                g=sp_dense(g.shape),
            ),
            [sp_dense(1, 1)],
        )
        c.init()

        QPSolvers = {
            "OOQP": (OOQPSolver, {}),
            "IPOPT": (NLPQPSolver, {"nlp_solver": IpoptSolver, "nlp_solver_options": {}}),
            "CPLEX": (CplexSolver, {}),
        }

        qp_solver_ = QPSolvers[qp_solver]

        Solvers = {
            "ipopt": (
                IpoptSolver,
                {"tol": 1e-10, "linear_solver": "ma57"},
                {
                    "mu_init": 1e-11,
                    "warm_start_bound_push": 1e-11,
                    "warm_start_mult_bound_push": 1e-11,
                    "warm_start_init_point": "yes",
                },
            ),
            "worhp": (WorhpSolver, {"TolOpti": 1e-9, "UserHM": True}, {"InitialLMest": False}),
            "sqp": (SQPMethod, {"qp_solver": qp_solver_[0], "qp_solver_options": qp_solver_[1]}, {}),
            "knitro": (KnitroSolver, {}, {}),
        }

        solver_ = Solvers[nlp_solver]
        solver = solver_[0](ff, gf)
        solver.setOption("expand_f", True)
        solver.setOption("expand_g", True)
        solver.setOption("generate_hessian", True)
        if with_plotting:
            solver.setOption("iteration_callback", c)
        solver.setOption(solver_[1])
        if hot_start and (init is not None):
            solver.setOption(solver_[2])
        # solver.setOption("mu_strategy","monotone")

        print("Init solver start")
        solver.init()
        print("Init solver end")

        figure(3)
        matshow(DMatrix(solver.getH().output().sparsity(), 1))
        show()

        # Time must be positive
        solver.input(NLP_LBX)[V.i_PP.T] = 0

        solver.input(NLP_LBG).setAll(0)
        solver.input(NLP_UBG).setAll(0)

        if ubound > 0:
            solver.input(NLP_LBX)[vertcat(V.i_U)] = -ubound
            solver.input(NLP_UBX)[vertcat(V.i_U)] = ubound

        solver.input(NLP_UBG)[vertcat(g.i_hyper)] = Inf
        solver.input(NLP_LBX)[V.i_PP.m] = model.parameters_
        solver.input(NLP_UBX)[V.i_PP.m] = model.parameters_
        if with_cov:
            sww = diag([1] * p)
            try:
                makeDense(sww)
            except:
                pass
            solver.input(NLP_LBX)[V.i_PP.Sigma_ww] = sww
            solver.input(NLP_UBX)[V.i_PP.Sigma_ww] = sww
        if with_gamma:
            solver.input(NLP_LBX)[V.i_PP.gamma] = gamma_
            solver.input(NLP_X_INIT)[V.i_PP.gamma] = gamma_
            solver.input(NLP_UBX)[V.i_PP.gamma] = gamma_

        if init is None:  # Initalize from scratch
            T_ = 4.0
            solver.input(NLP_X_INIT)[V.i_PP.T] = T_
            solver.input(NLP_X_INIT)[V.i_PP.m] = model.parameters_

            # Initialize in circle
            for k in range(N):
                for j in range(d + 1):
                    t = T_ * ((k + 0.0) / N + (rho_root[j] + 0.0) / N)
                    solver.input(NLP_X_INIT)[V.i_Z[k][states.iv_x, j]] = 3 * sin(2 * pi * t / T_)
                    solver.input(NLP_X_INIT)[V.i_Z[k][states.iv_y, j]] = 3 * cos(2 * pi * t / T_)

            for k in range(N + 1):
                t = T_ * (k + 0.0) / N
                solver.input(NLP_X_INIT)[V.i_X[k][states.iv_x]] = 3 * sin(2 * pi * t / T_)
                solver.input(NLP_X_INIT)[V.i_X[k][states.iv_y]] = 3 * cos(2 * pi * t / T_)

        else:  # Initialize from another object
            isolver = init.solver
            solver.input(NLP_X_INIT)[V.i_PP.T] = isolver.output(NLP_X_OPT)[init.V.i_PP.T]
            solver.output(NLP_LAMBDA_X)[V.i_PP.T] = isolver.output(NLP_LAMBDA_X)[init.V.i_PP.T]
            solver.input(NLP_X_INIT)[V.i_PP.m] = isolver.output(NLP_X_OPT)[init.V.i_PP.m]
            solver.output(NLP_LAMBDA_X)[V.i_PP.m] = isolver.output(NLP_LAMBDA_X)[init.V.i_PP.m]

            for k in range(N + 1):
                solver.input(NLP_X_INIT)[V.i_X[k]] = isolver.output(NLP_X_OPT)[init.V.i_X[k]]
                solver.output(NLP_LAMBDA_X)[V.i_X[k]] = isolver.output(NLP_LAMBDA_X)[init.V.i_X[k]]
            for k in range(N):
                solver.input(NLP_X_INIT)[V.i_U[k]] = isolver.output(NLP_X_OPT)[init.V.i_U[k]]
                solver.output(NLP_LAMBDA_X)[V.i_U[k]] = isolver.output(NLP_LAMBDA_X)[init.V.i_U[k]]
                solver.input(NLP_X_INIT)[V.i_Z[k]] = isolver.output(NLP_X_OPT)[init.V.i_Z[k]]
                solver.output(NLP_LAMBDA_X)[V.i_Z[k]] = isolver.output(NLP_LAMBDA_X)[init.V.i_Z[k]]

            if with_cov and init.with_cov:
                for k in range(N + 1):
                    solver.input(NLP_X_INIT)[V.i_P[k]] = isolver.output(NLP_X_OPT)[init.V.i_P[k]]
                    solver.output(NLP_LAMBDA_X)[V.i_P[k]] = isolver.output(NLP_LAMBDA_X)[init.V.i_P[k]]
                for k in range(N):
                    solver.input(NLP_X_INIT)[V.i_P[k]] = isolver.output(NLP_X_OPT)[init.V.i_P[k]]
                    solver.output(NLP_LAMBDA_X)[V.i_P[k]] = isolver.output(NLP_LAMBDA_X)[init.V.i_P[k]]

            for k in range(N):
                solver.input(NLP_LAMBDA_INIT)[g.i_F[k]] = isolver.output(NLP_LAMBDA_G)[init.g.i_F[k]]
                solver.input(NLP_LAMBDA_INIT)[g.i_G[k]] = isolver.output(NLP_LAMBDA_G)[init.g.i_G[k]]
                solver.input(NLP_LAMBDA_INIT)[g.i_hyper[k]] = isolver.output(NLP_LAMBDA_G)[init.g.i_hyper[k]]
            solver.input(NLP_LAMBDA_INIT)[g.i_fix] = isolver.output(NLP_LAMBDA_G)[init.g.i_fix]
            solver.input(NLP_LAMBDA_INIT)[g.i_periodic] = isolver.output(NLP_LAMBDA_G)[init.g.i_periodic]

        solver.solve()

        if log_results is not None:
            figure(1).savefig("results/%s_x.eps" % log_results, format="eps")
            figure(2).savefig("results/%s_u.eps" % log_results, format="eps")

        if with_plotting and plot_hold:
            matplotlib.interactive(False)
            show()

        print(solver.output(NLP_LAMBDA_G)[veccat(g.i_hyper)])

        # Make public
        self.solver = solver
        self.V = V
        self.param = param
        self.g = g
        self.with_cov = with_cov
        self.states = states
        self.h = h
        if with_gamma:
            self.hr = hr


if __name__ == "__main__":
    Collocation(N=40, with_cov=False, with_plotting=True)
