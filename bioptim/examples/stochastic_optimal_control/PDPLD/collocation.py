from formulation import *


def Collocation(with_cov=False,
                N=40,
                # nlp_solver="ipopt",
                # qp_solver="IPOPT",
                with_plotting=True,
                plot_hold=True,
                last_sol_for_init=None,
                # hot_start=False,
                with_gamma=False,
                gamma=0,
                hyper=[{"a": 1, "b": 1, "x0": 0, "y0": 0, "n": 2}],
                regularisation=1e-2,
                ubound=0,
                log_results=None,
                plot_clear=False,
                plot_initial=False):
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

    n = len(model.states.keys())-1
    m = len(model.controls.keys())-1
    p = len(model.dist.keys())-1
    q = len(hyper)

    x, u, w = model.states["states"], model.controls["controls"], model.dist["dist"]
    Sigma_ww = cas.SX_eye(2) * w

    # param_m = model.parameters["parameters"]
    # pp = cas.vertcat(param_m)
    # if with_cov:
    #     Sigma_ww = cas.SX.sym("Sigma_ww", p, p)
    #     # pp = cas.vertcat(pp, Sigma_ww)
    T = cas.SX.sym("T")
    # pp = cas.vertcat(pp, T)
    # if with_gamma:
    #     gamma = cas.SX.sym("gamma")
    #     pp = cas.vertcat(pp, gamma)

    # ============= Collocation on one control interval =============

    # Non-dimensional time-base running from 0 to 1 on collocation interval
    rho = cas.SX.sym("rho")

    # Choose collocation points
    rho_root = [0, 0.046910, 0.230765, 0.500000, 0.769235, 0.953090]

    # Degree of interpolating polynomial + 1
    d = len(rho_root) - 1

    # The helper state sample used by collocation
    z = cas.SX.sym("z", n, d + 1)

    # Collocation polynomial
    Le = cas.DM()
    for j in range(d + 1):
        coeff = 1
        for r in range(d + 1):
            if r != j:
                coeff *= (rho - rho_root[r]) / (rho_root[j] - rho_root[r])
        Le = cas.vertcat(Le, coeff)

    Pi = cas.Function("Pi", [rho, z], [z @ Le])
    dPi = cas.Function("jac_Pi_rho", [rho, z], [cas.jacobian(Pi(rho, z), rho)])

    # System dynamics

    f = cas.Function("dStates_dt", [x, u, w], [model.rhs["derivative_fun"](x, u, w)])

    # The function G in 0 = G(x_k,z_k,u_k,w_k)
    G_argout = []
    G_argout += [Pi(0, z) - x]
    for rho_i in rho_root[1:]:
        G_argout += [dPi(rho_i, z) - T / N * f(Pi(rho_i, z), u, w)]
    G = cas.Function("G", [x, z, u, w, T], [cas.vertcat(*G_argout)])

    # The function F in x_{k+1} = F(z_k)
    F = cas.Function("F", [z], [Pi(1, z)])

    # ======== Covariance-related function on one control interval ========
    if with_cov:
        P = cas.SX.sym("P", n, n)
        M = cas.SX.sym("M", n, n * (d + 1))

        Gdx = cas.jacobian(G(x, z, u, w, T), x)
        Gdz = cas.jacobian(G(x, z, u, w, T), z)
        Gdw = cas.jacobian(G(x, z, u, w, T), w)
        Fdz = cas.jacobian(F(z), z)

        # Covariance propagation rule
        Pf = cas.Function("P_next", [x, z, u, w, T, P, M], [M @ (Gdx @ P @ Gdx.T + Gdw @ Sigma_ww @ Gdw.T) @ M.T])

        # Equality defining M
        Mf = cas.Function("M_plus", [x, z, u, w, T, P, M], [Fdz.T - Gdz.T @ M.T])

    # ==== Superellipsoid constraints on one control interval =====

    superellipsoid_constraint = cas.SX()
    for h in hyper:
        superellipsoid_constraint = cas.vertcat(superellipsoid_constraint, ((x[0] - h["x0"]) / h["a"]) ** h["n"] + ((x[1] - h["y0"]) / h["b"]) ** h["n"] - 1)
    h = cas.Function("superellipsoid_constraint", [x], [superellipsoid_constraint])

    hdx = cas.jacobian(h(x), x)
    hx = h(x)

    if with_gamma:
        # The robustified superellipsoid constraints
        superellipsoid_constraint = cas.SX()
        for i in range(q):
            superellipsoid_constraint = cas.vertcat(superellipsoid_constraint, hx[i, 0] - gamma * cas.sqrt(hdx[i, :] @ P @ hdx[i, :].T))
        hr = cas.Function("robustified_superellipsoid_constraint", [x, P], [superellipsoid_constraint])

    # ============= OCP Formulation =============

    # Decision variables
    variables = cas.MX()
    model.i_x = list(range(variables.shape[0], variables.shape[0] + n * (N + 1)))
    V_X = [cas.MX.sym(f"X[{i}]", n) for i in range(N + 1)]
    variables = cas.vertcat(variables, *V_X)
    current_n_z = 0
    V_Z = []
    model.i_z = [[] for i in range(N)]
    reshaped_z = cas.MX()
    for i in range(N):
        model.i_z[i] = list(range(variables.shape[0] + current_n_z, variables.shape[0] + current_n_z + n * (d + 1)))
        current_n_z += n * (d + 1)
        V_Z += [[cas.MX.sym(f"Z[{i},{j}]", n, 1) for j in range(d + 1)]]
        reshaped_z = cas.vertcat(reshaped_z, cas.vertcat(V_Z[i][0], V_Z[i][1], V_Z[i][2], V_Z[i][3], V_Z[i][4], V_Z[i][5]))
    variables = cas.vertcat(variables, reshaped_z)
    V_P = []
    model.i_p = [[] for i in range(N+1)]
    reshaped_p = cas.MX()
    V_M = []
    model.i_m = [[] for i in range(N+1)]
    reshaped_m = cas.MX()
    if with_cov:
        current_n_p = 0
        for i in range(N + 1):
            model.i_p[i] = list(range(variables.shape[0] + current_n_p, variables.shape[0] + current_n_p + n * n))
            current_n_p += n * n
            V_P += [[cas.MX.sym(f"P[{i},{j}]", n, 1) for j in range(n)]]
            reshaped_p = cas.vertcat(reshaped_p, cas.vertcat(V_P[i][0], V_P[i][1], V_P[i][2], V_P[i][3]))
        variables = cas.vertcat(variables, reshaped_p)
        current_n_m = 0
        for i in range(N + 1):
            model.i_m[i] = list(range(variables.shape[0] + current_n_m, variables.shape[0] + current_n_m + n * n* (d + 1)))
            current_n_m += n * n * (d + 1)
            V_M += [[cas.MX.sym(f"M[{i},{j}]", n, 1) for j in range(n*(d+1))]]
            for j in range(n * (d+1)):
                reshaped_m = cas.vertcat(reshaped_m, cas.vertcat(V_M[i][j]))
        variables = cas.vertcat(variables, reshaped_m)
    model.i_u = list(range(variables.shape[0], variables.shape[0] + m * N))
    V_U = [cas.MX.sym(f"U[{i}]", m, 1) for i in range(N)]
    variables = cas.vertcat(variables, *V_U)
    model.i_t = variables.shape[0]
    T = cas.MX.sym("T", T.shape)
    variables = cas.vertcat(variables, T)
    V = {"X": V_X, "Z": V_Z, "P": V_P, "M": V_M, "U": V_U, "T": T, "V": variables}

    # PP_params = cas.MX()
    # PP_m = cas.MX.sym("p", param_m.shape[0])
    # PP_params = cas.vertcat(PP_params, PP_m)
    # PP_Sigma_ww = None
    # if with_cov:
    #     PP_Sigma_ww = cas.MX.sym("Sigma_ww", Sigma_ww.shape)
    #     PP_params = cas.vertcat(PP_params, PP_Sigma_ww)
    # PP_T = cas.MX.sym("T", T.shape)
    # PP_params = cas.vertcat(PP_params, PP_T)
    # PP_gamma = None
    # if with_gamma:
    #     PP_gamma = cas.MX.sym("gamma", gamma.shape)
    #     PP_params = cas.vertcat(PP_params, PP_gamma)
    # PP = {"PP_m": PP_m, "PP_Sigma_ww": PP_Sigma_ww, "PP_T": PP_T, "PP_gamma": PP_gamma, "PP": PP_params}
    # V["PP"] = PP["PP"]
    # V["T"] = PP_T

    # Zero nominal disturbance
    w0 = cas.DM.zeros(w.shape)

    # Constraints
    gf = cas.MX()
    model.i_g_f = list(range(gf.shape[0], gf.shape[0] + n * N))
    g_f = [F(cas.horzcat(*V["Z"][k])) - V["X"][k + 1] for k in range(N)]
    gf = cas.vertcat(gf, *g_f)
    model.i_g_g = list(range(gf.shape[0], gf.shape[0] + n * (d + 1) * N))
    g_g = [G(V["X"][k], cas.horzcat(*V["Z"][k]), V["U"][k], w0, T) for k in range(N)]
    gf = cas.vertcat(gf, *g_g)
    model.i_g_p = None
    model.i_g_m = None
    if with_cov:
        model.i_g_p = list(range(gf.shape[0], gf.shape[0] + n * n * (N + 1)))
        g_P = cas.MX()
        for k in range(N):
            p_constraint = Pf(V["X"][k], cas.horzcat(*V["Z"][k]), V["U"][k], w0, T, cas.horzcat(*V["P"][k]), cas.horzcat(*V["M"][k])) - cas.horzcat(*V["P"][k])
            g_P = cas.vertcat(g_P, cas.vertcat(p_constraint[:, 0], p_constraint[:, 1], p_constraint[:, 2], p_constraint[:, 3]))
        gf = cas.vertcat(gf, g_P)
        model.i_g_m = list(range(gf.shape[0], gf.shape[0] + n * n * (d + 1) * N))
        g_M = cas.MX()
        for k in range(N):
            m_constraint = Mf(V["X"][k], cas.horzcat(*V["Z"][k]), V["U"][k], w0, T, cas.horzcat(*V["P"][k]), cas.horzcat(*V["M"][k]))
            g_M = cas.vertcat(g_M, cas.vertcat(m_constraint[:, 0], m_constraint[:, 1], m_constraint[:, 2], m_constraint[:, 3]))
        gf = cas.vertcat(gf, g_M)
    model.i_g_periodic = list(range(gf.shape[0], gf.shape[0] + n))
    g_periodic = V["X"][0] - V["X"][-1]
    gf = cas.vertcat(gf, g_periodic)
    model.i_g_periodic_p = None
    if with_cov:
        model.i_g_periodic_p = list(range(gf.shape[0], gf.shape[0] + n * n))
        g_periodic_P = cas.horzcat(*V["P"][0]) - cas.horzcat(*V["P"][-1])
        gf = cas.vertcat(gf, cas.vertcat(g_periodic_P[:, 0], g_periodic_P[:, 1], g_periodic_P[:, 2], g_periodic_P[:, 3]))
    model.i_g_fix = list(range(gf.shape[0], gf.shape[0] + n))
    g_fix = V["X"][0][0]
    gf = cas.vertcat(gf, g_fix)
    model.i_g_hyper = None
    if with_gamma:
        model.i_g_hyper = list(range(gf.shape[0], gf.shape[0] + q * N))
        g_hyper = [hr(V["X"][k], cas.horzcat(*V["P"][k])) for k in range(N)]
        gf = cas.vertcat(gf, cas.vertcat(*g_hyper))
    else:
        model.i_g_hyper = list(range(gf.shape[0], gf.shape[0] + q * N))
        g_hyper = [h(V["X"][k]) for k in range(N)]
        gf = cas.vertcat(gf, *g_hyper)
    # gf = cas.Function("all_g", [V["X"]], [g])

    ff = V["T"] + regularisation * cas.sum1(cas.vertcat(*V["U"]) ** 2) / 2 / N

    # ===========================================
    # ===========================================

    # class NLPSolutionInspector:
    #     def __init__(self):
    #         self.iter = 0
    #         self.log = np.zeros((4, 1000))
    #         self.colors = list("bgrcmyk" * 5)
    #
    #         if plot_clear:
    #             figure(1)
    #             clf()
    #             figure(2)
    #             clf()
    #         figure(1)
    #
    #         subplot(111)
    #         theta = np.linspace(0, 2 * np.pi, 1000)
    #         for h in hyper:
    #             fill(
    #                 h["a"] * np.abs(np.cos(theta)) ** (2.0 / h["n"]) * sign(np.cos(theta)) + h["x0"],
    #                 h["b"] * np.abs(np.sin(theta)) ** (2.0 / h["n"]) * sign(np.sin(theta)) + h["y0"],
    #                 "r",
    #             )
    #         title("X(t)")
    #
    #         # A sampled circle, used to draw uncertainty ellipsoid
    #         self.circle = array([[np.sin(x), np.cos(x)] for x in np.linspace(0, 2 * np.pi, 100)]).T
    #
    #     def __call__(self, f, *args):
    #         sol = f.input(NLP_X_OPT)
    #
    #         X_opt = horzcat([sol[i] for i in V.i_X])
    #         Z_opt = horzcat([sol[i] for i in V.i_Z])
    #         U_opt = horzcat([sol[i] for i in V.i_U])
    #
    #         if hasattr(self, "xlines"):
    #             self.xlines[0].set_xdata(X_opt[states.i_x, :].T)
    #             self.xlines[0].set_ydata(X_opt[states.i_y, :].T)
    #             self.xlines[1].set_xdata(X_opt[states.i_x, 0].T)
    #             self.xlines[1].set_ydata(X_opt[states.i_y, 0].T)
    #
    #             for k, v in enumerate(fabs(f.input(NLP_LAMBDA_G)[veccat(g.i_hyper)]).data()):
    #                 j = k % len(hyper)
    #                 i = (k - j) / len(hyper)
    #                 if v > 1e-6:
    #                     self.glines[i][j].set_xdata(DMatrix([hyper[j]["x0"], X_opt[states.i_x, i]]))
    #                     self.glines[i][j].set_ydata(DMatrix([hyper[j]["y0"], X_opt[states.i_y, i]]))
    #                 else:
    #                     self.glines[i][j].set_xdata(DMatrix.nan(2))
    #                     self.glines[i][j].set_ydata(DMatrix.nan(2))
    #             self.zlines[0].set_xdata(Z_opt[states.i_x, :].T)
    #             self.zlines[0].set_ydata(Z_opt[states.i_y, :].T)
    #             self.zlines[1].set_xdata(Z_opt[states.i_x, 0].T)
    #             self.zlines[1].set_ydata(Z_opt[states.i_y, 0].T)
    #             for k in range(N):
    #                 Pi.input(1).set(sol[V.i_Z[k]])
    #                 s_ = numSample1D(Pi, DMatrix(linspace(0.0, 1, 20)).T)
    #                 self.smoothlines[k].set_xdata(s_[states.i_x, :].T)
    #                 self.smoothlines[k].set_ydata(s_[states.i_y, :].T)
    #             self.ulines[0].set_xdata(U_opt[0, IMatrix(range(N) + [0])].T)
    #             self.ulines[0].set_ydata(U_opt[1, IMatrix(range(N) + [0])].T)
    #             self.xlines2[0].set_xdata(X_opt[states.i_x, :].T)
    #             self.xlines2[0].set_ydata(X_opt[states.i_y, :].T)
    #             for k in range(N):
    #                 self.strings[k].set_xdata(horzcat([X_opt[states.i_x, k], U_opt[0, k]]))
    #                 self.strings[k].set_ydata(horzcat([X_opt[states.i_y, k], U_opt[1, k]]))
    #             if with_cov:
    #                 for k in range(N):
    #                     Pxy = sol[V.i_P[k]][states.iv_x + states.iv_y, states.iv_x + states.iv_y]
    #                     w, v = numpy.linalg.eig(Pxy)
    #                     W = mul(v, diag(sqrt(w)))
    #                     e = mul(W, self.circle)
    #                     self.plines[k].set_xdata(e[0, :].T + X_opt[states.i_x, k])
    #                     self.plines[k].set_ydata(e[1, :].T + X_opt[states.i_y, k])
    #         else:
    #             figure(1)
    #             subplot(111)
    #             if plot_initial:
    #                 # Draw the initial guess
    #                 for k in range(N):
    #                     Pi.input(1).set(sol[V.i_Z[k]])
    #                     s_ = numSample1D(Pi, DMatrix(linspace(0.0, 1, 20)).T)
    #                     plot(s_[states.i_x, :].T, s_[states.i_y, :].T, "g")[0]
    #             self.xlines = []
    #             self.xlines.append(plot(X_opt[states.i_x, :].T, X_opt[states.i_y, :].T, "ko")[0])
    #             self.xlines.append(plot(X_opt[states.i_x, 0].T, X_opt[states.i_y, 0].T, "ko", markersize=10)[0])
    #             self.zlines = []
    #             self.zlines.append(plot(Z_opt[states.i_x, :].T, Z_opt[states.i_y, :].T, "b.")[0])
    #             self.zlines.append(plot(Z_opt[states.i_x, 0], Z_opt[states.i_y, 0], "bo")[0])
    #             self.smoothlines = []
    #             for k in range(N):
    #                 Pi.input(1).set(sol[V.i_Z[k]])
    #                 s_ = numSample1D(Pi, DMatrix(linspace(0.0, 1, 20)).T)
    #                 self.smoothlines.append(plot(s_[states.i_x, :].T, s_[states.i_y, :].T, "g")[0])
    #             if with_cov:
    #                 self.plines = []
    #                 for k in range(N):
    #                     Pxy = sol[V.i_P[k]][states.iv_x + states.iv_y, states.iv_x + states.iv_y]
    #                     w, v = numpy.linalg.eig(Pxy)
    #                     W = mul(v, diag(sqrt(w)))
    #                     e = mul(W, self.circle)
    #                     self.plines.append(
    #                         plot(e[0, :].T + X_opt[states.i_x, k], e[1, :].T + X_opt[states.i_y, k], "k")[0]
    #                     )
    #             self.glines = []
    #             for i in range(N):
    #                 self.glines.append([])
    #                 for j in range(len(hyper)):
    #                     self.glines[i].append(plot(DMatrix.nan(2), DMatrix.nan(2), "k")[0])
    #             figure(2)
    #             subplot(111)
    #             self.ulines = []
    #             self.ulines.append(
    #                 plot(U_opt[0, IMatrix(range(N) + [0])].T, U_opt[1, IMatrix(range(N) + [0])].T, "bo-")[0]
    #             )
    #             self.xlines2 = []
    #             self.xlines2.append(plot(X_opt[states.i_x, :].T, X_opt[states.i_y, :].T, "ko")[0])
    #             self.strings = []
    #             for k in range(N):
    #                 self.strings.append(
    #                     plot(
    #                         horzcat([X_opt[states.i_x, k], U_opt[0, k]]),
    #                         horzcat([X_opt[states.i_y, k], U_opt[1, k]]),
    #                         "k",
    #                     )[0]
    #                 )
    #
    #         figure(1)
    #         subplot(111).relim()
    #         subplot(111).autoscale_view()
    #         if log_results is not None:
    #             gcf().savefig("results/s_x_%03d.eps" % (log_results, self.iter), format="eps")
    #         figure(2)
    #         subplot(111).relim()
    #         subplot(111).autoscale_view()
    #         if log_results is not None:
    #             gcf().savefig("results/%s_u_%03d.eps" % (log_results, self.iter), format="eps")
    #         draw()
    #
    #         # reportBounds(solver.output(NLP_G),solver.input(NLP_LBG),solver.output(NLP_UBG),g.getLabels())
    #         # raw_input()
    #
    #         # print f.input(NLP_LAMBDA_G)[veccat(g.i_hyper)]
    #         # print fabs(f.input(NLP_LAMBDA_G)[veccat(g.i_hyper)])>1e-8
    #
    #         self.iter += 1

    # iterationInspector = NLPSolutionInspector()
    # #! We wrap the logging instance in a PyFunction
    # c = PyFunction(
    #     iterationInspector,
    #     nlpsolverOut(
    #         x_opt=sp_dense(V.shape),
    #         cost=sp_dense(1, 1),
    #         lambda_x=sp_dense(V.shape),
    #         lambda_g=sp_dense(g.shape),
    #         g=sp_dense(g.shape),
    #     ),
    #     [sp_dense(1, 1)],
    # )
    # c.init()

    nlp = {"x": V["V"], "f": ff, "g": gf}
    options = {"ipopt.tol": 1e-10,
               "ipopt.linear_solver": "ma57",
               "ipopt.mu_init": 1e-11,
               "ipopt.warm_start_bound_push": 1e-11,
               "ipopt.warm_start_bound_frac": 1e-11,
               "ipopt.warm_start_init_point": "yes",
               }
    solver = cas.nlpsol("solver", "ipopt", nlp, options)
    # solver.setOption("expand_f", True)
    # solver.setOption("expand_g", True)
    # solver.setOption("generate_hessian", True)
    # if with_plotting:
    #     solver.setOption("iteration_callback", c)

    # print("Init solver start")
    # solver.init()
    # print("Init solver end")

    # figure(3)
    # matshow(DMatrix(solver.getH().output().sparsity(), 1))
    # show()

    bounds_init = {
        "lbx": np.ones(V["V"].shape) * -np.inf,
        "ubx": np.ones(V["V"].shape) * np.inf,
        "lbg": np.zeros(gf.shape),
        "ubg": np.zeros(gf.shape),
        "x0": np.zeros(V["V"].shape),
    }
    lam_x0 = np.ones(V["V"].shape)
    lam_g0 = np.ones(gf.shape)

    # Time must be positive
    bounds_init["lbx"][model.i_t] = 0

    if ubound > 0:
        bounds_init["lbx"][model.i_u] = -ubound
        bounds_init["ubx"][model.i_u] = ubound

    bounds_init["ubg"][model.i_g_hyper] = cas.inf
    # if with_cov:
    #     sww = diag([1] * p)
    #     try:
    #         makeDense(sww)
    #     except:
    #         pass
    #     solver.input(NLP_LBX)[V.i_PP.Sigma_ww] = sww
    #     solver.input(NLP_UBX)[V.i_PP.Sigma_ww] = sww
    # if with_gamma:
    #     solver.input(NLP_LBX)[V.i_PP.gamma] = gamma_
    #     solver.input(NLP_X_INIT)[V.i_PP.gamma] = gamma_
    #     solver.input(NLP_UBX)[V.i_PP.gamma] = gamma_

    if last_sol_for_init is None:  # Initalize from scratch
        T_ = 4.0
        bounds_init["x0"][model.i_t] = T_
        # solver.input(NLP_X_INIT)[V.i_PP.m] = model.parameters_

        # Initialize in circle
        for k in range(N):
            for j in range(d + 1):
                t = T_ * ((k + 0.0) / N + (rho_root[j] + 0.0) / N)
                bounds_init["x0"][model.i_z[k][4*j]] = 3 * cas.sin(2 * pi * t / T_)
                bounds_init["x0"][model.i_z[k][4*j+1]] = 3 * cas.cos(2 * pi * t / T_)
                # solver.input(NLP_X_INIT)[V.i_Z[k][states.iv_x, j]] = 3 * sin(2 * pi * t / T_)
                # solver.input(NLP_X_INIT)[V.i_Z[k][states.iv_y, j]] = 3 * cos(2 * pi * t / T_)

        for k in range(N + 1):
            t = T_ * (k + 0.0) / N
            bounds_init["x0"][model.i_x[2*k]] = 3 * cas.sin(2 * pi * t / T_)
            bounds_init["x0"][model.i_x[2*k]+1] = 3 * cas.cos(2 * pi * t / T_)
            # solver.input(NLP_X_INIT)[V.i_X[k][states.iv_x]] = 3 * sin(2 * pi * t / T_)
            # solver.input(NLP_X_INIT)[V.i_X[k][states.iv_y]] = 3 * cos(2 * pi * t / T_)

    else:  # Initialize from another object
        bounds_init["x0"][model.i_t] = last_sol_for_init["x"][last_sol_for_init["i_t"]]
        lam_x0[model.i_t] = last_sol_for_init["lam_x"][last_sol_for_init["i_t"]]
        # solver.input(NLP_X_INIT)[V.i_PP.m] = last_sol_for_init(NLP_X_OPT)[init.V.i_PP.m]
        # solver.output(NLP_LAMBDA_X)[V.i_PP.m] = last_sol_for_init(NLP_LAMBDA_X)[init.V.i_PP.m]

        for k in range(N + 1):
            bounds_init["x0"][model.i_x[2*k]] = last_sol_for_init["x"][last_sol_for_init["i_x"][2*k]]
            bounds_init["x0"][model.i_x[2*k]+1] = last_sol_for_init["x"][last_sol_for_init["i_x"][2*k]+1]
            lam_x0[model.i_x[2*k]] = last_sol_for_init["lam_x"][last_sol_for_init["i_x"][2*k]]
            lam_x0[model.i_x[2*k]+1] = last_sol_for_init["lam_x"][last_sol_for_init["i_x"][2*k]+1]
        for k in range(N):
            bounds_init["x0"][model.i_u[2*k]] = last_sol_for_init["x"][last_sol_for_init["i_u"][2*k]]
            bounds_init["x0"][model.i_u[2*k+1]] = last_sol_for_init["x"][last_sol_for_init["i_u"][2*k+1]]
            lam_x0[model.i_u[2*k]] = last_sol_for_init["lam_x"][last_sol_for_init["i_u"][2*k]]
            lam_x0[model.i_u[2*k+1]] = last_sol_for_init["lam_x"][last_sol_for_init["i_u"][2*k+1]]
            bounds_init["x0"][model.i_z[k]] = last_sol_for_init["x"][last_sol_for_init["i_z"][k]]
            lam_x0[model.i_z[k]] = last_sol_for_init["lam_x"][last_sol_for_init["i_z"][k]]

        # If the initial guess contains covariance, use it
        if with_cov and len(last_sol_for_init["i_p"][0]) != 0:
            for k in range(N + 1):
                bounds_init["x0"][model.i_p[k]] = last_sol_for_init["x"][last_sol_for_init["i_p"][k]]
                lam_x0[model.i_p[k]] = last_sol_for_init["lam_x"][last_sol_for_init["i_p"][k]]
            # for k in range(N):  # BUG ?
            #     bounds_init["x0"][model.i_m[k]] = last_sol_for_init["x"][model.i_m[k]]
            #     bounds_init["lam_x0"][model.i_m[k]] = last_sol_for_init["lam_x"][model.i_m[k]]
            #     solver.input(NLP_X_INIT)[V.i_P[k]] = last_sol_for_init(NLP_X_OPT)[init.V.i_P[k]]
            #     solver.output(NLP_LAMBDA_X)[V.i_P[k]] = last_sol_for_init(NLP_LAMBDA_X)[init.V.i_P[k]]

        lam_g0[model.i_g_f] = last_sol_for_init["lam_g"][last_sol_for_init["i_g_f"]]
        lam_g0[model.i_g_g] = last_sol_for_init["lam_g"][last_sol_for_init["i_g_g"]]
        lam_g0[model.i_g_hyper] = last_sol_for_init["lam_g"][last_sol_for_init["i_g_hyper"]]
        lam_g0[model.i_g_periodic] = last_sol_for_init["lam_g"][last_sol_for_init["i_g_periodic"]]
        lam_g0[model.i_g_fix] = last_sol_for_init["lam_g"][last_sol_for_init["i_g_fix"]]

        # for k in range(N):
        #     solver.input(NLP_LAMBDA_INIT)[g.i_F[k]] = last_sol_for_init(NLP_LAMBDA_G)[init.g.i_F[k]]
        #     solver.input(NLP_LAMBDA_INIT)[g.i_G[k]] = last_sol_for_init(NLP_LAMBDA_G)[init.g.i_G[k]]
        #     solver.input(NLP_LAMBDA_INIT)[g.i_hyper[k]] = last_sol_for_init(NLP_LAMBDA_G)[init.g.i_hyper[k]]
        # solver.input(NLP_LAMBDA_INIT)[g.i_fix] = last_sol_for_init(NLP_LAMBDA_G)[init.g.i_fix]
        # solver.input(NLP_LAMBDA_INIT)[g.i_periodic] = last_sol_for_init(NLP_LAMBDA_G)[init.g.i_periodic]

        bounds_init["lam_x0"] = lam_x0
        bounds_init["lam_g0"] = lam_g0

    # solver.solve()
    sol = solver.call(bounds_init)
    f_opt = sol["f"]
    x_opt = sol["x"]
    g_sol = sol["g"]
    lam_x_opt = sol["lam_x"]
    lam_g_opt = sol["lam_g"]
    sol["i_x"] = model.i_x
    sol["i_z"] = model.i_z
    sol["i_p"] = model.i_p
    sol["i_m"] = model.i_m
    sol["i_u"] = model.i_u
    sol["i_t"] = model.i_t
    sol["i_g_f"] = model.i_g_f
    sol["i_g_g"] = model.i_g_g
    sol["i_g_hyper"] = model.i_g_hyper
    sol["i_g_periodic"] = model.i_g_periodic
    sol["i_g_fix"] = model.i_g_fix
    sol["i_g_periodic_p"] = model.i_g_periodic_p
    sol["i_g_p"] = model.i_g_p
    sol["i_g_m"] = model.i_g_m

    if log_results is not None:
        def draw_superellipse(ax, hyper, resolution=100):
            for i in range(2):
                a = hyper[i]["a"]
                b = hyper[i]["b"]
                x0 = hyper[i]["x0"]
                y0 = hyper[i]["y0"]
                n = hyper[i]["n"]

                x = np.linspace(-2 * a + x0, 2 * a + x0, resolution)
                y = np.linspace(-2 * b + y0, 2 * b + y0, resolution)

                X, Y = np.meshgrid(x, y)
                Z = ((X - x0) / a) ** n + ((Y - y0) / b) ** n - 1

                ax.contourf(X, Y, Z, levels=[-1000, 0], colors=["#DA1984"], alpha=0.5)
                # ax.contour(X, Y, Z, levels=[0], colors='black')

        # figure(1).savefig("results/%s_x.eps" % log_results, format="eps")
        # figure(2).savefig("results/%s_u.eps" % log_results, format="eps")

        positions = np.zeros((2, N+1))
        positions[0, :] = np.reshape(x_opt[model.i_x][::4], (N+1, ))
        positions[1, :] = np.reshape(x_opt[model.i_x][1::4], (N+1, ))
        fig, ax = plt.subplots(1, 1)
        draw_superellipse(ax, hyper)
        ax.plot(positions[0, :], positions[1, :], '.b')
        fig.savefig(f"{log_results}.png")
        fig.show()

    # if with_plotting and plot_hold:
    #     matplotlib.interactive(False)
    #     show()

    # print(solver.output(NLP_LAMBDA_G)[veccat(g.i_hyper)])

    # # Make public
    # self.solver = solver
    # self.V = V
    # self.param = param
    # self.g = g
    # self.with_cov = with_cov
    # self.states = states
    # self.h = h
    # if with_gamma:
    #     self.hr = hr

    return sol


if __name__ == "__main__":
    Collocation(N=40, with_cov=False, with_plotting=True)
