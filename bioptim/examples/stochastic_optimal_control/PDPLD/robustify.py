from collocation import *
import os

hyper = [{"a": 1, "b": 1, "x0": 0, "y0": 0, "n": 2}]

#
initial = Collocation(nlp_solver="worhp", with_cov=False, plot_hold=False, hyper=hyper, plot_clear=True)

cov_propagation = Collocation(
    nlp_solver="worhp", with_cov=True, plot_hold=False, init=initial, hyper=hyper, plot_clear=True
)

robustify = Collocation(
    nlp_solver="worhp",
    with_cov=True,
    plot_hold=False,
    init=cov_propagation,
    with_gamma=True,
    gamma=1,
    hyper=hyper,
    plot_clear=True,
    plot_initial=True,
)
raw_input()
os.system("reset")

#
initial = Collocation(
    nlp_solver="worhp", with_cov=False, plot_hold=False, hyper=hyper, log_results="nominal", plot_clear=True
)

cov_propagation = Collocation(
    nlp_solver="worhp",
    with_cov=True,
    plot_hold=False,
    init=initial,
    hyper=hyper,
    log_results="cov_propagation",
    plot_clear=True,
)

robustify = Collocation(
    nlp_solver="worhp",
    with_cov=True,
    plot_hold=False,
    init=cov_propagation,
    with_gamma=True,
    gamma=1,
    hyper=hyper,
    log_results="robustify",
    plot_clear=True,
    plot_initial=True,
)

raw_input()
os.system("reset")

result = robustify
param = result.param

solver = result.solver

V = result.V

sol = solver.output()

g = result.g

states = result.states

X = sol[V.i_X[0]]

h = result.h

hdx = h.jac(0, 0)
hdxf = SXFunction([x], [hdx])

P = sol[V.i_P[0]]
