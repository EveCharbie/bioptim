from collocation import *
import os

hyper = [{"a": 1, "b": 1, "x0": 0, "y0": 0, "n": 2}]

# nlp_solver = "worhp"
initial = Collocation(ubound=10, with_cov=False, plot_hold=False, hyper=hyper, plot_clear=True, log_results="nominal")

cov_propagation = Collocation(ubound=10, with_cov=True, plot_hold=False, last_sol_for_init=initial, hyper=hyper, plot_clear=True, log_results="cov_propagation")

robustify = Collocation(
    ubound=10,
    with_cov=True,
    plot_hold=False,
    last_sol_for_init=cov_propagation,
    with_gamma=True,
    gamma=1,
    hyper=hyper,
    plot_clear=True,
    plot_initial=True,
    log_results="robustify",
)

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
