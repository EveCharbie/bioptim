from collocation import *
import os

# nlp_solver = "worhp"
initial = Collocation(ubound=10, with_cov=False, log_results="nominal")

cov_propagation = Collocation(ubound=10, with_cov=True, last_sol_for_init=initial, log_results="cov_propagation")

robustify = Collocation(
    ubound=10,
    with_cov=True,
    last_sol_for_init=cov_propagation,
    with_gamma=True,
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
