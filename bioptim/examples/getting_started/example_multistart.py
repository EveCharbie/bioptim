"""
An example of how to use multi-start to find local minima from different initial guesses.
This example is a variation of the pendulum example in getting_started/pendulum.py.
"""
import pickle
import os


from bioptim import (
    BiorbdModel,
    OptimalControlProgram,
    DynamicsFcn,
    Dynamics,
    Bounds,
    InitialGuess,
    ObjectiveFcn,
    Objective,
    CostType,
    Solver,
    InterpolationType,
    MultiStart,
    Solution,
    MagnitudeType,
)

def prepare_ocp(
    bio_model_path: str,
    final_time: float,
    n_shooting: int,
    seed: int = 0,
) -> OptimalControlProgram:
    """
    The initialization of an ocp

    Parameters
    ----------
    bio_model_path: str
        The path to the biorbd model
    final_time: float
        The time in second required to perform the task
    n_shooting: int
        The number of shooting points to define int the direct multiple shooting program
    seed: int
        The seed to use for the random initial guess

    Returns
    -------
    The OptimalControlProgram ready to be solved
    """

    bio_model = BiorbdModel(bio_model_path)

    # Add objective functions
    objective_functions = Objective(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau")

    # Dynamics
    dynamics = Dynamics(DynamicsFcn.TORQUE_DRIVEN)

    # Path constraint
    x_bounds = bio_model.bounds_from_ranges(["q", "qdot"])
    x_bounds[:, [0, -1]] = 0
    x_bounds[1, -1] = 3.14

    # Initial guess
    n_q = bio_model.nb_q
    n_qdot = bio_model.nb_qdot
    x_init = InitialGuess([0] * (n_q + n_qdot), interpolation=InterpolationType.CONSTANT)
    x_init = x_init.add_noise(
        bounds=x_bounds,
        magnitude=0.5,
        magnitude_type=MagnitudeType.RELATIVE,
        n_shooting=n_shooting + 1,
        seed=seed,
    )

    # Define control path constraint
    n_tau = bio_model.nb_tau
    tau_min, tau_max, tau_init = -100, 100, 0
    u_bounds = Bounds([tau_min] * n_tau, [tau_max] * n_tau)
    u_bounds[1, :] = 0  # Prevent the model from actively rotate

    u_init = InitialGuess([0] * n_tau, interpolation=InterpolationType.CONSTANT)
    u_init = u_init.add_noise(
        bounds=u_bounds,
        magnitude=0.5,
        magnitude_type=MagnitudeType.RELATIVE,
        n_shooting=n_shooting,
        seed=seed,
    )

    ocp = OptimalControlProgram(
        bio_model,
        dynamics,
        n_shooting,
        final_time,
        x_init=x_init,
        u_init=u_init,
        x_bounds=x_bounds,
        u_bounds=u_bounds,
        objective_functions=objective_functions,
        n_threads=1,  # You cannot use multi-threading for the resolution of the ocp with multi-start
    )

    ocp.add_plot_penalty(CostType.ALL)

    return ocp
def save_results(
    sol: Solution,
    biorbd_model_path: str,
    final_time: float,
    n_shooting: int,
    seed: int,
    save_folder: str,
    only_save_filename: bool = False,
):
    """
    Solving the ocp
    Parameters
    ----------
    sol: Solution
        The solution to the ocp at the current pool
    biorbd_model_path: str
        The path to the biorbd model
    final_time: float
        The time in second required to perform the task
    n_shooting: int
        The number of shooting points to define int the direct multiple shooting program
    seed: int
        The seed to use for the random initial guess
    only_save_filename: bool
        True if you want to return only the name of the file without saving, else False
    """
    # OptimalControlProgram.save(sol, f"solutions/pendulum_multi_start_random{seed}.bo", stand_alone=True)
    bio_model = biorbd_model_path.split('/')[-1].removesuffix('.bioMod')
    filename = f"pendulum_multi_start_random_states_{n_shooting}_{seed}.pkl"
    if only_save_filename == True:
        return filename
    states = sol.states["all"]
  #  save_folder = "./temporary_results/"

    with open(f"{save_folder}/{filename}", "wb") as file:
        pickle.dump(states, file)

def should_solve(args, save_folder,save_results=save_results):
    """
    Check if the filename already appears in the folder where files are saved, if not ocp must be solved
    """
    #save_folder = "/home/laseche/Documents/Stage_Lisa/Lisa/Sol"
    already_done_filenames = os.listdir(f"{save_folder}")
    return save_results([None], *args, only_save_filename=True) not in already_done_filenames

def prepare_multi_start(
    combinatorial_parameters: dict[tuple,...],
    n_pools: int = 1,
    save_folder: str = None,

) -> MultiStart:
    """
    The initialization of the multi-start
    """
    return MultiStart(
        combinatorial_parameters=combinatorial_parameters,
        prepare_ocp_callback=prepare_ocp,
        post_optimization_callback=save_results,
        should_solve_callback=lambda *args, **kwargs: should_solve(*args, **kwargs, save_folder=save_folder),
        solver=Solver.IPOPT(show_online_optim=False),  # You cannot use show_online_optim with multi-start
        n_pools=n_pools,
    )


def main():
    # --- Prepare the multi-start and run it --- #

    #Creates a folder to save the solutions
    save_folder = "./temporary_solutions"
    os.mkdir(f"{save_folder}")

#    global save_folder
    #save_folder = "/home/laseche/Documents/Stage_Lisa/Lisa/Sol"
    already_done_filenames = os.listdir(f"{save_folder}")

    bio_model_path = ["models/pendulum.bioMod"]
    final_time = [1]
    n_shooting = [30, 40, 50]
    seed = [0, 1, 2, 3]

    combinatorial_parameters = {'bio_model_path': bio_model_path, 'final_time': final_time, 'n_shooting': n_shooting,
                                'seed': seed}
    multi_start = prepare_multi_start(
        combinatorial_parameters=combinatorial_parameters,
        n_pools=4,
        save_folder=save_folder,
    )

    multi_start.solve()

    #Delete the solutions
    #os.rmdir(f"path/{save_folder}")

if __name__ == "__main__":
    main()
