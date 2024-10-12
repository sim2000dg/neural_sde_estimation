import numpy as np
import tensorflow as tf
from sklearn.model_selection import ParameterGrid
from typing import List, Dict, Tuple

from tensorflow.python.framework.errors_impl import InvalidArgumentError

from ..simulation_sde import milstein_sim, euler_sim, SDECoefficient
from tqdm import tqdm
import random
import tensorflow.keras as keras
import platform
import pandas as pd

tf.config.set_visible_devices([], "GPU")


def grid_test(
    coefficient: SDECoefficient,
    mc_iterations: int,
    depth_grid: List[int],
    hidden_dim_grid: List[int],
    scale_noise_grid: List[float],
    skip_grid: List[float],
    delta_sim: float,
    time_horizon_grid: List[int],
    epochs: int,
    init: np.ndarray,
    compact_set: np.ndarray,
    scale_shift_process: np.ndarray,
    seed: int,
    milstein: bool,
) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Main method for testing over the chosen grid the generalization of the neural network estimator for
    the drift coefficient proposed by Koike and Oga (2024)
    :param coefficient: An instance of the coefficient set characterizing the SDE.
    :param mc_iterations: Number of Monte Carlo iterations
     used for approximating the expectation in the risk formulation.
    :param depth_grid: The depths of the MLP to consider.
    :param hidden_dim_grid: The dimensionalities of the hidden layers of the MLP to consider.
    :param scale_noise_grid: The values of the noise scaling to test.
    :param skip_grid: A list of integers that determine the number of points of the simulation skipped
     between any two points used for the fitting.
    :param delta_sim: The value of the time discretization for the simulation of the sample path.
    :param time_horizon_grid: The values of the maximum time to consider when testing.
    :param epochs: The epochs for training.
    :param init: Initial value of the SDE.
    :param compact_set: The compact set we are estimating the drift in, the cartesian product of the closed intervals
     given by the endpoints in each row of the array.
    :param scale_shift_process: Scale and shift for the whole process, for each component.
     First column is scale, second is shift.
    :param seed: Seed for the random number generator.
    :param milstein: Whether to use Milstein or Euler-Maruyama.
    :return: A tuple with a NumPy array with the simulation results in terms of mean square error w.r.t.
     the test diffusion for each MC iteration (columns) for each combination of hyperparameters (rows).
     The other element of the tuple is a Pandas dataframe with the ordered hyperparameters combinations.
    """

    # This is the generator we use *everywhere*, this makes the generated processes deterministic
    generator = np.random.default_rng(seed)
    # The two things below make Tensorflow data shuffle and weight initialization ops deterministic
    random.seed(seed)  # Keras weight init uses the standard random module...
    tf.random.set_seed(seed)
    dict_eval = {
        "depth": depth_grid,
        "hidden_dim": hidden_dim_grid,
        "skip": skip_grid,
        "time_horizon": time_horizon_grid,
        "scale_noise": scale_noise_grid,
    }

    parameter_grid = ParameterGrid(
        dict_eval
    )  # Utility function from scikit to get list of dict of combinations
    parameter_grid = list(parameter_grid)

    grid_results = np.zeros(
        shape=(len(parameter_grid), mc_iterations), dtype=np.float64
    )  # Allocate array for results, rows for parameter combinations, columns for Monte Carlo iterates

    train_grid_results = np.zeros(
        shape=(len(parameter_grid), mc_iterations), dtype=np.float64
    )  # Allocate array for results, rows for parameter combinations, columns for Monte Carlo iterates

    for i, parameters in (
        bar := tqdm(enumerate(parameter_grid), total=len(parameter_grid))
    ):
        bar.set_description(f"Grid iteration {i + 1} of {len(parameter_grid)}")
        mse_vector, mse_train_vector = monte_carlo_evaluation(  # Call underlying Monte Carlo routine
            **parameters,
            coefficient=coefficient,
            epochs=epochs,
            mc_iterations=mc_iterations,
            milstein=milstein,
            init=init,
            compact_set=compact_set,
            scale_shift_process=scale_shift_process,
            generator=generator,
            tqdm_bar=bar,
            delta_sim=delta_sim,
        )
        grid_results[i] = mse_vector  # Save result as specific row
        train_grid_results[i] = mse_train_vector  # Save train result as specific row

    return grid_results, pd.DataFrame.from_dict(parameter_grid), train_grid_results


def monte_carlo_evaluation(
    coefficient: SDECoefficient,
    mc_iterations: int,
    depth: int,
    hidden_dim: int,
    epochs: int,
    skip: float,
    delta_sim: float,
    time_horizon: int,
    scale_noise: float,
    init: np.ndarray,
    compact_set: np.ndarray,
    scale_shift_process: np.ndarray,
    milstein: bool,
    generator: np.random.Generator,
    tqdm_bar: tqdm,
) -> np.ndarray:
    """
    Function for Monte Carlo estimation of the generalization risk for the neural estimator of the drift coefficient,
    given experiment, training and model hyperparameters.

    :param coefficient: An instance of the coefficient set characterizing the SDE.
    :param mc_iterations: Number of Monte Carlo iterations
     used for approximating the expectation in the risk formulation.
    :param depth: The depths of the MLP to consider.
    :param hidden_dim: The dimensionalities of the hidden layers of the MLP to consider.
    :param scale_noise: The values of the noise scaling to test.
    :param skip: skip-1 is the number of points of the simulation skipped between any two points used for the fitting.
    :param delta_sim: The value of the time discretization for the simulation of the sample path.
    :param time_horizon: The values of the maximum time to consider when testing.
    :param epochs: The epochs for training.
    :param init: Initial value of the SDE.
    :param compact_set: The compact set we are estimating the drift in, the cartesian product of the closed intervals
     given by the endpoints in each row of the array.
    :param scale_shift_process: Scale and shift for the whole process, for each component.
     First column is scale, second is shift.
    :param milstein: Whether to use Milstein or Euler-Maruyama.
    :param generator: NumPy generator for PRNG.
    :param tqdm_bar: Bar object for tqdm; this is needed to update the progress bar with MC iterations.
    :return: A vector of mean squared errors, one for each Monte Carlo iteration.
    """

    result_vector = np.zeros(
        mc_iterations, dtype=np.float64
    )  # Allocate vector for MSEs

    result_vector_train = np.zeros(
        mc_iterations, dtype=np.float64
    )  # Allocate vector for train MSEs

    for i in range(mc_iterations):  # Iterate over number of Monte Carlo iterations
        tqdm_bar.set_postfix({"MC iteration": i + 1})
        # Get approximation of the sample path for the process and its copy
        while True:
            if milstein:
                process = milstein_sim(
                    coefficient,
                    init,
                    time_horizon,
                    delta_sim,
                    generator,
                    scale_noise,
                    scale_shift_process,
                )
                test_process = milstein_sim(
                    coefficient,
                    init,
                    time_horizon,
                    delta_sim,
                    generator,
                    scale_noise,
                    scale_shift_process,
                )
            else:
                process = euler_sim(
                    coefficient,
                    init,
                    time_horizon,
                    delta_sim,
                    generator,
                    scale_noise,
                    scale_shift_process,
                )
                test_process = euler_sim(
                    coefficient,
                    init,
                    time_horizon,
                    delta_sim,
                    generator,
                    scale_noise,
                    scale_shift_process,
                )

            difference_quotients = np.diff(process[:, ::skip]) / (
                delta_sim * skip
            )  # Compute difference quotients which we use to train the model, use range syntax to skip observations
            process = process[:, ::skip][
                :, :-1
            ]  # We cannot compute any difference quotient at the boundary of the time interval

            # Filter the observations w.r.t. the compact set we are considering
            mask_compact_train = (
                np.all((process <= compact_set[:, 1][:, np.newaxis]), axis=0)
            ) & (np.all(process >= compact_set[:, 0][:, np.newaxis], axis=0))
            process = process[:, mask_compact_train]
            difference_quotients = difference_quotients[:, mask_compact_train]

            if np.sum(mask_compact_train) <= 2:
                continue

            trained = model_fit_routine(
                process, difference_quotients, depth, hidden_dim, 64, epochs
            )  # Call the underlying model fit routine to initialize and train the model

            test_process = test_process[
                :, ::skip
            ]  # Skip observations for the test process
            # Disregard observations outside the compact set we are considering
            mask_compact_test = (
                np.all(test_process <= compact_set[:, 1][:, np.newaxis], axis=0)
            ) & (np.all(test_process >= compact_set[:, 0][:, np.newaxis], axis=0))
            test_process = test_process[
                :,
                mask_compact_test,
            ]
            if np.sum(mask_compact_test) <= 2:
                continue

            # Get the value of the drift at the point of the sample path of the independent copy of the process
            drift_test = (
                coefficient.drift(
                    test_process / scale_shift_process[:, 0][:, np.newaxis]
                    - scale_shift_process[:, 1][:, np.newaxis]
                )
                * scale_shift_process[:, 0][:, np.newaxis]
            )

            drift_train = (  # Get actual drift for training trajectory
                coefficient.drift(
                    process / scale_shift_process[:, 0][:, np.newaxis]
                    - scale_shift_process[:, 1][:, np.newaxis]
                )
                * scale_shift_process[:, 0][:, np.newaxis]
            )

            mse = trained.evaluate(  # Compute test MSE
                test_process.transpose(), drift_test.transpose(), verbose=0
            )[0]
            result_vector[i] = mse  # Save the value
            mse_train = trained.evaluate(  # Compute train MSE
                process.transpose(), drift_train.transpose(), verbose=0
            )[0]
            result_vector_train[i] = mse_train  # Save train MSE
            break

    return result_vector, result_vector_train


def model_fit_routine(
    inputs: np.ndarray,
    outputs: np.ndarray,
    depth: int,
    hidden_dim: int,
    batch_size: int,
    epochs: int,
) -> keras.Model:
    """
    Function to fit a MLP to the provided data, with specified training and model hyperparameters.
    :param inputs: Input array.
    :param outputs: Output array.
    :param depth: Depth of the model.
    :param hidden_dim: Hidden dimensionality of the MLP.
    :param batch_size: Batch size for training.
    :param epochs: Number of epochs for training.
    :return: The trained model.
    """
    dimension = inputs.shape[0]  # Dimensionality of the SDE
    model = keras.Sequential()  # Usual and straightforward Keras Sequential API
    model.add(
        keras.layers.InputLayer((dimension,))
    )  # The input is a vector with SDE dimensionality
    for layer in range(depth):
        model.add(
            keras.layers.Dense(
                hidden_dim,
                activation="relu",
                kernel_initializer="he_normal",  # Best weight initialization for ReLU activation
                bias_initializer=keras.initializers.Constant(
                    0.01
                ),  # Best practice to avoid excess of dead units
                # The constraints below are weights rescaling that are a "nice"
                # relaxation of the weights constrained imposed in Koike and Oga (2024)
                kernel_constraint=keras.constraints.MaxNorm(
                    max_value=np.sqrt(hidden_dim) if layer != 0 else np.sqrt(dimension)
                ),
                bias_constraint=keras.constraints.MaxNorm(
                    max_value=np.sqrt(hidden_dim)
                ),
            )
        )
    # Output layer
    model.add(
        keras.layers.Dense(
            dimension,
            kernel_initializer="he_normal",
            kernel_constraint=keras.constraints.MaxNorm(max_value=np.sqrt(hidden_dim)),
            bias_constraint=keras.constraints.MaxNorm(max_value=np.sqrt(hidden_dim)),
        )
    )

    model.compile(  # Specify what is needed for training
        loss=keras.losses.MeanSquaredError(),
        optimizer=(
            tf.keras.optimizers.legacy.Adam()
            if "macOS" in platform.platform()
            else tf.keras.optimizers.Adam()
        ),
        run_eagerly=False,
        metrics=["mse"],
    )
    model.fit(  # Model fit
        inputs.transpose(),
        outputs.transpose(),
        epochs=epochs,
        verbose=0,
        batch_size=batch_size,
    )

    return model
