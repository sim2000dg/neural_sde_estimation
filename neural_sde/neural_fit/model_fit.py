import numpy as np
import tensorflow as tf
from sklearn.model_selection import ParameterGrid
from typing import List
from ..simulation_sde import milstein_sim, euler_sim, SDECoefficient
from tqdm import tqdm
import keras
import random


def grid_test(
    coefficient: SDECoefficient,
    mc_iterations: int,
    depth_grid: List[int],
    hidden_dim_grid: List[int],
    scale_noise_grid: List[int],
    delta_grid: List[int],
    time_horizon_grid: List[int],
    epochs: int,
    init: np.ndarray,
    seed: int,
    milstein: bool,
) -> np.ndarray:
    """
    Main method for testing over the chosen grid the generalization of the neural network estimator for
    the drift coefficient proposed by Koike and Oga (2024)
    :param coefficient: An instance of the coefficient set characterizing the SDE.
    :param mc_iterations: Number of Monte Carlo iterations
     used for approximating the expectation in the risk formulation.
    :param depth_grid: The depths of the MLP to consider.
    :param hidden_dim_grid: The dimensionalities of the hidden layers of the MLP to consider.
    :param scale_noise_grid: The values of the noise scaling to test.
    :param delta_grid: The values of the time discretization to test.
    :param time_horizon_grid: The values of the maximum time to consider when testing.
    :param epochs: The epochs for training.
    :param init: Initial value of the SDE.
    :param seed: Seed for the random number generator.
    :param milstein: Whether to use Milstein or Euler-Maruyama.
    :return: A NumPy array with the simulation results in terms of mean square error w.r.t. the test diffusion
     for each MC iteration (columns) for each combination of hyperparameters (rows).
    """
    # This is the generator we use *everywhere*, this makes the generated processes deterministic
    generator = np.random.default_rng(seed)
    # The two things below make Tensorflow data shuffle and weight initialization ops deterministic
    random.seed(seed)  # Keras weight init uses the standard random module...
    tf.random.set_seed(seed)
    dict_eval = {
        "depth": depth_grid,
        "hidden_dim": hidden_dim_grid,
        "delta": delta_grid,
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
    for i, parameters in (
        bar := tqdm(enumerate(parameter_grid), total=len(parameter_grid))
    ):
        bar.set_description(f"Grid iteration {i + 1} of {len(parameter_grid)}")
        mse_vector = monte_carlo_evaluation(  # Call underlying Monte Carlo routine
            **parameters,
            coefficient=coefficient,
            epochs=epochs,
            mc_iterations=mc_iterations,
            milstein=milstein,
            init=init,
            generator=generator,
            tqdm_bar=bar,
        )
        grid_results[i] = mse_vector  # Save result as specific row

    return grid_results


def monte_carlo_evaluation(
    coefficient: SDECoefficient,
    mc_iterations: int,
    depth: int,
    hidden_dim: int,
    epochs: int,
    delta: float,
    time_horizon: int,
    scale_noise: float,
    init: np.ndarray,
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
    :param delta: The values of the time discretization to test.
    :param time_horizon: The values of the maximum time to consider when testing.
    :param epochs: The epochs for training.
    :param init: Initial value of the SDE.
    :param milstein: Whether to use Milstein or Euler-Maruyama.
    :param generator: NumPy generator for PRNG.
    :param tqdm_bar: Bar object for tqdm; this is needed to update the progress bar with MC iterations.
    :return: A vector of mean squared errors, one for each Monte Carlo iteration.
    """

    result_vector = np.zeros(
        mc_iterations, dtype=np.float64
    )  # Allocate vector for MSEs
    for i in range(mc_iterations):  # Iterate over number of Monte Carlo iterations
        tqdm_bar.set_postfix({"MC iteration": i + 1})
        # Get approximation of the sample path for the process and its copy
        if milstein:
            process = milstein_sim(
                coefficient, init, time_horizon, delta, generator, scale_noise
            )
            test_process = milstein_sim(
                coefficient, init, time_horizon, delta, generator, scale_noise
            )
        else:
            process = euler_sim(
                coefficient, init, time_horizon, delta, generator, scale_noise
            )
            test_process = euler_sim(
                coefficient, init, time_horizon, delta, generator, scale_noise
            )

        difference_quotients = (
            np.diff(process) / delta
        )  # Compute difference quotients which we use to train the model
        process = process[
            :, :-1
        ]  # We cannot compute any difference quotient at the boundary of the time interval
        trained = model_fit_routine(
            process, difference_quotients, depth, hidden_dim, 64, epochs
        )  # Call the underlying model fit routine to initialize and train the model
        # Get the value of the drift at the point of the sample path of the independent copy of the process
        drift_test = coefficient.drift(test_process)
        # Compute test MSE
        mse = trained.evaluate(
            test_process.transpose(), drift_test.transpose(), verbose=0
        )[0]
        result_vector[i] = mse  # Save the value

    return result_vector


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
        keras.layers.InputLayer(shape=(dimension,))
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
    # Output layer, no activation should be put here!
    model.add(
        keras.layers.Dense(
            dimension,
            kernel_initializer="he_normal",
            kernel_constraint=keras.constraints.MaxNorm(max_value=np.sqrt(hidden_dim)),
            bias_constraint=keras.constraints.MaxNorm(max_value=np.sqrt(hidden_dim)),
        )
    )
    optimizer = (
        keras.optimizers.Adam()
    )  # Default Adam, nothing more than this should be needed for this task

    model.compile(  # Specify what is needed for training
        loss=keras.losses.MeanSquaredError(),
        optimizer=optimizer,
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
