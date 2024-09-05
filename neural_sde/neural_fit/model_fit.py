import numpy as np
import tensorflow as tf
from sklearn.model_selection import ParameterGrid
from typing import List
from ..simulation_sde import milstein_sim, euler_sim, SDECoefficient
from tqdm import tqdm
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
    # The two things below make Tensorflow shuffle and weight initialization ops deterministic
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
    )
    for i, parameters in tqdm(enumerate(parameter_grid), total=len(parameter_grid)):
        print(
            f'Starting iteration {i + 1} of {len(parameter_grid)}. \n Delta : {parameters["delta"]}, '
            f'time_horizon : {parameters["time_horizon"]}, '
            f"scale_noise : {parameters['scale_noise']}"
            f'hidden_dim: {parameters["hidden_dim"]}, '
            f'depth: {parameters["depth"]}'
        )
        mse_vector = monte_carlo_evaluation(
            **parameters,
            coefficient=coefficient,
            epochs=epochs,
            mc_iterations=mc_iterations,
            milstein=milstein,
            init=init,
            generator=generator,
        )
        grid_results[i] = mse_vector

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
) -> float:

    result_vector = np.zeros(mc_iterations, dtype=np.float64)
    for i in range(mc_iterations):
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

        difference_quotients = np.diff(process) / delta
        process = process[:, :-1]
        trained = model_fit_routine(
            process, difference_quotients, depth, hidden_dim, 64, epochs
        )
        drift_test = coefficient.drift(test_process)
        mse = trained.evaluate(test_process.transpose(), drift_test.transpose())[0]
        result_vector[i] = mse

    return result_vector


def model_fit_routine(
    inputs: np.ndarray,
    outputs: np.ndarray,
    depth: int,
    hidden_dim: int,
    batch_size: int,
    epochs: int,
) -> tf.keras.Model:
    """
    Function for final model fitting, without considering validation dynamics.
    :param inputs: Input array.
    :param outputs: Output array.
    :param depth: Depth of the model.
    :param hidden_dim: Hidden dimensionality of the MLP.
    :param batch_size: Batch size for training.
    :param epochs: Number of epochs for training.
    :return: The trained model.
    """
    dimension = inputs.shape[0]
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.InputLayer(shape=(dimension, )))
    for layer in range(depth):
        model.add(
            tf.keras.layers.Dense(
                hidden_dim,
                activation="sigmoid",
                use_bias=True,
                kernel_initializer="he_normal",
                bias_initializer="zeros",
                kernel_constraint=tf.keras.constraints.MaxNorm(
                    max_value=np.sqrt(hidden_dim) if layer != 0 else np.sqrt(dimension)
                ),
                bias_constraint=tf.keras.constraints.MaxNorm(
                    max_value=np.sqrt(hidden_dim)
                ),
            )
        )
    model.add(
        tf.keras.layers.Dense(
            dimension,
            kernel_constraint=tf.keras.constraints.MaxNorm(
                max_value=np.sqrt(hidden_dim)
            ),
            bias_constraint=tf.keras.constraints.MaxNorm(max_value=np.sqrt(hidden_dim)),
        )
    )
    optimizer = tf.keras.optimizers.Adam()

    model.compile(
        loss=tf.keras.losses.MeanSquaredError(),
        optimizer=optimizer,
        run_eagerly=False,
        metrics=["mse"],
    )
    model.fit(
        inputs.transpose(),
        outputs.transpose(),
        epochs=epochs,
        verbose=0,
        batch_size=batch_size,
    )

    return model
