import numpy as np


def euler_sim(
    coefficient,
    init: np.ndarray,
    time_horizon: float,
    delta: float,
    generator: np.random.Generator,
    scale_noise: float = 1.0,
):
    """
    Euler-Maruyama SDE solver.

    :param coefficient: An instance of the coefficient set characterizing the SDE:
    :param init: Initial value of the SDE.
    :param time_horizon: The value giving the end value of the interval where the SDE is approximated.
    :param delta: Time discretization used by the solvers.
    :param generator: NumPy random generator to use for PRNG.
    :param scale_noise: Scaling factor increasing/decreasing the impact of the diffusion term in the SDE.
    :return: The time-discrete approximation of the SDE.
    """
    n_points = int(np.floor(time_horizon / delta))  # Compute actual number of points in the discretization
    dimension = init.shape[0]  # Dimensionality of the vector SDE

    # Compute brownian motion increments
    bm_increments = generator.normal(0, np.sqrt(delta), size=(dimension, n_points))
    process = np.zeros(shape=(dimension, n_points + 1), dtype=np.float64)  # Preallocate discretized process vector
    process[:, 0] = init  # Set the initial value

    for i in range(n_points):  # Sequentially iterate according to the gaussian approximation
        drift, diffusion = coefficient(
            process[:, i], milstein=False, scale_noise=scale_noise
        )  # Get drift and diffusion
        deterministic = drift * delta  # Deterministic part of the step
        noise = diffusion @ bm_increments[:, i]  # Stochastic part of the step
        process[:, i + 1] = process[:, i] + deterministic + noise  # Save new point

    return process


def milstein_sim(
    coefficient,
    init: np.ndarray,
    time_horizon: float,
    delta: float,
    generator: np.random.Generator,
    scale_noise: float = 1.0,
):
    """
    Milstein SDE solver assuming diagonal noise.

    :param coefficient: An instance of the coefficient set characterizing the SDE:
    :param init: Initial value of the SDE.
    :param time_horizon: The value giving the end value of the interval where the SDE is approximated.
    :param delta: Time discretization used by the solvers.
    :param generator: NumPy random generator to use for PRNG.
    :param scale_noise: Scaling factor increasing/decreasing the impact of the diffusion term in the SDE.
    :return: The time-discrete approximation of the SDE.
    """
    n_points = int(np.floor(time_horizon / delta))  # Compute actual number of points in the discretization
    dimension = init.shape[0]  # Dimensionality of the vector SDE

    # Compute brownian motion increments
    bm_increments = generator.normal(0, np.sqrt(delta), size=(dimension, n_points))
    process = np.zeros(shape=(dimension, n_points + 1), dtype=np.float64)  # Preallocate discretized process vector
    process[:, 0] = init  # Set the initial value

    for i in range(n_points):  # Sequentially iterate according to first order Ito-Taylor expansion
        drift, diffusion_diag, derivative_diag_diff = coefficient(
            process[:, i], scale_noise=scale_noise
        )  # Get drift and diffusion and needed partial derivatives
        deterministic = drift * delta  # Deterministic part of the step
        noise = diffusion_diag * bm_increments[:, i]  # Stochastic part of the step
        higher_order_noise = (
            1
            / 2
            * derivative_diag_diff
            * diffusion_diag
            * ((bm_increments[:, i] ** 2) - delta)
        )  # Higher order noise: this is what distinguishes Milstein from Euler-Maruyama
        process[:, i + 1] = process[:, i] + deterministic + noise + higher_order_noise  # Save new point

    return process
