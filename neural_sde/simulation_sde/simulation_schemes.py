import numpy as np
from .coefficients import SDECoefficient
from typing import Optional
from tqdm import tqdm


def euler_sim(
    coefficient: SDECoefficient,
    init: np.ndarray,
    time_horizon: float,
    delta: float,
    generator: np.random.Generator,
    scale_noise: float = 1.0,
    scale_shift_process: Optional[np.ndarray] = None,
):
    """
    Euler-Maruyama SDE solver.

    :param coefficient: An instance of the coefficient set characterizing the SDE.
    :param init: Initial value of the SDE.
    :param time_horizon: The value giving the end value of the interval where the SDE is approximated.
    :param delta: Time discretization used by the solvers.
    :param generator: NumPy random generator to use for PRNG.
    :param scale_noise: Scaling factor increasing/decreasing the impact of the diffusion term in the SDE.
    :param scale_shift_process: Scale and shift for the whole process, for each component.
     First column is scale, second is shift.
    :return: The time-discrete approximation of the SDE.
    """
    n_points = int(
        np.floor(time_horizon / delta)
    )  # Compute actual number of points in the discretization
    dimension = init.shape[0]  # Dimensionality of the vector SDE

    # Compute brownian motion increments
    bm_increments = generator.normal(0, np.sqrt(delta), size=(dimension, n_points))
    process = np.zeros(
        shape=(dimension, n_points + 1), dtype=np.float32
    )  # Preallocate discretized process vector
    process[:, 0] = init  # Set the initial value

    for i in range(
        n_points
    ):  # Sequentially iterate according to the gaussian approximation
        drift, diffusion = coefficient(
            process[:, i], milstein=False, scale_noise=scale_noise
        )  # Get drift and diffusion
        deterministic = drift * delta  # Deterministic part of the step
        noise = diffusion @ bm_increments[:, i]  # Stochastic part of the step
        process[:, i + 1] = process[:, i] + deterministic + noise  # Save new point

    # Scale and shift the process, if needed
    if scale_shift_process is not None:
        process = scale_shift_process[:, 0][:, np.newaxis] * (
            process + scale_shift_process[:, 1][:, np.newaxis]
        )

    return process


def milstein_sim(
    coefficient: SDECoefficient,
    init: np.ndarray,
    time_horizon: float,
    delta: float,
    generator: np.random.Generator,
    scale_noise: float = 1.0,
    scale_shift_process: Optional[np.ndarray] = None,
):
    """
    Milstein SDE solver assuming diagonal noise.

    :param coefficient: An instance of the coefficient set characterizing the SDE.
    :param init: Initial value of the SDE.
    :param time_horizon: The value giving the end value of the interval where the SDE is approximated.
    :param delta: Time discretization used by the solvers.
    :param generator: NumPy random generator to use for PRNG.
    :param scale_noise: Scaling factor increasing/decreasing the impact of the diffusion term in the SDE.
    :param scale_shift_process: Scale and shift for the whole process, for each component.
     First column is scale, second is shift.
    :return: The time-discrete approximation of the SDE.
    """
    n_points = int(
        time_horizon / delta
    )  # Compute actual number of points in the discretization
    dimension = init.shape[0]  # Dimensionality of the vector SDE

    # Compute brownian motion increments
    bm_increments = generator.normal(0, np.sqrt(delta), size=(dimension, n_points))
    process = np.zeros(
        shape=(dimension, n_points + 1), dtype=np.float32
    )  # Preallocate discretized process vector
    process[:, 0] = init  # Set the initial value

    for i in range(
        n_points
    ):  # Sequentially iterate according to first order Ito-Taylor expansion
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
        process[:, i + 1] = (
            process[:, i] + deterministic + noise + higher_order_noise
        )  # Save new point

    # Scale and shift the process, if needed
    if scale_shift_process is not None:
        process = scale_shift_process[:, 0][:, np.newaxis] * (
            process + scale_shift_process[:, 1][:, np.newaxis]
        )

    return process


def test_compact(
    coefficient: SDECoefficient,
    init: np.ndarray,
    time_horizon: float,
    delta: float,
    generator: np.random.Generator,
    compact_set: np.ndarray,
    scale_noise: float = 1.0,
    scale_shift_process: Optional[np.ndarray] = None,
    milstein: bool = True,
    mc_iter: int = 100000
):
    result_array = np.zeros(mc_iter, np.int32)
    for i in tqdm(range(mc_iter), total=mc_iter):
        if milstein:
            process = milstein_sim(
                coefficient,
                init,
                time_horizon,
                delta,
                generator,
                scale_noise,
                scale_shift_process,
            )
        else:
            process = euler_sim(
                coefficient,
                init,
                time_horizon,
                delta,
                generator,
                scale_noise,
                scale_shift_process,
            )

        mask_compact = (np.all((process <= compact_set[:, 1][:, np.newaxis]), axis=0)) & (
            np.all(process >= compact_set[:, 0][:, np.newaxis], axis=0)
        )

        outside_n = np.sum(mask_compact)
        result_array[i] = outside_n

    return result_array
