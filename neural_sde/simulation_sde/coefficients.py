import numpy as np
from typing import List
from abc import ABC, abstractmethod


class SDECoefficient(ABC):
    @abstractmethod
    def __call__(
        self, x: np.ndarray, milstein: bool = True, scale_noise: float = 1.0
    ) -> List[np.ndarray]:
        """
        Method returning drift, diffusion and - optionally - the derivatives needed for the Milstein scheme: everything
        is evaluated in a specific point of the process.
        :param x: The point where the functions are evaluated. Must be a 1D NumPy array of appropriate length.
        :param milstein: Boolean indicating whether the external routine calling the method is a Milstein scheme solver.
        :param scale_noise: A scaling term to decrease/increase the scale of the noise/diffusion term.
        :return: A list with drift, diffusion and partial derivatives of the diffusion (if needed).
         The diffusion is returned as a matrix if we are not assuming Milstein, otherwise only the diagonal is returned.
        """
        raise NotImplementedError("This is only an abstract class")

    @abstractmethod
    def drift(self, x: np.ndarray) -> np.ndarray:
        """
        Function evaluating the drift for a set of points, passed as a (d, n) NumPy array, with "n"
        being the number of points and "d" being the SDE dimensionality.
        :param x: The array containing the points where to evaluate the drift.
        :return: The evaluated drift, a (d, n) NumPy array.
        """
        raise NotImplementedError("This is only an abstract class")


def sigmoid(x):
    out = 1.0 / (1.0 + np.exp(-x))
    return out


class SinusoidDriftSigmoidDiffusion(SDECoefficient):
    """
    The coefficient class for a two-dimensional diffusion model satisfying ergodic assumptions. This model has
    a drift having a negative linear component and a sinusoidal one; the diffusion coefficient is a diagonal matrix
    where each diagonal element is the sum of a constant term and a value given by a sigmoid function.
    """

    def __init__(
        self,
        alpha_1: float,
        alpha_2: float,
        alpha_3: float,
        alpha_4: float,
        beta_1: float,
        beta_2: float,
        beta_3: float,
    ):
        """
        Initialization method for the coefficient class.
        :param alpha_1: The term scaling the first component of the SDE in the first component of the drift. This term
         is negated in the expression.
        :param alpha_2: The term scaling the sinusoidal function of the second component of the SDE in the first
         component of the drift.
        :param alpha_3: The term scaling the sinusoidal function of the first component of the SDE in the
         second component of the drift.
        :param alpha_4: The term scaling the second component of the SDE in the second component of the drift. This term
         is negated in the expression.
        :param beta_1: The term scaling the sigmoid function of the first component of the SDE in the first element
         of the diagonal of the diffusion.
        :param beta_2: The term scaling the sigmoid function of the second component of the SDE in the second element
         of the diagonal of the diffusion.
        :param beta_3: The constant term in the diagonal of the diffusion. This needs to be strictly positive.
        """
        if not (  # Check whether the parameters are valid
            alpha_1 > 0
            and alpha_2 >= 0
            and alpha_3 >= 0
            and alpha_4 > 0
            and abs(beta_3) > 0
            and abs(beta_1) >= 0
            and abs(beta_2) >= 0
        ):
            raise ValueError("Parameterization does not respect the constraints.")
        self.params = {  # Save the parameters
            "alpha_1": alpha_1,
            "alpha_2": alpha_2,
            "alpha_3": alpha_3,
            "alpha_4": alpha_4,
            "beta_1": beta_1,
            "beta_2": beta_2,
            "beta_3": beta_3,
        }

    def __call__(
        self, x: np.ndarray, milstein: bool = True, scale_noise: float = 1.0
    ) -> List[np.ndarray]:

        if len(x) != 2:
            raise ValueError(
                "This is a 2D coefficient, check the input shape"
            )  # Check whether input shape is correct

        # Evaluate drift
        drift = np.array(
            [
                -self.params["alpha_1"] * x[0]
                + self.params["alpha_2"] * (np.sin(x[1]) + 2),
                self.params["alpha_3"] * (np.cos(x[0]) + 2)
                - self.params["alpha_4"] * x[1],
            ],
            dtype=np.float32,
        )

        if milstein:  # Check if milstein
            # The Milstein implementation we use assumes diagonal noise, so we can just return the diagonal
            # of the diffusion as a vector
            diffusion = scale_noise * np.array(
                [
                    self.params["beta_1"] * sigmoid(x[0]) + self.params["beta_3"],
                    self.params["beta_2"] * sigmoid(x[1]) + self.params["beta_3"],
                ],
                dtype=np.float32,
            )
            # The noise is assumed to be "diagonal": Milstein's scheme only needs these two partial derivatives.
            derivative_diff = scale_noise * np.array(
                [
                    self.params["beta_1"] * sigmoid(x[0]) * (1 - sigmoid(x[0])),
                    self.params["beta_2"] * sigmoid(x[1]) * (1 - sigmoid(x[1])),
                ],
                dtype=np.float32,
            )
        else:  # If not milstein with diagonal noise, the diffusion is returned as a matrix
            diffusion = scale_noise * np.array(
                [
                    [self.params["beta_1"] * sigmoid(x[0]) + self.params["beta_3"], 0],
                    [
                        0,
                        self.params["beta_2"] * sigmoid(x[1]) + self.params["beta_3"],
                    ],
                ],
                dtype=np.float32,
            )

        if milstein:
            return drift, diffusion, derivative_diff
        else:
            return drift, diffusion

    def drift(self, x: np.ndarray) -> np.ndarray:

        if len(x) != 2:
            raise ValueError(
                "This is a 2D coefficient, check the input shape"
            )  # Check whether input shape is correct

        out = np.concatenate(
            [
                np.expand_dims(
                    -self.params["alpha_1"] * x[0, ...]
                    + self.params["alpha_2"] * (np.sin(x[1, ...]) + 2),
                    axis=0,
                ),
                np.expand_dims(
                    self.params["alpha_3"] * (np.cos(x[0, ...]) + 2)
                    - self.params["alpha_4"] * x[1, ...],
                    axis=0,
                ),
            ],
            axis=0,
            dtype=np.float32,
        )
        return out
