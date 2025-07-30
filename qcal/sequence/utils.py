"""Helper functions for sequencer.

"""
import logging
import numpy as np

from numpy.typing import NDArray
from scipy.linalg import solve
from typing import List

logger = logging.getLogger(__name__)


# def calculate_third_order_z_component(
#         theta_t: NDArray, 
#         theta_dot: NDArray, 
#         theta_ddot: NDArray, 
#         theta_dddot: NDArray, 
#         omega_eff: float
#     ) -> NDArray:
#     """Calculate the Z-component of third-order superadiabatic corrections.

#     Based on the theory of superadiabatic transformations, the third-order
#     Z-component involves multiple terms from nested commutators.

#     Args:
#         theta_t (NDArray): phase as a function of time.
#         theta_dot (NDArray): first time derivative of the phase.
#         theta_ddot (NDArray): second time derivative of the phase.
#         theta_dddot (NDArray): third time derivative of the phase.
#         omega_eff (float): effective Rabi frequency.

#     Returns:
#         NDArray: third order Z-component.
#     """
#     # Third-order Z-component in the original frame
#     # These formulas come from the recursive superadiabatic transformation
    
#     # Term 1: Direct third-order contribution
#     term1 = theta_dddot * np.cos(theta_t) / (8 * omega_eff**2)
    
#     # Term 2: Mixed derivative term
#     term2 = -3 * theta_dot * theta_ddot * np.cos(theta_t) / (4 * omega_eff**3)
    
#     # Term 3: Velocity-cubed term
#     term3 = theta_dot**3 * np.sin(theta_t) * np.cos(theta_t) / (2 * omega_eff**3)
    
#     # Term 4: Correction from second-order back-transformation
#     term4 = -theta_ddot * theta_dot * np.sin(theta_t) / (4 * omega_eff**2)
    
#     # Total Z-component
#     cd3_z = term1 + term2 + term3 + term4
    
#     return cd3_z


def clip_amplitude(
        amp: float | NDArray, min_amp: float = -1.0, max_amp: float = 1.0
    ) -> float | NDArray:
    """Clip the amplitude of a pulse to a maximum value.

    Args:
        amp (float | NDArray): amplitude value or pulse arrray.
        min_amp (float, optional): minimum amplitude. Defaults to -1.0.
        max_amp (float, optional): maximum amplitude. Defaults to 1.0.

    Returns:
        NDArray: pulse array with clipped amplitude.
    """
    return np.clip(amp, min_amp, max_amp)


def compute_matrix_A(
        N: int, freq_intervals: List, weights: List, t_p: float
    ) -> NDArray:
    """Compute matrix A for the optimization problem
    A_nm = sum_j w_j * integral(g_hat_n(f) * g_hat_m*(f) df)

    Args:
        N (int): number of Fourier terms
        freq_intervals (List): list of [f_low, f_high] pairs
        weights (List): weights for each frequency interval
        t_p (float): pulse duration

    Returns:
        NDArray: A matrix.
    """
    A = np.zeros((N, N))
    
    for n in range(N):
        for m in range(N):
            integral_sum = 0
            
            for j, (f_low, f_high) in enumerate(freq_intervals):
                weight = weights[j]
                
                # Numerical integration over frequency interval
                n_points = 200
                f_range = np.linspace(f_low, f_high, n_points)
                
                if len(f_range) > 1:
                    df = f_range[1] - f_range[0]
                    
                    # Get Fourier transforms
                    g_n_hat = fourier_transform_basis(n+1, f_range, t_p)
                    g_m_hat = fourier_transform_basis(m+1, f_range, t_p)
                    
                    # Compute integrand
                    integrand = g_n_hat * np.conj(g_m_hat)
                    
                    # Numerical integration (trapezoidal rule)
                    integral_sum += weight * np.trapz(integrand.real, dx=df)
            
            A[n, m] = integral_sum
    
    return A


def cosine_basis_function(n: int, t: NDArray, t_p: float) -> NDArray:
    """Basis function g_n(t) = [1 - cos(2πnt/t_p)] * Π(t/t_p - 1/2) for a
    cosine series.

    Args:
        n (int): basis function index.
        t (NDArray): time array.
        t_p (float): pulse duration.

    Returns:
        NDArray: Cosine dressed by a rectangular window.
    """
    # Rectangular window function
    window = np.where((t >= 0) & (t <= t_p), 1.0, 0.0)
    
    # Cosine basis
    cosine_term = 1 - np.cos(2 * np.pi * n * t / t_p)
    
    return cosine_term * window


def fourier_transform_basis(n: int, f: NDArray, t_p: float) -> NDArray:
    """Analytical Fourier transform of basis function g_n(t).

    Args:
        n (int): basis function index.
        f (NDArray): frequency array.
        t_p (float): pulse duration.

    Returns:
        NDArray: Fourier transform of the basis function.
    """
    # Handle array inputs properly
    f = np.asarray(f)
    
    # Initialize result array
    result = np.zeros_like(f, dtype=complex)
    
    # Handle f=0 case
    zero_mask = np.abs(f) < 1e-12
    result[zero_mask] = t_p / 2
    
    # Handle non-zero frequencies
    nonzero_mask = ~zero_mask
    if np.any(nonzero_mask):
        f_nz = f[nonzero_mask]
        
        # Simplified analytical form for the Fourier transform
        # of [1 - cos(2πnt/t_p)] * rect(t/t_p)
        sinc_term = np.sinc(f_nz * t_p)
        delta_pos = np.sinc((f_nz - n/t_p) * t_p)
        delta_neg = np.sinc((f_nz + n/t_p) * t_p)
        
        result[nonzero_mask] = t_p * (sinc_term - 0.5 * (delta_pos + delta_neg))
    
    return result


def solve_coefficients( 
        N:              int, 
        t_p:            float, 
        theta:          float, 
        freq_intervals: List, 
        weights:        List, 
    ) -> NDArray:
    """Solve for Fourier coefficients using the matrix equation.

    Args:
        N (int): number of Fourier terms.
        t_p (float): pulse duration.
        theta (float): rotation angle.
        freq_intervals (List): list of [f_low, f_high] pairs.
        weights (List): weights for each frequency interval.

    Returns:
        NDArray: Fourier coefficients.
    """
    # Compute matrix A
    A_matrix = compute_matrix_A(N, freq_intervals, weights, t_p)
    
    # Add small regularization to avoid singular matrix
    A_matrix += 1e-10 * np.eye(N)
    
    # Set up the augmented matrix equation
    # [A + A^T  -b] [c]   [0]
    # [b^T       0] [μ] = [θ/t_p]
    
    b = np.ones((N, 1))
    zero_vec = np.zeros((N, 1))
    
    # Construct the augmented matrix
    top_left = A_matrix + A_matrix.T
    top_right = -b
    bottom_left = b.T
    bottom_right = np.array([[0]])
    
    aug_matrix = np.block([[top_left, top_right],
                           [bottom_left, bottom_right]])
    
    # Right-hand side
    rhs = np.vstack([zero_vec, [[theta / t_p]]])
    
    try:
        # Solve the system
        solution = solve(aug_matrix, rhs)
        # Extract coefficients (exclude Lagrangian multiplier)
        coefficients = solution[:-1].flatten()
    except np.linalg.LinAlgError:
        # Fallback: use pseudoinverse if matrix is singular
        logger.warning(" Using pseudoinverse due to singular matrix!")
        coefficients = np.linalg.pinv(aug_matrix) @ rhs
        coefficients = coefficients[:-1].flatten()
    
    return coefficients
