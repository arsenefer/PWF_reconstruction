# Code from Ferriere et al. (2024) : https://doi.org/10.48550/arXiv.2408.15677

from .utils import c_light, n_atm, mean, _inv_cho
import numpy as np
from scipy.optimize import brentq


def Linear_solver(Xants, tants, c=c_light, n=n_atm, sigma=None):
    """
    Solve for the best-fit vector k* given antenna positions and arrival times with no constraint.
    (Equation 9 in the paper)

    Parameters:
    Xants (ndarray): Antenna positions in meters, shape (nants, 3).
    tants (ndarray): Antenna arrival times in seconds, shape (nants,).
    c (float): Speed of light in m/s, default is  299792458 m/s
    n (float or ndarray): Indices of refraction (vector or constant), default is 1.000136

    Returns:
    tuple: Best-fit vector k (ndarray) and the pseudoinverse of the design matrix (ndarray).
    """
    t_ants = (c/n * (tants - mean(tants, sigma)))[:, None]
    P_1 = (Xants - mean(Xants, sigma)[None, :])
    if type(sigma) is np.ndarray and sigma.ndim==2:
        Q_1 = _inv_cho(sigma)
        pseudoinverse = np.linalg.pinv(P_1.T @ Q_1 @ P_1 )
        M = pseudoinverse @ P_1.T @ Q_1
    elif type(sigma) is np.ndarray and sigma.ndim==1:
        P1_scaled = (1/sigma[:,None]**2) * P_1
        pseudoinverse = np.linalg.pinv(P_1.T @ ( P1_scaled ))
        M = pseudoinverse @ P1_scaled.T
    else:
        pseudoinverse = np.linalg.pinv(P_1.T @ P_1)
        M = pseudoinverse @ P_1.T

    res = M @ t_ants
    return res.flatten(), pseudoinverse


def _projector(k, Inv):
    """
    Compute the projection of vector k* on the unit sphere along the largest axis of distribution.
    (Equation 12 in the paper)

    Parameters:
    k (ndarray): Vector to be projected, shape (3,).
    Inv (ndarray): Inverse of the covariance matrix, shape (3, 3).

    Returns:
    ndarray: Projected vector on the unit sphere, shape (3,).
    """
    d, R = np.linalg.eigh(Inv)
    directions = R[:, -1] @ np.array([[0.], [0], [1]])
    R[:, -1] *= np.sign(directions).T

    k_lin_rot = R.T @ k
    n2 = np.linalg.norm(k_lin_rot[:2])
    k_opt_rot = np.array(
        [*(k_lin_rot[:2] / max(1, n2)), -np.sqrt(1 - min(1, n2**2))])

    k_opt = R @ k_opt_rot
    return k_opt


def PWF_projection(Xants, tants, c=c_light, n=n_atm, sigma=None):
    """
    Computes the projection method described in the paper. Gives a very good approximation of the polar angles $\\theta$ and $\\phi$ 
    (See paper section 2.1.1)

    Parameters:
    Xants (ndarray): Antenna positions in meters, shape (nants, 3).
    tants (ndarray): Antenna arrival times in seconds, shape (nants,).
    c (float): Speed of light in m/s, default is  299792458 m/s
    n (float or ndarray): Indices of refraction (vector or constant), default is 1.000136

    Returns:
    tuple: Theta and phi angles in radians.
    """
    k_lin, Inv = Linear_solver(Xants, tants, c=c, n=n, sigma=sigma)
    k_opt = _projector(k_lin, Inv)

    theta_opt = np.arccos(-k_opt[2])
    phi_opt = np.arctan2(-k_opt[1], -k_opt[0])
    return np.array((theta_opt, phi_opt))

def PWF_semianalytical(Xants, tants, verbose=False, c=c_light, n=n_atm, sigma=None):
    """
    Solve the minimization problem using a semi-analytical approach.
    (see section 2.1.2)

    Parameters:
    Xants (ndarray): Antenna positions in meters, shape (nants, 3).
    tants (ndarray): Antenna arrival times in seconds, shape (nants,).
    verbose (bool): Verbose output, default is False.
    c (float): Speed of light in m/s, default is  299792458 m/s
    n (float or ndarray): Indices of refraction (vector or constant), default is 1.000136

    Returns:
    ndarray: Theta and phi angles in radians.
    """
    nants = tants.shape[0]

    if (Xants.shape[0] != nants):
        print("Shapes of tants and Xants are incompatible", tants.shape, Xants.shape)
        return None

    tants_cor = tants * c / n
    PXT = Xants - mean(Xants, sigma)[None, :]
    A = np.dot(Xants.T, PXT)
    b = np.dot(Xants.T, tants_cor - mean(tants_cor, sigma))
    d, W = np.linalg.eigh(A)
    beta = np.dot(b, W)
    nbeta = np.linalg.norm(beta)

    if (np.abs(beta[0] / nbeta) < 1e-14):
        if (verbose):
            print("Degenerate case")
        mu = -d[0]
        c_ = np.zeros(3)
        c_[1] = beta[1] / (d[1] + mu)
        c_[2] = beta[2] / (d[2] + mu)
        si = np.sign(np.dot(W[:, 0], np.array([0, 0, 1.])))
        c_[0] = -si * np.sqrt(1 - c_[1]**2 - c_[2]**2)
        k_opt = np.dot(W, c_)

    else:
        def nc(mu):
            c_ = beta / (d + mu)
            return ((c_**2).sum() - 1.)
        mu_min = -d[0] + beta[0]
        mu_max = -d[0] + np.linalg.norm(beta)
        mu_opt = brentq(nc, mu_min, mu_max, maxiter=1000)
        c_ = beta / (d + mu_opt)
        k_opt = np.dot(W, c_)

    if k_opt[2] > 1e-2:
        k_opt = k_opt - 2 * (k_opt @ W[:, 0]) * W[:, 0]

    theta_opt = np.arccos(-k_opt[2])
    phi_opt = np.arctan2(-k_opt[1], -k_opt[0])

    if phi_opt < 0:
        phi_opt += 2 * np.pi
    return theta_opt, phi_opt, k_opt

def cov_matrix(theta_pred, phi_pred, Xants, sigma, c=c_light, n=n_atm, method=1):
    """
    if method 1 :
    Compute the Fisher matrix variance for theta and phi given predictions and antenna data.
    Obtained from equation 24 in the paper.
    else:
    Compute the covariance matrix of $\\theta$ and $\\phi$ given prediction, antenna data and std vector of timings for all antennas. 
    Adapted from equation 22 in the paper.

    Parameters:
    theta_pred (float): Predicted theta angle in radians.
    phi_pred (float): Predicted phi angle in radians.
    Xants (ndarray): Antenna positions in meters, shape (nants, 3).
    sigma (float): Standard deviation of arrival times.
    c (float): Speed of light in m/s, default is  299792458 m/s
    n (float or ndarray): Indices of refraction (vector or constant), default is 1.000136

    Returns:
    ndarray: Covariance matrix of theta and phi in radians^2, shape (2, 2).
    """
    if method==1:
        B = np.array([
            [-np.cos(theta_pred)*np.cos(phi_pred), np.sin(theta_pred)*np.sin(phi_pred)],
            [-np.cos(theta_pred)*np.sin(phi_pred), -np.sin(theta_pred)*np.cos(phi_pred)],
            [np.sin(theta_pred), 0]
        ])
        Xants_cor = (Xants - mean(Xants, sigma)[None, :]) / (c / np.array(n).reshape(-1, 1))
        return np.linalg.pinv(B.T @ Xants_cor.T @ Xants_cor @ B) * (sigma**2)
    
    else:
        Xants_cor = (Xants - mean(Xants, sigma)[None, :]) / (c / np.array(n).reshape(-1, 1))
        Sigma = (sigma)**2 * np.linalg.pinv(Xants_cor.T @ Xants_cor)

        Q = np.linalg.pinv(np.array([[-np.sin(theta_pred)*np.cos(phi_pred), -np.cos(theta_pred)*np.cos(phi_pred), np.sin(theta_pred)*np.sin(phi_pred)],
                                    [-np.sin(theta_pred)*np.sin(phi_pred), -np.cos(theta_pred)*np.sin(phi_pred), -np.sin(theta_pred)*np.cos(phi_pred)],
                                    [-np.cos(theta_pred), np.sin(theta_pred), 0]]))
        QSigQt = Q @ Sigma @ Q.T

        Sigma_aa = QSigQt[1:, 1:]
        Sigma_ar = QSigQt[0, 1:]
        Sigma_rr = QSigQt[0, 0]
        Sigma_bar = Sigma_aa - 1 / Sigma_rr * Sigma_ar[:, None] @ Sigma_ar[None, :]
        return Sigma_bar

def angular_error(theta_pred, Covar):
    """
    Compute the pointing direction error from the zenith angle and the covariance matrix. 
    (from eq., with square root)

    Parameters:
    theta_pred (float): Predicted theta angle in radians.
    Covar (ndarray): Covariance matrix of theta and phi in radians^2, shape (2, 2).
    Returns:
    float: absolute pointing accuracy in radians.
    """
    return np.sqrt(Covar[0,0] + np.sin(theta_pred)**2 * Covar[1,1])



def PWF_time(k_opt):
    """
    Calculate the PWF_time using the provided k_opt from PWF_semianalytical.

    Parameters:
    k_opt (numpy array): The optimal k values from PWF_semianalytical.

    Returns:
    numpy array: The calculated PWF_time in nanoseconds.
    """
    # Calculate distances by projecting the difference of Xants and its mean onto k_opt
    distances = np.dot(Xants - Xants.mean(axis=0), k_opt)
    
    # Calculate expected times by dividing distances by the speed of light (in meters per second)
    expected_times = distances / c_light
    
    # Convert expected times to nanoseconds
    PWF_time = expected_times * 10**9
    
    return PWF_time



def chi2_PWF(t_meas, t_PWF):
    """
    Calculates the chi-squared value for a time reconstruction - in this case a PWF :)

    Args:
        t_meas: A NumPy array of measured (or simulated) times.
        t_PWF: A NumPy array of reconstructed times.

    Returns:
        The chi-squared value (float). 
    """
    sigma = np.std(t_meas, ddof=1)
    time_diff = t_meas - t_PWF
    chi2_value = np.sum((time_diff / sigma)**2)
    return chi2_value
