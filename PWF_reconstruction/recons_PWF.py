from .utils import R_earth, R2D, ns, kr, c_light, n_atm
import numpy as np
from scipy.optimize import brentq
import scipy.optimize as so
import numdifftools as nd


def Linear_solver(Xants, tants, c=c_light, n=n_atm):
    """
    Solve for the best-fit vector k given antenna positions and arrival times.

    Parameters:
    Xants (ndarray): Antenna positions in meters, shape (nants, 3).
    tants (ndarray): Antenna arrival times in seconds, shape (nants,).
    c (float): Speed of light in m/s, default is  299792458 m/s
    n (float or ndarray): Indices of refraction (vector or constant), default is 1.000136

    Returns:
    tuple: Best-fit vector k (ndarray) and the pseudoinverse of the design matrix (ndarray).
    """
    t_ants = (c/n * (tants - tants.mean()))[:, None]
    P_1 = (Xants - Xants.mean(axis=0))

    pseudoinverse = np.linalg.pinv(P_1.T @ P_1)
    M = pseudoinverse @ P_1.T
    res = M @ t_ants
    return res.flatten(), pseudoinverse


def _projector(k, Inv):
    """
    Compute the projection of vector k on the unit sphere along the largest axis of distribution.

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


def PWF_projection(Xants, tants, c=c_light, n=n_atm):
    """
    Compute the projection of k on the unit sphere along the largest axis of distribution.

    Parameters:
    Xants (ndarray): Antenna positions in meters, shape (nants, 3).
    tants (ndarray): Antenna arrival times in seconds, shape (nants,).
    c (float): Speed of light in m/s, default is  299792458 m/s
    n (float or ndarray): Indices of refraction (vector or constant), default is 1.000136

    Returns:
    tuple: Theta and phi angles in radians.
    """
    k_lin, Inv = Linear_solver(Xants, tants, c=c, n=n)
    k_opt = _projector(k_lin, Inv)

    theta_opt = np.arccos(-k_opt[2])
    phi_opt = np.arctan2(-k_opt[1], -k_opt[0])
    return theta_opt, phi_opt


def PWF_semianalytical(Xants, tants, verbose=False, c=c_light, n=n_atm):
    """
    Solve the minimization problem using a semi-analytical approach.

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
    PXT = Xants - Xants.mean(axis=0)
    A = np.dot(Xants.T, PXT)
    b = np.dot(Xants.T, tants_cor - tants_cor.mean(axis=0))
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
    return np.array([theta_opt, phi_opt])


def Covariance_tangentplane(theta_pred, phi_pred, Xants, sigma, c=c_light, n=n_atm):
    """
    Compute the covariance matrix of theta and phi given predictions and antenna data.

    Parameters:
    theta_pred (float): Predicted theta angle in radians.
    phi_pred (float): Predicted phi angle in radians.
    Xants (ndarray): Antenna positions in meters, shape (nants, 3).
    sigma (float): Standard deviation of arrival times.
    c (float): Speed of light in m/s, default is  299792458 m/s
    n (float or ndarray): Indices of refraction (vector or constant), default is 1.000136

    Returns:
    ndarray: Covariance matrix of theta and phi, shape (2, 2).
    """
    Xants_cor = (Xants - Xants.mean(axis=0)[None, :]) / (c / np.array(n).reshape(-1, 1))
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


def fisher_Variance(theta_pred, phi_pred, Xants, sigma, c=c_light, n=n_atm):
    """
    Compute the Fisher matrix variance for theta and phi given predictions and antenna data.

    Parameters:
    theta_pred (float): Predicted theta angle in radians.
    phi_pred (float): Predicted phi angle in radians.
    Xants (ndarray): Antenna positions in meters, shape (nants, 3).
    sigma (float): Standard deviation of arrival times.
    c (float): Speed of light in m/s, default is  299792458 m/s
    n (float or ndarray): Indices of refraction (vector or constant), default is 1.000136

    Returns:
    ndarray: Fisher matrix variance, shape (2, 2).
    """
    B = np.array([
        [-np.cos(theta_pred)*np.cos(phi_pred), np.sin(theta_pred)*np.sin(phi_pred)],
        [-np.cos(theta_pred)*np.sin(phi_pred), -np.sin(theta_pred)*np.cos(phi_pred)],
        [np.sin(theta_pred), 0]
    ])
    Xants_cor = (Xants - Xants.mean(axis=0)[None, :]) / (c / np.array(n).reshape(-1, 1))

    return np.linalg.pinv(B.T @ Xants_cor.T @ Xants_cor @ B) * (sigma**2)


def cov_matrix(theta_pred, phi_pred, Xants, sigma, c=c_light, n=n_atm):
    """
    Wrapper for Covariance_tangentplane function.

    Parameters:
    theta_pred (float): Predicted theta angle in radians.
    phi_pred (float): Predicted phi angle in radians.
    Xants (ndarray): Antenna positions in meters, shape (nants, 3).
    sigma (float): Standard deviation of arrival times.
    c (float): Speed of light in m/s, default is  299792458 m/s
    n (float or ndarray): Indices of refraction (vector or constant), default is 1.000136

    Returns:
    ndarray: Covariance matrix of theta and phi, shape (2, 2).
    """
    return Covariance_tangentplane(theta_pred, phi_pred, Xants, sigma, c=c, n=n)


def PWF_loss(params, Xants, tants, verbose=False):
    """
    Define the Chi-squared loss function for antenna data.

    Parameters:
    params (tuple): Spherical coordinates (theta, phi) of unit shower direction vector K.
    Xants (ndarray): Antenna positions in meters, shape (nants, 3).
    tants (ndarray): Antenna arrival times in m, shape (nants,).
    verbose (bool): Verbose output, default is False.

    Returns:
    float: Chi-squared loss value.
    """
    theta, phi = params
    nants = tants.shape[0]
    ct = np.cos(theta)
    st = np.sin(theta)
    cp = np.cos(phi)
    sp = np.sin(phi)
    K = np.array([st*cp, st*sp, ct])

    if (Xants.shape[0] != nants):
        print("Shapes of tants and Xants are incompatible", tants.shape, Xants.shape)
        return None

    xk = np.dot(Xants, K)
    DXK = np.subtract.outer(xk, xk)
    DT = np.subtract.outer(tants, tants)
    chi2 = ((DXK - DT)**2).sum() / 2.
    
    if verbose:
        print("params = ", np.rad2deg(params))
        print("Chi2 = ", chi2)
    return chi2


def PWF_gradient(Xants, tants, c=c_light, n=n_atm):
    """
    Perform gradient descent to find the best-fit parameters for antenna data.

    Parameters:
    Xants (ndarray): Antenna positions in meters, shape (nants, 3).
    tants (ndarray): Antenna arrival times in seconds, shape (nants,).
    c (float): Speed of light in m/s, default is  299792458 m/s
    n (float or ndarray): Indices of refraction (vector or constant), default is 1.000136

    Returns:
    ndarray: Best-fit parameters (theta, phi) in radians.
    """
    bounds = ((np.pi / 2 + 1e-7, np.pi), (0, 2 * np.pi))
    params_in = np.array(bounds).mean(axis=1)

    args = (Xants, tants * c / n)

    res = so.minimize(PWF_loss, params_in, args=args, bounds=bounds, method='BFGS')

    params_out = res.x
    params_out = params_out % (2 * np.pi)
    if params_out[0] > np.pi:
        params_out[0] = 2 * np.pi - params_out[0]
        params_out[1] = (params_out[1] + np.pi) % (2 * np.pi)

    if params_out[0] < np.pi / 2:
        params_in = params_out
        params_out[0] = np.pi - params_out[0]
        res = so.minimize(PWF_loss, params_out, args=args)

        params_out = res.x % (2 * np.pi)
        if params_out[0] > np.pi:
            params_out[0] = 2 * np.pi - params_out[0]
            params_out[1] = (params_out[1] + np.pi) % (2 * np.pi)

    params_out[0] = (np.pi - params_out[0]) % (2 * np.pi)
    params_out[1] = (np.pi + params_out[1]) % (2 * np.pi)
    return params_out
