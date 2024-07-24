from .utils import R_earth, R2D, ns, kr, c_light, n_atm
import numpy as np
from scipy.optimize import brentq
import scipy.optimize as so
import numdifftools as nd


def Linear_solver(Xants, tants, c=c_light, n=n_atm):
    """
    Solve for the best-fit vector k given antenna positions and arrival times.
    (from eq.)

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
    (from eq.)

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
    (from eq.)

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
    (from eq.)

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
    (from eq.)

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


def Covariance_schurcomplement(theta_pred, phi_pred, Xants, sigma, c=c_light, n=n_atm):
    """
    Compute the covariance matrix of theta and phi given predictions and antenna data.
    (from eq.)

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
    B = np.array([
        [-np.cos(theta_pred)*np.cos(phi_pred), np.sin(theta_pred)*np.sin(phi_pred)],
        [-np.cos(theta_pred)*np.sin(phi_pred), -np.sin(theta_pred)*np.cos(phi_pred)],
        [np.sin(theta_pred), 0]
    ])
    Xants_cor = (Xants - Xants.mean(axis=0)[None, :]) / (c / np.array(n).reshape(-1, 1))

    return np.linalg.pinv(B.T @ Xants_cor.T @ Xants_cor @ B) * (sigma**2)


def cov_matrix(theta_pred, phi_pred, Xants, sigma, c=c_light, n=n_atm):
    """
    Wrapper for Covariance_schurcomplement function.

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
    return Covariance_schurcomplement(theta_pred, phi_pred, Xants, sigma, c=c, n=n)

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


###Index of refraction

def ZHSEffectiveRefractionIndex(X0, Xa):
    """Compute mean refraction index along trajectory for antenna at Xa and source at X0"""
    R02 = X0[0]**2 + X0[1]**2

    # Altitude of emission in km
    h0 = (np.sqrt((X0[2]+R_earth)**2 + R02) - R_earth)/1e3
    # print('Altitude of emission in km = ',h0)
    # print(h0)

    # Refractivity at emission
    rh0 = ns*np.exp(kr*h0)

    modr = np.sqrt(R02)
    # print(modr)

    if (modr > 1e3):

        # Vector between antenna and emission point
        U = Xa-X0
        # Divide into pieces shorter than 10km
        # nint = np.int(modr/2e4)+1
        nint = int(modr/2e4)+1
        K = U/nint

        # Current point coordinates and altitude
        Curr = X0
        currh = h0
        s = 0.

        for i in np.arange(nint):
            Next = Curr + K  # Next point
            nextR2 = Next[0]*Next[0] + Next[1]*Next[1]
            nexth = (np.sqrt((Next[2]+R_earth)**2 + nextR2) - R_earth)/1e3
            if (np.abs(nexth-currh) > 1e-10):
                s += (np.exp(kr*nexth)-np.exp(kr*currh))/(kr*(nexth-currh))
            else:
                s += np.exp(kr*currh)

            Curr = Next
            currh = nexth
            # print (currh)

        avn = ns*s/nint
        # print(avn)
        n_eff = 1. + 1e-6*avn  # Effective (average) index

    else:

        # without numerical integration
        hd = Xa[2]/1e3  # Antenna altitude
        # if (np.abs(hd-h0) > 1e-10):
        avn = (ns/(kr*(hd-h0)))*(np.exp(kr*hd)-np.exp(kr*h0))
        # else:
        #    avn = ns*np.exp(kr*h0)

        n_eff = 1. + 1e-6*avn  # Effective (average) index

    return (n_eff)


def ZHSEffectiveRefractionIndexvect(X0, Xa):
    """Same as previous but with a vector of antenna position Xa (N*3)"""
    R02 = X0[0]**2 + X0[1]**2

    # Altitude of emission in km
    h0 = (np.sqrt((X0[2]+R_earth)**2 + R02) - R_earth)/1e3
    # print('Altitude of emission in km = ',h0)
    # print(h0)

    # Refractivity at emission
    rh0 = ns*np.exp(kr*h0)

    modr = np.sqrt(R02)
    # print(modr)

    if (modr > 1e3):

        # Vector between antenna and emission point
        U = Xa-X0
        # Divide into pieces shorter than 10km
        # nint = np.int(modr/2e4)+1
        nint = int(modr/2e4)+1
        K = U/nint

        # Current point coordinates and altitude
        Curr = X0
        currh = h0
        s = np.zeros(Xa.shape[0])

        for i in np.arange(nint):
            Next = Curr + K  # Next point
            nextR2 = Next[:, 0]*Next[:, 0] + Next[:, 1]*Next[:, 1]
            nexth = (np.sqrt((Next[:, 2]+R_earth)**2 + nextR2) - R_earth)/1e3
            mask = np.abs(nexth-currh) > 1e-10
            s[mask] += (np.exp(kr*nexth[mask])-np.exp(kr*currh)) / \
                (kr*(nexth[mask]-currh))
            s[1-mask] += np.exp(kr*currh)

            Curr = Next
            currh = nexth
            # print (currh)

        avn = ns*s/nint
        # print(avn)
        n_eff = 1. + 1e-6*avn  # Effective (average) index

    else:

        # without numerical integration
        hd = Xa[:, 2]/1e3  # Antenna altitude
        # if (np.abs(hd-h0) > 1e-10):
        avn = (ns/(kr*(hd-h0)))*(np.exp(kr*hd)-np.exp(kr*h0))
        # else:
        #    avn = ns*np.exp(kr*h0)

        n_eff = 1. + 1e-6*avn  # Effective (average) index

    return (n_eff)
