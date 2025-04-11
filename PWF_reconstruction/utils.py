# Code from Ferriere et al. (2024) : https://doi.org/10.48550/arXiv.2408.15677

import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve

# Defining constants
R2D = 180/np.pi
# Physical constants
c_light = 2.997924580e8
n_atm = 1.000136


def _inv_cho(A):
    c, low = cho_factor(A)
    A_inv = cho_solve((c, low), np.eye(A.shape[0]))
    return A_inv

def mean(X:np.ndarray, sigma=None):
    if type(sigma) is np.ndarray and sigma.ndim==1:
        return ( 1/(1/sigma).sum() ) * ( (1/sigma) @ X )
    elif type(sigma) is np.ndarray and sigma.ndim==2:
        Q_1 = _inv_cho(sigma)
        return ( 1/Q_1.sum() ) * ( Q_1.sum(axis=0) @ X )
    else:
        return X.mean(axis=0)
    

def cart2sph(k):
    """
    Convert cartesian coordinate to spherical coordinate
    """
    r = np.linalg.norm(k, axis=1)
    tp = np.linalg.norm(k[:, :2], axis=1)
    theta = np.arctan2(tp, k[:, 2])
    phi = np.arctan2(k[:, 1], k[:, 0])
    return r, theta, phi


def sph2cart(theta, phi, r=1):
    """
    Convert spherical coordinate to cartesian coordinate
    """
    x = np.array(r*np.sin(theta)*np.cos(phi))
    y = np.array(r*np.sin(theta)*np.sin(phi))
    z = np.array(r*np.cos(theta))
    return np.concatenate((x[..., None], y[..., None], z[..., None]), axis=-1)


def opening_folders(cur):

    event_name = np.loadtxt(cur + 'input_simus.txt', usecols=[0])
    zenith = np.loadtxt(cur + 'input_simus.txt', usecols=[1])
    azimuth = np.loadtxt(cur + 'input_simus.txt', usecols=[2])
    energy = np.loadtxt(cur + 'input_simus.txt', usecols=[3])
    primary = np.loadtxt(cur + 'input_simus.txt', usecols=[4])
    Xmax_dist = np.loadtxt(cur + 'input_simus.txt', usecols=[5])
    slantXmax = np.loadtxt(cur + 'input_simus.txt', usecols=[6])
    x_Xmax = np.loadtxt(cur + 'input_simus.txt', usecols=[7])
    y_Xmax = np.loadtxt(cur + 'input_simus.txt', usecols=[8])
    z_Xmax = np.loadtxt(cur + 'input_simus.txt', usecols=[9])
    n_ants = np.loadtxt(cur + 'input_simus.txt', usecols=[10])
    energy_unit = np.loadtxt(cur + 'input_simus.txt', usecols=[11])

    df_input = pd.DataFrame({
        "event_name": event_name,
        "zenith": zenith,
        "azimuth": azimuth,
        "energy": energy,
        "primary": primary,
        "Xmax_dist": Xmax_dist,
        "slantXmax": slantXmax,
        "x_Xmax": x_Xmax,
        "y_Xmax": y_Xmax,
        "z_Xmax": z_Xmax,
        "n_ants": n_ants,
        "energy_unit": energy_unit
    })

    ant_ids = np.loadtxt(cur + 'Rec_coinctable.txt', usecols=[0])
    event_name = np.loadtxt(cur + 'Rec_coinctable.txt', usecols=[1])
    time = np.loadtxt(cur + 'Rec_coinctable.txt', usecols=[2])
    amplitude = np.loadtxt(cur + 'Rec_coinctable.txt', usecols=[3])
    df_timings = pd.DataFrame({
        "ant_ids": ant_ids,
        "event_name": event_name,
        "time": time,
        "amplitude": amplitude,
    })

    ant_ids = np.loadtxt(cur + 'coord_antennas.txt', usecols=[0])
    x_antenna = np.loadtxt(cur + 'coord_antennas.txt', usecols=[1])
    y_antenna = np.loadtxt(cur + 'coord_antennas.txt', usecols=[2])
    z_antenna = np.loadtxt(cur + 'coord_antennas.txt', usecols=[3])

    df_antennas = pd.DataFrame({
        "ant_ids": ant_ids,
        "x_ant": x_antenna,
        "y_ant": y_antenna,
        "z_ant": z_antenna,
    })

    total_df = pd.merge(df_input, df_timings, on='event_name', how='inner')
    total_df = pd.merge(total_df, df_antennas, on='ant_ids', how='inner')
    return total_df

def create_times(Xants:np.array, k, sigma=0, c=c_light, n=n_atm):
    """
    Producing antenna timings with Gaussian noise of scale $\sigma$.

    Parameters:
    Xants (ndarray): Antenna positions in meters, shape (nants, 3).
    k (ndarray): signal propagation direction, shape (3).
    sigma (float, np.ndarray): Standard deviation of arrival times.
    c (float): Speed of light in m/s, default is  299792458 m/s
    n (float or ndarray): Indices of refraction (vector or constant), default is 1.000136

    Returns:
    ndarray: Theta and phi angles in radians.
    """
    assert type(k) is np.ndarray
    P0 = mean(Xants, sigma)
    T = (Xants-P0) @ k / (c/n)
    T += np.random.normal(0, sigma, T.shape)
    return T

def chi2_PWF(t_meas, t_PWF, sigma=None):
    """
    Calculates the chi-squared value for a time reconstruction - in this case a PWF :)
    Args:
        t_meas: A NumPy array of measured (or simulated) times.
        t_PWF: A NumPy array of reconstructed times.
    Returns:
        The chi-squared value (float). 
    """
    if sigma is None:
        # sigma = np.std(t_meas-t_PWF, ddof=2)  #this is bad, chi2 will always be 1
        sigma = 1                               #Better nothing than something wrong
    time_diff = t_meas - t_PWF
    chi2_value = np.sum((time_diff / sigma)**2)
    return chi2_value

def chi2_PWF_n(t_meas, t_PWF, sigma=None):
    """
    Calculates the chi-squared value for a time reconstruction - in this case a PWF :)
    Args:
        t_meas: A NumPy array of measured (or simulated) times.
        t_PWF: A NumPy array of reconstructed times.
    Returns:
        The chi-squared value (float). 
    """
    if sigma is None:
        # sigma = np.std(t_meas-t_PWF, ddof=2)  #this is bad, chi2 will always be 1
        sigma = 1                               #Better nothing than something wrong
    time_diff = t_meas - t_PWF
    chi2_value = np.sum((time_diff / sigma)**2)/(len(t_meas)-2)
    return chi2_value
