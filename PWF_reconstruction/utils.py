import numpy as np
import pandas as pd

# Defining constants
R2D = 180/np.pi
# Physical constants
c_light = 2.997924580e8
n_atm = 1.000136


def cart2sph(k):
    r = np.linalg.norm(k, axis=1)
    tp = np.linalg.norm(k[:, :2], axis=1)
    theta = np.arctan2(tp, k[:, 2])
    phi = np.arctan2(k[:, 1], k[:, 0])
    return r, theta, phi


def sph2cart(theta, phi, r=1):
    x = np.array(r*np.sin(theta)*np.cos(phi))
    y = np.array(r*np.sin(theta)*np.sin(phi))
    z = np.array(r*np.cos(theta))
    return np.concatenate((x[..., None], y[..., None], z[..., None]), axis=-1)
