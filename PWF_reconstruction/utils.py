import numpy as np
import pandas as pd

# Defining constants
R2D = 180/np.pi
# Physical constants
c_light = 2.997924580e8
# Constant for variable index
R_earth = 6371007.0
ns = 325
kr = -0.1218
groundAltitude = 1086.0
shower_core = np.array([0, 0, groundAltitude])


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


if __name__ == '__main__':
    theta, phi = np.array((0, np.pi/4, np.pi/2, np.pi/1.5)
                          ), np.array((0, np.pi/2, np.pi, np.pi*1.5))
    print(sph2cart(theta, phi))


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
    print(len(total_df))
    return total_df
