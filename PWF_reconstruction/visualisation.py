import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from scipy.stats import norm

from .utils import R2D
from .utils import sph2cart
from .recons_PWF import cov_matrix


def NLL(theta_pred, phi_pred, theta, phi, Sigma):
    """
    Compute the negative log-likelihood for the given parameters.
    The likelyhood of (theta, phi) given the mean (theta_pred, phi_pred) and covariance Sigma

    Parameters:
    theta_pred (float): Predicted theta angle in radians.
    phi_pred (float): Predicted phi angle in radians.
    theta (float): theta angle in radians to compute NLL.
    phi (float): phi angle in radians to compute NLL.
    Sigma (ndarray): Estimated covariance matrix, shape (2, 2).

    Returns:
    float: Negative log-likelihood value.
    """
    mu = np.array((theta_pred, phi_pred))
    x = np.array((theta, phi))
    return (x-mu) @ np.linalg.inv(Sigma) @ (x-mu)


def _NLLmap(theta_range, phi_range, theta, phi, estimate_covar, n=100):
    """
    Generate a map of negative log-likelihood values over specified ranges.

    Parameters:
    theta_range (tuple): Range of theta values in radians (start, end).
    phi_range (tuple): Range of phi values in radians (start, end).
    theta (float): Actual theta angle in radians.
    phi (float): Actual phi angle in radians.
    estimate_covar (ndarray): Estimated covariance matrix, shape (2, 2).
    n (int): Number of points for the grid, default is 100.

    Returns:
    tuple: Meshgrid of theta values, meshgrid of phi values, NLL values grid.
    """
    thetas, phis = np.mgrid[theta_range[0]:theta_range[1]:n*1j, 
                            phi_range[0]:phi_range[1]:n*1j]
    pos = np.dstack((thetas, phis))
    mu = np.array([theta, phi])

    def ellispoids(pos): 
        return np.einsum("hkj,hkj->hk",
                         np.einsum("hkj, ji -> hki", pos-mu[None, None, :], 
                                   np.linalg.inv(estimate_covar)),
                         pos-mu[None, None, :])

    pdf = ellispoids(pos)
    return thetas, phis, pdf


def plot_distribs(theta_dis: np.ndarray,
                  phi_dis: np.ndarray,
                  estimate_covar: np.ndarray,
                  theta: float,
                  phi: float,
                  theta_pred: float,
                  phi_pred: float):
    """
    Plot distributions of predicted theta and phi values along with their uncertainties.
    If only one prediction as been made and no distribution is available, put np.array([]) for both
    Parameters:
    theta_dis (ndarray): Distribution of theta values in radians, shape (N,).
    phi_dis (ndarray): Distribution of phi values in radians, shape (N,).
    estimate_covar (ndarray): Estimated covariance matrix, shape (2, 2).
    theta (float): Actual theta value in radians.
    phi (float): Actual phi value in radians.
    theta_pred (float): Predicted theta value in radians.
    phi_pred (float): Predicted phi value in radians.

    Returns:
    None
    """
    distrib = pd.DataFrame({"phi": phi_dis*R2D, "theta": theta_dis*R2D})
    if not(type(estimate_covar) is type(None)):
        half_size = 4*np.sqrt(np.diag(estimate_covar).max())*R2D
    else:
        half_size = max(np.abs(theta_dis-theta).max(), np.abs(phi_dis-phi).max())*R2D

    theta_range = ((theta*R2D-half_size), (theta*R2D+half_size))
    phi_range = ((phi*R2D-half_size), (phi*R2D+half_size))

    g = sns.JointGrid(distrib, x='phi', y='theta',
                      xlim=phi_range, ylim=theta_range, height=9)

    hist_col = 'steelblue'
    g.plot_marginals(sns.histplot, color=hist_col)
    g.plot_joint(sns.histplot, color=hist_col)

    g.ax_joint.scatter(phi_pred*R2D, theta_pred*R2D, c='r', marker='x')
    sp = g.ax_joint.scatter(phi*R2D, theta*R2D, c='b', marker='x')
    if not(type(estimate_covar) is type(None) or type(theta_pred) is type(None) or type(phi_pred) is type(None)):
        x00 = np.linspace(*phi_range, 100)
        y00 = norm.pdf(x00, loc=phi_pred*R2D,
                       scale=np.sqrt(estimate_covar[1, 1])*R2D)
        y00 = y00/y00.max()*g.ax_marg_x.get_ylim()[1]

        g.ax_marg_x.plot(x00, y00, c="r")
        g.ax_marg_x.plot([phi*R2D, phi*R2D], [y00.min(), y00.max()],
                         c='b', label='True $\\phi$')

        x11 = np.linspace(*theta_range, 100)
        y11 = norm.pdf(x11, loc=theta_pred*R2D,
                       scale=np.sqrt(estimate_covar[0, 0])*R2D)
        y11 = y11/y11.max()*g.ax_marg_y.get_xlim()[1]

        g.ax_marg_y.plot(y11, x11, c="r")
        g.ax_marg_y.plot([y11.min(), y11.max()], 
                         [theta*R2D,theta*R2D], 
                         c='b', 
                         label='True $\\theta$')

        thetas, phis, pdf = _NLLmap(
            theta_range, phi_range, theta_pred*R2D, phi_pred*R2D, estimate_covar*R2D**2)
        contour_values = np.array(
            [-2*np.log(1-0.68), -2*np.log(1-0.95), -2*np.log(1-0.997)])
        CS = g.ax_joint.contour(
            phis, thetas, pdf, levels=contour_values, colors='r')

        strs = ['1 $\\sigma$', '2 $\\sigma$', '3 $\\sigma$']
        fmt = {l: s for l, s in zip(CS.levels, strs)}
        plt.clabel(CS, CS.levels[::], inline=True, fmt=fmt, fontsize=10)
        h, l = CS.legend_elements()

        blue_patch = mpatches.Patch(color=hist_col)
        g.ax_joint.legend([blue_patch, sp, h[0]], 
                          ['Error distribution', "True arrival direction", 'Estimated distribution'])
    else:
        strs = ['1 $\\sigma$', '2 $\\sigma$', '3 $\\sigma$']

        blue_patch = mpatches.Patch(color=hist_col)
        g.ax_joint.legend([blue_patch, sp], 
                          ['Error distribution', "True arrival direction"])

    g.figure.suptitle(
        f"Estimated uncertainty distribution vs error distribution for PWF with $\\theta={theta*R2D:.1f}$°, $\\phi={phi*R2D:.1f}$°")
    g.figure.subplots_adjust(top=0.95)