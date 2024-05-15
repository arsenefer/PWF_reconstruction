import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches 

from scipy.stats import norm

from .utils import R2D
from .utils import sph2cart
from .recons_PWF import cov_matrix, fisher_Variance, Covariance_tangentplane
def NLL(theta_pred, phi_pred, theta, phi, estimate_cova):
    mu = np.array((theta_pred, phi_pred))
    x = np.array((theta, phi))
    return (x-mu) @ np.linalg.inv(estimate_cova) @ (x-mu)

def _NLLmap(theta_range, phi_range, theta, phi, estimate_covar, n=100):
    thetas,phis = np.mgrid[theta_range[0]:theta_range[1]:n*1j, phi_range[0]:phi_range[1]:n*1j]
    pos = np.dstack((thetas, phis))
    mu = np.array([theta, phi])
    ellispoids = lambda pos : np.einsum("hkj,hkj->hk",
        np.einsum("hkj, ji -> hki", pos-mu[None,None,:], np.linalg.inv(estimate_covar)),
        pos-mu[None,None,:])

    pdf = ellispoids(pos)
    return thetas, phis, pdf

def plot_distribs(theta_dis:np.ndarray, 
                  phi_dis:np.ndarray, 
                  estimate_covar:np.ndarray, 
                  theta:float, 
                  phi:float, 
                  theta_pred:float, 
                  phi_pred:float):
    #Plotting histograms of predicted values
    distrib = pd.DataFrame({"phi":phi_dis*R2D, "theta":theta_dis*R2D})

    half_size = 4*np.sqrt(np.diag(estimate_covar).max())*R2D
    theta_range = ((theta*R2D-half_size), (theta*R2D+half_size))
    phi_range = ((phi*R2D-half_size), (phi*R2D+half_size))

    g = sns.JointGrid(distrib, x='phi', y='theta', xlim=phi_range, ylim=theta_range, height=9)

    hist_col = 'steelblue'
    g.plot_marginals(sns.histplot, color=hist_col)
    g.plot_joint(sns.histplot, color=hist_col)

    #Plotting estimated uncertainty
    x00 = np.linspace(*phi_range, 100)
    y00 = norm.pdf(x00, loc=phi_pred*R2D, scale = np.sqrt(estimate_covar[1,1])*R2D)
    y00 = y00/y00.max()*g.ax_marg_x.get_ylim()[1]

    g.ax_marg_x.plot(x00, y00, c="r")
    g.ax_marg_x.plot([phi*R2D, phi*R2D], [y00.min(), y00.max()], c='b', label='True $\\phi$')


    x11 = np.linspace(*theta_range, 100)
    y11 = norm.pdf(x11, loc=theta_pred*R2D, scale = np.sqrt(estimate_covar[0,0])*R2D)
    y11 = y11/y11.max()*g.ax_marg_y.get_xlim()[1]

    g.ax_marg_y.plot(y11, x11, c="r")
    g.ax_marg_y.plot([y11.min(), y11.max()], [theta*R2D,theta*R2D] , c='b', label='True $\\theta$')

    thetas, phis ,pdf = _NLLmap(theta_range, phi_range, theta_pred*R2D, phi_pred*R2D, estimate_covar*R2D**2)
    contour_values = np.array([-2*np.log(1-0.68),-2*np.log(1-0.95),-2*np.log(1-0.997)])
    # contour = np.sqrt(contour)
    CS = g.ax_joint.contour(phis, thetas, pdf, levels=contour_values,colors='r')
    sp = g.ax_joint.scatter(phi*R2D, theta*R2D, c='b', marker='x')

    strs = ['1 $\\sigma$', '2 $\\sigma$', '3 $\\sigma$']
    fmt = {l:s for l, s in zip(CS.levels, strs)}
    plt.clabel(CS, CS.levels[::], inline=True, fmt=fmt, fontsize=10)
    h, l = CS.legend_elements()

    blue_patch = mpatches.Patch(color=hist_col) 
    g.ax_joint.legend([blue_patch, sp, h[0]], ['Error distribution',"True arrival direction", 'Estimated distribution'])

    g.figure.suptitle(f"Estimated uncertaity distribution vs error distribution for PWF with $\\theta={theta:.1f}$°, $\\phi={phi:.1f}$°")
    # g.ax_joint.collections[0].set_alpha(0)
    g.figure.subplots_adjust(top=0.95)

def _is_in_interval(theta_pred, phi_pred, theta, phi, estimate_cova, contour):
    likelyhood = NLL(theta_pred, phi_pred, theta, phi, estimate_cova)
    return int((likelyhood< contour))

def PICP(total_df:pd.DataFrame, method, c_light=1):
    contours = np.array([-2*np.log(1-0.68),-2*np.log(1-0.95),-2*np.log(1-0.997)])

    for name, event_df in total_df.groupby('event_name'):
        X_ants = event_df[["x_ant", "y_ant", "z_ant"]].values
        Xmax_coord = event_df[["x_Xmax","y_Xmax","z_Xmax"]].values[0]

        theta,phi = event_df[["zenith","azimuth"]].values[0]/R2D

        k = -sph2cart(theta, phi)

        T_ants = event_df["time"].values
        try:
            sigma = event_df.sigma.values[0]
        except AttributeError:
            sigma = 1e-8
        theta_pred, phi_pred = method(X_ants, T_ants, c=c_light)
        estimate_cova = cov_matrix(theta_pred, phi_pred, X_ants, sigma, c=c_light)
        total_df.loc[total_df.event_name == name,'68'] = _is_in_interval(theta_pred, phi_pred, theta,phi, estimate_cova, contours[0])
        total_df.loc[total_df.event_name == name,'95'] = _is_in_interval(theta_pred, phi_pred, theta,phi, estimate_cova, contours[1])
        total_df.loc[total_df.event_name == name,'99'] = _is_in_interval(theta_pred, phi_pred, theta,phi, estimate_cova, contours[2])

    res_df = total_df.groupby("event_name", as_index=False).first()
    print(res_df.groupby("zenith", as_index=False).sum())
    picp_groups = res_df.groupby("zenith")[['68', '95', '99']]
    picp = picp_groups.mean()

    count = picp_groups.count()
    std68 = np.sqrt(0.68*(1-0.68)/count['68'])
    std95 = np.sqrt(0.95*(1-0.95)/count['95'])
    std99 = np.sqrt(0.995*(1-0.995)/count['99'])

    plt.scatter(picp.index, picp['68']*100, label="PICP 68% estimation", c='tab:blue')
    plt.fill_between(count['68'].index, 100*(0.68-2*std68), 100*(0.68+2*std68), alpha=.4, color='tab:blue')
    plt.plot([picp.index.min(), picp.index.max()], [68,68], 'k:')
    
    plt.scatter(picp.index, picp['95']*100, label="PICP 95% estimation", c='tab:green')
    plt.fill_between(count['95'].index, 100*(0.95-2*std95), 100*(0.95+2*std95), alpha=.4, color='tab:green')
    plt.plot([picp.index.min(), picp.index.max()], [95,95], 'k:')

    plt.scatter(picp.index, picp['99']*100, label="PICP 99.5% estimation", c='tab:purple')
    plt.fill_between(count['99'].index, 100*(0.995-2*std99), 100*(0.995+2*std99), alpha=.4, color="tab:purple")
    plt.plot([picp.index.min(), picp.index.max()], [99.5,99.5], 'k:')


    plt.gca().set_ylim(0,105)
    # plt.gca().set_yscale("log")
    plt.gca().set_xlabel('Zenith angle in °')
    plt.gca().set_ylabel('PICP value in %')
    plt.title("PICP diagram")
    plt.legend()

