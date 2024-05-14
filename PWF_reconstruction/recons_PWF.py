import numpy as np
import time
from scipy.optimize import brentq
import scipy.optimize as so
import numdifftools as nd

# Defining constants
R2D = 180/np.pi
# Physical constants
c_light = 2.997924580e8
#Constant for variable index
R_earth = 6371007.0
ns = 325
kr = -0.1218
groundAltitude = 1086.0
shower_core = np.array([0,0,groundAltitude])


def ZHSEffectiveRefractionIndex(X0,Xa):
    """Compute mean refraction index along trajectory for antenna at Xa and source at X0"""
    R02 = X0[0]**2 + X0[1]**2

    # Altitude of emission in km
    h0 = (np.sqrt( (X0[2]+R_earth)**2 + R02 ) - R_earth)/1e3
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
        #nint = np.int(modr/2e4)+1
        nint = int(modr/2e4)+1
        K = U/nint

        # Current point coordinates and altitude
        Curr  = X0
        currh = h0
        s = 0.

        for i in np.arange(nint):
            Next = Curr + K # Next point
            nextR2 = Next[0]*Next[0] + Next[1]*Next[1]
            nexth  = (np.sqrt( (Next[2]+R_earth)**2 + nextR2 ) - R_earth)/1e3
            if (np.abs(nexth-currh) > 1e-10):
                s += (np.exp(kr*nexth)-np.exp(kr*currh))/(kr*(nexth-currh))
            else:
                s += np.exp(kr*currh)

            Curr = Next
            currh = nexth
            # print (currh)

        avn = ns*s/nint
        # print(avn)
        n_eff = 1. + 1e-6*avn # Effective (average) index

    else:

        # without numerical integration
        hd = Xa[2]/1e3 # Antenna altitude
        #if (np.abs(hd-h0) > 1e-10):
        avn = (ns/(kr*(hd-h0)))*(np.exp(kr*hd)-np.exp(kr*h0))
        #else:
        #    avn = ns*np.exp(kr*h0)

        n_eff = 1. + 1e-6*avn # Effective (average) index

    return (n_eff)

def ZHSEffectiveRefractionIndexvect(X0,Xa):
    """Same as previous but with a vector of antenna position Xa (N*3)"""
    R02 = X0[0]**2 + X0[1]**2

    # Altitude of emission in km
    h0 = (np.sqrt( (X0[2]+R_earth)**2 + R02 ) - R_earth)/1e3
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
        #nint = np.int(modr/2e4)+1
        nint = int(modr/2e4)+1
        K = U/nint

        # Current point coordinates and altitude
        Curr  = X0
        currh = h0
        s = np.zeros(Xa.shape[0])

        for i in np.arange(nint):
            Next = Curr + K # Next point
            nextR2 = Next[:,0]*Next[:,0] + Next[:,1]*Next[:,1]
            nexth  = (np.sqrt( (Next[:,2]+R_earth)**2 + nextR2 ) - R_earth)/1e3
            mask = np.abs(nexth-currh) > 1e-10
            s[mask] += (np.exp(kr*nexth[mask])-np.exp(kr*currh))/(kr*(nexth[mask]-currh))
            s[1-mask] += np.exp(kr*currh)

            Curr = Next
            currh = nexth
            # print (currh)

        avn = ns*s/nint
        # print(avn)
        n_eff = 1. + 1e-6*avn # Effective (average) index

    else:

        # without numerical integration
        hd = Xa[:,2]/1e3 # Antenna altitude
        #if (np.abs(hd-h0) > 1e-10):
        avn = (ns/(kr*(hd-h0)))*(np.exp(kr*hd)-np.exp(kr*h0))
        #else:
        #    avn = ns*np.exp(kr*h0)

        n_eff = 1. + 1e-6*avn # Effective (average) index

    return (n_eff)

def Linear_solver(Xants, tants, cr=1., c=1.):
    """
    Given Xants in meters, tants in seconds cr the indices of refraction (vector or constant) 
    and c the speed of light (should be 3e8 but can be 1 if tants in meters)
    Computes the unconstraint best fit for k
    """
    t_ants = (c/cr*(tants-tants.mean()))[:,None]
    P_1 = (Xants-Xants.mean(axis=0))

    pseudoinverse = np.linalg.inv(P_1.T @ P_1)
    M = pseudoinverse @ P_1.T
    res = M@t_ants
    return res.flatten(), pseudoinverse  

def _projector(k, Inv):
    """
    Given a vector k and the inverse of covariance matrix. Compute projection on unit sphere along large axis of distribution.
    """
    d, R = np.linalg.eigh(Inv)
    # R = R[:,::-1]
    directions = R[:,-1]@np.array([[0.],[0],[1]])
    R[:,-1] *= np.sign(directions).T
    
    k_lin_rot = R.T @ k
    n2 = np.linalg.norm(k_lin_rot[:2])
    k_opt_rot =np.array([*(k_lin_rot[:2]/max(1,n2)), - np.sqrt(1 - min(1, n2**2))])

    k_opt = R @ k_opt_rot
    return k_opt
def PWF_projection(Xants, tants, verbose=False, cr=1.0, c=1.):
    '''
    Given Xants in meters, tants in seconds cr the indices of refraction (vector or constant) 
    and c the speed of light (should be 3e8 but can be 1 if tants in meters)
    Computes the projection of k on the unit sphere along large axis of distribution. Approx C. Best approximation.
    Returns theta and phi
    '''

    k_lin, Inv = Linear_solver(Xants, tants, cr=cr, c=c)
    k_opt = _projector(k_lin, Inv)
    # Now get angles from k_opt coordinates

    theta_opt = np.arccos(-k_opt[2])
    phi_opt = np.arctan2(-k_opt[1],-k_opt[0])
    return theta_opt, phi_opt


def PWF_semianalytical(Xants, tants, verbose=False, cr=1.0, c=1.):
    '''
    Given Xants in meters, tants in seconds cr the indices of refraction (vector or constant) 
    and c the speed of light (should be 3e8 but can be 1 if tants in meters)

    Solves the minimization problem by using a special solution to the linear regression
    on K(\theta,\phi), with the ||K||=1 constraint. Note that this is a non-convex problem.
    This is formulated as 
    argmin_k k^T.A.k - 2 b^T.k, s.t. ||k||=1
    '''
    nants = tants.shape[0]
    # Make sure tants and Xants are compatible

    if (Xants.shape[0] != nants):
        print("Shapes of tants and Xants are incompatible",tants.shape,Xants.shape)
        return None
    ## Compute A matrix (3x3) and b (3-)vector, see above
    tants_cor = tants*c/cr
    PXT = Xants - Xants.mean(axis=0) # P is the centering projector, XT=Xants
    A = np.dot(Xants.T,PXT)
    b = np.dot(Xants.T,tants_cor-tants_cor.mean(axis=0))
    # Diagonalize A, compute projections of b onto eigenvectors
    d,W = np.linalg.eigh(A)
    beta = np.dot(b,W)
    nbeta = np.linalg.norm(beta)

    if (np.abs(beta[0]/nbeta) < 1e-14):
        if (verbose):
            print ("Degenerate case")
        # Degenerate case. This will be triggered e.g. when all antennas lie in a single plane.
        mu = -d[0]
        c = np.zeros(3)
        c[1] = beta[1]/(d[1]+mu)
        c[2] = beta[2]/(d[2]+mu)
        si = np.sign(np.dot(W[:,0],np.array([0,0,1.])))
        c[0] = -si*np.sqrt(1-c[1]**2-c[2]**2) # Determined up to a sign: choose descending solution
        k_opt = np.dot(W,c)
        # k_opt[2] = -np.abs(k_opt[2]) # Descending solution
    
    else:
        # Assume non-degenerate case, i.e. projections on smallest eigenvalue are non zero
        # Compute \mu such that \sum_i \beta_i^2/(\lambda_i+\mu)^2 = 1, using root finding on mu
        def nc(mu):
            # Computes difference of norm of k solution to 1. Coordinates of k are \beta_i/(d_i+\mu) in W basis
            c = beta/(d+mu)
            return ((c**2).sum()-1.)
        mu_min = -d[0]+beta[0]
        mu_max = -d[0]+np.linalg.norm(beta)
        mu_opt = brentq(nc,mu_min,mu_max,maxiter=1000)
        # Compute coordinates of k in W basis, return k
        c = beta/(d+mu_opt)
        k_opt = np.dot(W,c)
        
    # Now get angles from k_opt coordinates
    if k_opt[2]>1e-2:
        k_opt = k_opt-2*(k_opt@W[:,0])*W[:,0]

    theta_opt = np.arccos(-k_opt[2])
    phi_opt = np.arctan2(-k_opt[1],-k_opt[0])

    if phi_opt<0:
        phi_opt += 2*np.pi
    return(np.array([theta_opt,phi_opt]))

def Covariance_tangentplane(theta_pred, phi_pred, Xants, sigma, c=1., cr=1.):
    """
    Given a prediction Xants in meters, cr the indices of refraction (vector or constant) 
    and c the speed of light (should be 3e8 but can be 1 if tants in meters)

    Compute the covariance matrix of theta and phi
    Method described in paper
    """
    Xants_cor = (Xants-Xants.mean(axis=0)[None,:])/c*np.array(cr).reshape(-1,1)
    Sigma = (sigma)**2*np.linalg.inv(Xants_cor.T @ Xants_cor)

    Q = np.linalg.inv(np.array([[-np.sin(theta_pred)*np.cos(phi_pred), -np.cos(theta_pred)*np.cos(phi_pred), np.sin(theta_pred)*np.sin(phi_pred)  ],
                  [-np.sin(theta_pred)*np.sin(phi_pred), -np.cos(theta_pred)*np.sin(phi_pred), -np.sin(theta_pred)*np.cos(phi_pred) ],
                  [-np.cos(theta_pred)                 ,            np.sin(theta_pred)       ,         0         ]]))
    QSigQt = Q @ Sigma @ Q.T

    Sigma_aa = QSigQt[1:,1:]
    Sigma_ar = QSigQt[0,1:]
    Sigma_rr = QSigQt[0,0]
    Sigma_bar = Sigma_aa - 1/Sigma_rr * Sigma_ar[:,None] @ Sigma_ar[None,:]
    return Sigma_bar

def fisher_Variance(theta_pred, phi_pred, Xants, sigma, c=1., cr=1.):
    """
    Given Xants in meters, tants in seconds cr the indices of refraction (vector or constant) 
    and c the speed of light (should be 3e8 but can be 1 if tants in meters)

    Same as previous but with fisher matrix approach
    """
    B = np.array([
        [-np.cos(theta_pred)*np.cos(phi_pred), np.sin(theta_pred)*np.sin(phi_pred)],
        [-np.cos(theta_pred)*np.sin(phi_pred), -np.sin(theta_pred)*np.cos(phi_pred)],
        [           np.sin(theta_pred)       ,                 0                   ]
    ])
    Xants_cor = (Xants-Xants.mean(axis=0)[None,:])/c*np.array(cr).reshape(-1,1)

    return np.linalg.inv(B.T @ Xants_cor.T @ Xants_cor @ B)*(sigma**2)

##Gradient descent
def cov_matrix(theta_pred, phi_pred, Xants, sigma, c=1., cr=1.):
    return fisher_Variance(theta_pred, phi_pred, Xants, sigma, c=1., cr=1.)

def PWF_loss(params, Xants, tants, verbose=False, cr=1.0):
    '''
    Defines Chi2 by summing model residuals
    over antenna pairs (i,j):
    loss = \sum_{i>j} ((Xants[i,:]-Xants[j,:]).K - cr(tants[i]-tants[j]))**2
    where:
    params=(theta, phi): spherical coordinates of unit shower direction vector K
    Xants are the antenna positions (shape=(nants,3))
    tants are the antenna arrival times of the wavefront (trigger time, shape=(nants,))
    cr is radiation speed, by default 1 since time is expressed in m.
    '''

    theta,phi = params
    nants = tants.shape[0]
    ct = np.cos(theta); st = np.sin(theta); cp = np.cos(phi); sp = np.sin(phi)
    K = np.array([st*cp,st*sp,ct])
    # Make sure tants and Xants are compatible
    if (Xants.shape[0] != nants):
        print("Shapes of tants and Xants are incompatible",tants.shape,Xants.shape)
        return None
    # Use numpy outer methods to build matrix X_ij = x_i -x_j
    xk = np.dot(Xants,K)
    DXK = np.subtract.outer(xk,xk)
    DT  = np.subtract.outer(tants,tants)
    chi2 = ( (DXK - cr*DT)**2 ).sum() / 2. # Sum over upper triangle, diagonal is zero because of antisymmetry of DXK, DT
    if verbose:
        print("params = ",np.rad2deg(params))
        print("Chi2 = ",chi2)
    return(chi2)



def PWF_gradient(Xants, tants, cr=1.0, c=1., compute_errors=False):
    def no_periodicity(theta):
        return theta%(2*np.pi)

    def bound_phi(phi):
        return phi%2*np.pi
    
    bounds = ((np.pi/2+1e-7,np.pi),(0,2*np.pi))
    params_in = np.array(bounds).mean(axis=1)

    args = (Xants, tants*c/cr)
    res = so.minimize(PWF_loss,params_in,args=args,method='BFGS')
    res = so.minimize(PWF_loss,params_in, args=args, method='L-BFGS-B', bounds=bounds)

    res = so.minimize(PWF_loss, params_in, args=args, bounds=bounds, method='BFGS')
    # res = so.minimize(PWF_loss,params_in, args=(*params_in,1,True),method='L-BFGS-B', bounds=bounds)
    
    params_out = res.x
    # compute errors with numerical estimate of Hessian matrix, inversion and sqrt of diagonal terms
    if (compute_errors):
        hess = nd.Hessian(PWF_loss) (params_out,*args)
        errors = np.sqrt(np.diag(np.linalg.inv(hess)))
        print ("Best fit parameters = ",np.rad2deg(params_out))

        # Errors computation needs work: errors are coming both from noise on amplitude and time measurements
        print ("Errors on parameters (from Hessian) = ",np.rad2deg(errors))
        print ("Chi2 at best fit = ",PWF_loss(params_out,*args))
    # params_out[0] = np.pi - params_out[0]
    # params_out[1] += np.pi
    params_out = params_out%(2*np.pi)
    if params_out[0]>np.pi:
        params_out[0] = 2*np.pi-params_out[0]
        params_out[1] = (params_out[1] + np.pi)%(2*np.pi)

    if params_out[0]<np.pi/2:
        params_in = params_out
        params_out[0] = np.pi-params_out[0]
        res = so.minimize(PWF_loss, params_out, args=args)

        params_out = res.x%(2*np.pi)
        if params_out[0]>np.pi:
            params_out[0] = 2*np.pi-params_out[0]
            params_out[1] = (params_out[1] + np.pi)%(2*np.pi)
    return params_out

