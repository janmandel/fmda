
from diagnostics import diagnostics
import numpy as np


def numerical_solve_bisect(e2, eps2, k):
    """
    Solve the estimator equation using bisection.
    See Vejmelka et al, IAWF, 2013 for details.
    """
    N = e2.shape[0]
    tgt = N - k
    s2_eta_left = 0.0
    s2_eta_right = 0.1

    val_left = np.sum(e2 / eps2)
    val_right = np.sum(e2 / (eps2 + s2_eta_right))
    # if with th eminimum possible s2_eta (which is 0), we are below target
    # then a solution does not exist
    if val_left < tgt:
        return -1.0

    while val_right > tgt:
      s2_eta_right *= 2.0
      val_right = np.sum(e2 / (eps2 + s2_eta_right))

    while val_left - val_right > 1e-6:
        s2_eta_new = 0.5 * (s2_eta_left + s2_eta_right)
        val = np.sum(e2 / (eps2 + s2_eta_new))

        if val > tgt:
            val_left, s2_eta_left = val, s2_eta_new
        else:
            val_right, s2_eta_right = val, s2_eta_new

    return 0.5 * (s2_eta_left + s2_eta_right)



def trend_surface_model_kriging(obs_data, X, K, V):
    """
    Trend surface model kriging, which assumes spatially uncorrelated errors.
    The kriging results in the matrix K, which contains the kriged observations
    and the matrix V, which contains the kriging variance.
    """
    Nobs = len(obs_data)
    # we ensure we have at most Nobs covariates
    Nallcov = min(X.shape[2], Nobs)

    print('DEBUG: Nobs = %d Nallcov = %d' % (Nobs, Nallcov))
    # the matrix of covariates
    Xobs_arr = np.zeros((Nobs, Nallcov))
    Xobs = np.asmatrix(Xobs_arr)

    # the vector of target observations
    y = np.zeros((Nobs,1))

    # the vector of observation variances
    obs_var = np.zeros((Nobs,))

    # fill out matrix/vector structures
    for (obs,i) in zip(obs_data, range(Nobs)):
        p = obs.get_nearest_grid_point()
        y[i,0] = obs.get_value()
        Xobs[i,:] = X[p[0], p[1], 0:Nallcov]
        obs_var[i] = obs.get_measurement_variance()

    # remove covariates that contain only zeros
    nz_covs = np.nonzero([np.sum(Xobs_arr[:,i]**2) for i in range(Nallcov)])[0]
    Ncov = len(nz_covs)
    print('DEBUG: nz_covs = %s X.shape = %s Xobs.shape = %s' % (str(nz_covs), str(X.shape), str(Xobs.shape)))
    X = X[:,:,nz_covs]
    Xobs = Xobs[:,nz_covs]

    # initialize the iterative algorithm
    s2_eta_hat_old = 10.0
    s2_eta_hat = 0.0
    iters = 0
    subzeros = 0

    # while the relative change
    while abs( (s2_eta_hat_old - s2_eta_hat) / max(s2_eta_hat_old, 1e-8)) > 1e-2:

        s2_eta_hat_old = s2_eta_hat

        # recompute covariance matrix
        Sigma = np.diag(obs_var + s2_eta_hat)
        XtSX = Xobs.T * np.linalg.solve(Sigma, Xobs)

        # QR solution method of the least squares problem
        Sigma_1_2 = np.asmatrix(np.diag(np.diag(Sigma)**-0.5))
        yt = Sigma_1_2 * y
        Q, R = np.linalg.qr(Sigma_1_2 * Xobs)
        beta = np.linalg.solve(R, np.asmatrix(Q).T * yt)
        res2 = np.asarray(y - Xobs * beta)[:,0]**2

        # compute new estimate of variance of microscale variability
        print('DEBUG: res2.shape = %s obs_var.shape = %s' % (str(res2.shape), str(obs_var.shape)))
        s2_array = res2 - obs_var
        print('DEBUG: XtSX.shape = %s s2_array.shape = %s' % (str(XtSX.shape), str(s2_array.shape)))
        s2_array += np.diag(Xobs * np.linalg.solve(XtSX, Xobs.T))

        s2_eta_hat = numerical_solve_bisect(res2, obs_var, Ncov)
        if s2_eta_hat < 0.0:
          print("TSM: s2_eta_hat estimate below zero")
          s2_eta_hat = 0.0

        subzeros = np.count_nonzero(s2_array < 0.0)
        iters += 1

    print('DEBUG: iters = %d subzeros = %d' % (iters, subzeros))

    # map computed betas to original (possibly extended) betas which include zero variables
    beta_ext = np.asmatrix(np.zeros((Nallcov,1)))
    beta_ext[nz_covs] = beta
    diagnostics().push("s2_eta_hat", s2_eta_hat)
    diagnostics().push("kriging_beta", beta_ext)
    diagnostics().push("kriging_iters", iters)
    diagnostics().push("kriging_subzero_s2_estimates", subzeros)
    diagnostics().push("res2_sum", np.sum(res2))

    for i in range(X.shape[0]):
#        x_i = X[i,:,:]
#        K[i,:] = np.asarray(np.dot(x_i, beta))[:,0]
#        V[i,:] = s2_eta_hat + np.diag(np.dot(x_i, np.linalg.solve(XtSX, x_i.T)))

      for j in range(X.shape[1]):
        x_ij = X[i,j,:]
        K[i,j] = np.dot(x_ij, beta)
        V[i,j] = s2_eta_hat + np.dot(x_ij, np.linalg.solve(XtSX, x_ij))

