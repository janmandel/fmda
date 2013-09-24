# -*- coding: utf-8 -*-
"""
Created on Sun Oct 28 18:14:36 2012

@author: martin
"""

from kriging_methods import trend_surface_model_kriging
from wrf_model_data import WRFModelData
from cell_model_opt import CellMoistureModel
from observation_stations import MesoWestStation
from diagnostics import init_diagnostics, diagnostics

import numpy as np
import os
import sys
import string
from datetime import datetime
import pytz
import netCDF4


def total_seconds(tdelta):
    """
    Utility function for python < 2.7, 2.7 and above have total_seconds()
    as a function of timedelta.
    """
    return tdelta.microseconds / 1e6 + (tdelta.seconds + tdelta.days * 24 * 3600)


def build_observation_data(stations, obs_valid_nowype):
    """
    Repackage the matched time series into a time-indexed structure
    which gives details on the observed data and active observation stations.

        synopsis: obs_data = build_observation_data(stations, obs_valid_nowype)

    """
    Ns = len(stations)

    # accumulate all observations from stations
    observations = []
    for s in stations:
        observations.extend(s.get_observations(obs_valid_nowype))

    # repackage all the observations into a time-indexed structure which groups
    # observations at the same time together
    obs_data = {}
    for obs in observations:
        t = obs.get_time()
        o = obs_data[t] if obs_data.has_key(t) else []
        o.append(obs)
        obs_data[t] = o

    return obs_data


def parse_datetime(s):
    gmt_tz = pytz.timezone('GMT')
    dt = datetime.strptime(s, "%Y/%m/%d  %H:%M:%S")
    return dt.replace(tzinfo = gmt_tz)

def run_module():

    # read in configuration file to execute run
    print("Reading configuration from [%s]" % sys.argv[1])

    with open(sys.argv[1]) as f:
        cfg = eval(f.read())

    # ensure output path exists
    if not os.path.isdir(cfg['output_dir']): 
        os.mkdir(cfg['output_dir'])

    # configure diagnostics
    init_diagnostics(os.path.join(cfg['output_dir'], 'moisture_model_v1_diagnostics.txt'))

    # Error covariance matrix condition number in kriging
    diagnostics().configure_tag("kriging_cov_cond", True, True, True)
    diagnostics().configure_tag("s2_eta_hat", True, True, True)
    diagnostics().configure_tag("res2_mean", True, True, True)
    diagnostics().configure_tag("na_res2_mean", True, True, True)

    # Assimilation parameters
    diagnostics().configure_tag("assim_K1", True, True, True)
    diagnostics().configure_tag("assim_data", False, False, True)

    ### Load and preprocess WRF model data

    # load WRF data
    wrf_data = WRFModelData(cfg['wrf_output'],  ['T2', 'Q2', 'PSFC', 'RAINNC', 'RAINC', 'HGT'])
    wrf_data.slice_field('HGT')

    # re-open the WRF file for writing FMC_G variables
    wrf_file = netCDF4.Dataset(cfg['wrf_output'], 'a')
    fmc_gc = wrf_file.variables['FMC_GC']

    # read in spatial and temporal extent of WRF variables
    lat, lon = wrf_data.get_lats(), wrf_data.get_lons()
    hgt = wrf_data['HGT']
    tm = wrf_data.get_gmt_times()
    Nt = cfg['Nt'] if cfg.has_key('Nt') and cfg['Nt'] is not None else len(tm)
    dom_shape = lat.shape

    # determine simulation times
    tm_start = parse_datetime(cfg['start_time']) if cfg['start_time'] is not None else tm[0]
    tm_end = parse_datetime(cfg['end_time']) if cfg['end_time'] is not None else tm[-1]

    # if the required start time or end time are outside the simulation domain, exit with an error
    if tm_start < tm[0] or tm_end > tm[-1]:
        print('FATAL: invalid time range, required [%s-%s], availble [%s-%s]' %
              (str(tm_start), str(tm_end), str(tm[0]), str(tm[-1])))
        sys.exit(2)

    print('INFO: time limits are %s to %s\nINFO: simulation is from %s to %s' %
          (str(tm_start), str(tm_end), str(tm[0]), str(tm[-1])))

    # retrieve the rain variable
    rain = wrf_data['RAIN']
    T2 = wrf_data['T2']
    PSFC = wrf_data['PSFC']

    # moisture equilibria are now computed from averaged Q,P,T at beginning and end of period
    Ed, Ew = wrf_data.get_moisture_equilibria()

    # the first FMC_GC value gets filled out with equilibria
    fmc_gc[0, :Nk, :, :] = (Ed + Ew) / 2

    ### Load observation data from the stations

    # load station data from files
    with open(cfg['station_list_file'], 'r') as f:
        si_list = f.read().split('\n')

    si_list = filter(lambda x: len(x) > 0 and x[0] != '#', map(string.strip, si_list))

    # for each station id, load the station
    stations = []
    for code in si_list:
        mws = MesoWestStation(code)
        mws.load_station_info(os.path.join(cfg["station_data_dir"], "%s.info" % code))
        mws.register_to_grid(wrf_data)
        mws.load_station_data(os.path.join(cfg["station_data_dir"], "%s.obs" % code))
        stations.append(mws)

    print('Loaded %d stations.' % len(stations))

    # build the observation data
    obs_data_fm10 = build_observation_data(stations, 'FM')

    ### Initialize model and visualization

    # construct initial conditions from timestep 1 (because Ed/Ew at zero are zero)
    E = 0.5 * (Ed[1,:,:] + Ew[1,:,:])

    # set up parameters
    Nk = 4  # we simulate 4 types of fuel
    Q = np.diag(cfg['Q'])
    P0 = np.diag(cfg['P0'])
    Tk = np.array([1.0, 10.0, 100.0, 1000.0]) * 3600
    dt = (tm[1] - tm[0]).seconds
    print("INFO: Computed timestep from WRF is is %g seconds." % dt)
    mresV = np.zeros_like(E)
    Kf_fn = np.zeros_like(E)
    Vf_fn = np.zeros_like(E)
    mid = np.zeros_like(E)
    Kg = np.zeros((dom_shape[0], dom_shape[1], len(Tk)+2))

    # preprocess all static covariates
    cov_ids = cfg['covariates']
    Xd3 = len(cov_ids) + 1
    X = np.zeros((dom_shape[0], dom_shape[1], Xd3))
    Xr = np.zeros((dom_shape[0], dom_shape[1], Xd3))
    static_covar_map = { "lon" : lon, "lat" : lat, "elevation" : hgt, "constant" : np.ones(dom_shape) }
    dynamic_covar_map = { "temperature" : T2, "pressure" : PSFC, "rain" : rain }

    for i in range(1, Xd3):
        cov_id = cov_ids[i-1]
        if cov_id in static_covar_map:
          print('INFO: found static covariate %s' % cov_id)
          Xr[:, :, i] = static_covar_map[cov_id]
        elif cov_id in dynamic_covar_map:
          print('INFO: found dynamic covariate %s' % cov_id)
        else:
          print('FATAL: unknown covariate %s encountered' % cov_id)
          sys.exit(2)

    print("INFO: there are %d covariates (including model state)" % Xd3)

    # retrieve assimilation time window
    assim_time_win = cfg['assimilation_time_window']

    # construct model grid using standard fuel parameters
    models = np.zeros(dom_shape, dtype = np.object)
    models_na = np.zeros_like(models)
    for p in np.ndindex(dom_shape):
        models[p] = CellMoistureModel((lat[p], lon[p]), 4, E[p], Tk, P0 = P0)
        models_na[p] = CellMoistureModel((lat[p], lon[p]), 4, E[p], Tk, P0 = P0)

    ###  Run model for each WRF timestep and assimilate data when available
    t_start, t_end = 1, len(tm)-1
    while tm_start > tm[t_start]:
        t_start+=1
    while tm_end < tm[t_end]:
        t_end-=1

    print('INFO: running simulation from %s (%d) to %s (%d).' % (str(tm[t_start]), t_start, str(tm[t_end]), t_end))
    for t in range(t_start, t_end+1):
        model_time = tm[t]
        print("INFO: time: %s, step: %d" % (str(model_time), t))

        # run the model update
        for p in np.ndindex(dom_shape):
            i, j = p
            models[p].advance_model(Ed[t-1, i, j], Ew[t-1, i, j], rain[t-1, i, j], dt, Q)
            models_na[p].advance_model(Ed[t-1, i, j], Ew[t-1, i, j], rain[t-1, i, j], dt, Q)

        # prepare visualization data
        f = np.zeros((dom_shape[0], dom_shape[1], Nk))
        f_na = np.zeros((dom_shape[0], dom_shape[1], Nk))
        for p in np.ndindex(dom_shape):
            f[p[0], p[1], :] = models[p].get_state()[:Nk]
            f_na[p[0], p[1], :] = models_na[p].get_state()[:Nk]

        # run Kriging on each observed fuel type
        Kf = []
        Vf = []
        fn = []
        for obs_data, fuel_ndx in [ (obs_data_fm10, 1) ]:

            # run the kriging subsystem and the Kalman update only if have valid observations
            valid_times = [z for z in obs_data.keys() if abs(total_seconds(z - model_time)) < assim_time_win/2.0]
            print('INFO: there are %d valid times at model time %s' % (len(valid_times), str(model_time)))
            if len(valid_times) > 0:

                # retrieve observations for current time
                obs_valid_now = []
                for z in valid_times:
                    obs_valid_now.extend(obs_data[z])

                print('INFO: model time %s, assimilating %d observations.' % (str(model_time), len(obs_valid_now)))

                # construct covariates for this time instant 
                X[:,:,0] = f[:,:,fuel_ndx]
                for i in range(1, Xd3):
                  cov_id = cov_ids[i-1]
                  if cov_id in static_covar_map:
                    X[:, :, i] = Xr[:, :, i]
                  elif cov_id in dynamic_covar_map:
                    F = dynamic_covar_map[cov_id]
                    X[:, :, i] = F[t, :, :]
                  else:
                    error('FATAL: found unknown covariate %s' % cov_id)

                # find differences (residuals) between observed measurements and nearest grid points
                obs_vals = np.array([o.get_value() for o in obs_valid_now])
                mod_vals = np.array([f[:,:,fuel_ndx][o.get_nearest_grid_point()] for o in obs_valid_now])
                mod_na_vals = np.array([f_na[:,:,fuel_ndx][o.get_nearest_grid_point()] for o in obs_valid_now])
                diagnostics().push("na_res2_mean", np.mean((obs_vals - mod_na_vals)**2))

                # krige observations to grid points
                trend_surface_model_kriging(obs_valid_now, X, Kf_fn, Vf_fn)
                np.savetxt('V', Vf_fn)

                krig_vals = np.array([Kf_fn[o.get_nearest_grid_point()] for o in obs_valid_now])
                diagnostics().push("assim_data", (t, fuel_ndx, obs_vals, krig_vals, mod_vals, mod_na_vals))
                diagnostics().push("fm10_kriging_var", (t, np.mean(Vf_fn)))

                # append to storage for kriged fields in this time instant
                Kf.append(Kf_fn)
                Vf.append(Vf_fn)
                fn.append(fuel_ndx)


        # if there were any observations, run the kalman update step
        if len(fn) > 0:
            Nobs = len(fn)
            # run the kalman update in each model independently
            # gather the standard deviations of the moisture fuel after the Kalman update
            for p in np.ndindex(dom_shape):
                O = np.zeros((Nobs,))
                V = np.zeros((Nobs, Nobs))

                # construct observations for this position
                for i in range(Nobs):
                    O[i] = Kf[i][p]
                    V[i,i] = Vf[i][p]

                # execute the Kalman update
                Kp = models[p].kalman_update(O, V, fn)
                Kg[p[0], p[1], :] = Kp[:, 0]

            # push new diagnostic outputs
            diagnostics().push("assim_K1", (t, np.mean(Kg[:,:,1])))

        # store data in wrf_file variable FMC_G
        for p in np.ndindex(dom_shape):
            fmc_gc[t, :Nk, p[0], p[1]] = models[p].get_state()[:Nk]

    # store the diagnostics in a binary file
    diagnostics().dump_store(os.path.join(cfg['output_dir'], 'diagnostics.bin'))

    wrf_file.close() 


if __name__ == '__main__':
#    import profile
#    import pstats
#    profile.run('run_module(); print', 'spatial_model.stats')

#    stats = pstats.Stats('spatial_model.stats')
#    stats.strip_dirs()
#    stats.sort_stats('cumulative')
#    stats.print_stats()

    if len(sys.argv) != 2:
        print('USAGE: fmda.py <cfg-file>')
        sys.exit(1)

    run_module()

