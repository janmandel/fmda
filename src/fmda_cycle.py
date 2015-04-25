# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22nd 2015 - Martin Vejmelka
Copyright UC Denver 2015

Performs data assimilation for cycling forecasts using WRF.
The algorithm is as follows:

1) check if a previous cycle is available
  a) yes? 
    - run DA using RAWS observations from start of previous cycle up to now (use wrfoutput)
    - store the current state in wrfinput
  b) no?
    - initialize from equilibrium (compute from wrfinput atmospheric state), perform on DA step using P0
    - store fuel moisture state/parameters/covariance

"""
import sys
import os
import numpy as np
from datetime import datetime
import pytz
import netCDF4
from observation import Observation
from wrf_model_data import WRFModelData
from trend_surface_model import fit_tsm
from spatial_model_utilities import great_circle_distance, find_closest_grid_point
from grid_moisture_model import GridMoistureModel
from diagnostics import init_diagnostics, diagnostics


def total_seconds(tdelta):
  """
  Utility function for python < 2.7, 2.7 and above have total_seconds()
  as a function of timedelta.
  """
  return tdelta.microseconds / 1e6 + (tdelta.seconds + tdelta.days * 24 * 3600)


def load_raws_observations(obs_file,glat,glon,grid_dist_km):
  """
  Loads all of the RAWS observations valid at the time in question
  and converts them to Observation objects.
  """
  # load observations & register them to grid
  orig_obs = []
  if os.path.exists(obs_file):
    orig_obs = np.loadtxt(obs_file,dtype=np.object,delimiter=',')
  else:
    print('WARN: no observation file found.')
  obss = []
  omin, omax = 0.6, 0.0

  # format of file
  #   0   1  2  3  4  5  6   7      8        9    10      11
  # yyyy,mm,dd,hh,MM,ss,lat,lon,elevation,var_id,value,variance

  for oo in orig_obs:
    ts = datetime(int(oo[0]),int(oo[1]),int(oo[2]),int(oo[3]),int(oo[4]),int(oo[5]),tzinfo=pytz.timezone('GMT'))
    lat, lon, elev = float(oo[6]), float(oo[7]), float(oo[8])
    obs, ovar = float(oo[10]), float(oo[11])
    i, j = find_closest_grid_point(lat,lon,glat,glon)

    # compute distance to grid points
    dist_grid_pt = great_circle_distance(lon,lat,glon[i,j],glat[i,j])

    # check & remove nonsense zero-variance (or negative variance) observations
    if ovar > 0 and dist_grid_pt < grid_dist_km / 2.0:
      obss.append(Observation(ts,lat,lon,elev,oo[9],obs,ovar,(i,j)))
      omin = min(omin, obs)
      omax = max(omax, obs)

  print('INFO: loaded %d observations in range %g to %g [%d available]' % (len(obss),omin,omax,len(obss)))
  return obss


def check_overlap(wrf_path,ts_now):
  """
  Check if the WRF file <wrf_path> timstamps contain <ts_now>.
  """
  wrfout = WRFModelData(wrf_path)
  outts = wrfout['GMT']
  if ts_now in outts:
    return True
  else:
    print("INFO: previous forecast [%s - %s] exists, running DA till %s" % (str(outts[0]),str(outts[-1]),str(ts_now)))
    return False


def build_observation_data(obss):
  """
  Repackage the matched time series into a time-indexed structure
  which gives details on the observed data and active observation stations.

      synopsis: obs_data = build_observation_data(obss)
  """
  # repackage all the observations into a time-indexed structure which groups
  # observations at the same time together
  obs_data = {}
  for obs in obss:
      t = obs.get_time()
      o = obs_data[t] if obs_data.has_key(t) else []
      o.append(obs)
      obs_data[t] = o

  return obs_data


def compute_equilibria(T, P, Q):
  """
  Computes atmospheric drying/wetting moisture equilibria from the pressure P [Pa],
  water vapor mixing ratio Q [-] and the surface temperature [K].
  """
  # saturated vapor pressure (at each location, size n x 1)
  Pws = np.exp(54.842763 - 6763.22/T - 4.210 * np.log(T) + 0.000367*T + np.tanh(0.0415*(T - 218.8))
      * (53.878 - 1331.22/T - 9.44523 * np.log(T) + 0.014025*T))


  # water vapor pressure (at each location, size n x 1)
  Pw = P * Q / (0.622 + (1 - 0.622) * Q)

  # relative humidity (percent, at each location, size n x 1)
  H = 100 * Pw / Pws
  mxpos = np.unravel_index(np.argmax(H),H.shape)

  H = np.minimum(H, 100.)

  d = 0.924*H**0.679 + 0.000499*np.exp(0.1*H) + 0.18*(21.1 + 273.15 - T)*(1 - np.exp(-0.115*H))
  w = 0.618*H**0.753 + 0.000454*np.exp(0.1*H) + 0.18*(21.1 + 273.15 - T)*(1 - np.exp(-0.115*H))

  d *= 0.01
  w *= 0.01

  # this is here to _ensure_ that drying equilibrium is always higher than (or equal to) wetting equilibrium
  Ed = np.maximum(d, w)
  Ew = np.minimum(d, w)

  return Ed, Ew


def store_covariance_matrix(P, path):
  """
  Store the grid of covariance matrices P in the netCDF file path.
  """
  dom_shape = (P.shape[0], P.shape[1])
  d = netCDF4.Dataset(path, 'w')
  d.createDimension('fuel_moisture_classes_stag', 5)
  d.createDimension('south_north', dom_shape[0])
  d.createDimension('west_east', dom_shape[1])
  Ps = d.createVariable('P', 'f4', ('fuel_moisture_classes_stag', 'fuel_moisture_classes_stag', 'south_north', 'west_east'))
  Ps[:,:,:,:] = P.transpose((2,3,0,1))
  d.close()


def execute_da_step(model, model_time, covariates, fm10):
  """
  Execute a single DA step from the current state/extended parameters and covariance matrix using
  the <covariates> and observations <fm10>.  Assimilation time window is fixed at 60 mins.
  """
  valid_times = [z for z in fm10.keys() if abs(total_seconds(z - model_time)) < 1800]
  print('INFO: there are %d valid times at model time %s' % (len(valid_times), str(model_time)))
  if len(valid_times) > 0:

    # retrieve all observations for current time
    obs_valid_now = []
    for z in valid_times:
        obs_valid_now.extend(fm10[z])

    for o in obs_valid_now:
      print o

    fmc_gc = model.get_state()
    dom_shape = fmc_gc.shape[:2]

    # construct covariate storage
    Xd3 = len(covariates) + 1
    X = np.zeros((dom_shape[0], dom_shape[1], Xd3))
    X[:,:,0] = fmc_gc[:,:,1]
    for c,i in zip(covariates,np.arange(len(covariates))+1):
      X[:,:,i] = covariates[i-1]

    # run the trend surface model (clamp output to [0.0 - 2.5] to be safe)
    Kf_fn, Vf_fn = fit_tsm(obs_valid_now, X)
    Kf_fn[Kf_fn < 0.0] = 0.0
    Kf_fn[Kf_fn > 2.5] = 2.5

    # preallocate Kalman gain variable [not really used]
    Kg = np.zeros((dom_shape[0], dom_shape[1], 5))

    # run the data assimilation step now
    print("Mean Kf: %g Vf: %g state[0]: %g state[1]: %g state[2]: %g\n" %
      (np.mean(Kf_fn), np.mean(Vf_fn), np.mean(fmc_gc[:,:,0]), np.mean(fmc_gc[:,:,1]), np.mean(fmc_gc[:,:,2])))
    model.kalman_update_single2(Kf_fn[:,:,np.newaxis], Vf_fn[:,:,np.newaxis,np.newaxis], 1, Kg)
    print("Mean Kf: %g Vf: %g state[0]: %g state[1]: %g state[2]: %g\n" %
      (np.mean(Kf_fn), np.mean(Vf_fn), np.mean(fmc_gc[:,:,0]), np.mean(fmc_gc[:,:,1]), np.mean(fmc_gc[:,:,2])))


def init_from_equilibrium(wrf_model, fm10, ts_now, cfg):
  """
  Initialize from the wrf_model equilibrium.
  """
  lat, lon = wrf_model.get_lats(), wrf_model.get_lons()
  dom_shape = lat.shape
  T2 = wrf_model['T2']
  Q2 = wrf_model['Q2']
  PSFC = wrf_model['PSFC']
  hgt = wrf_model['HGT']
  rain = wrf_model['RAIN']
  rain = np.log(rain + 1.0)
  constant = np.ones_like(T2)
  Ed,Ew = compute_equilibria(T2,PSFC,Q2)
  E = 0.5 * (Ed[0,:,:] + Ew[0,:,:])
  P0 = np.diag(cfg['P0'])

  Tk = np.array([1.0, 10.0, 100.0]) * 3600
  model = GridMoistureModel(E[:,:,np.newaxis][:,:,np.zeros((3,),dtype=np.int)], Tk, P0)

  # execute single DA step on the equilibrium with background covariance
  covariates = [T2,PSFC,lon - np.mean(lon),lat - np.mean(lat),hgt - np.mean(hgt),
                np.ones(dom_shape),rain]
  execute_da_step(model, ts_now, covariates, fm10)

  return model


def run_data_assimilation(wrf_model, fm10, ts_now, cfg):
  lat, lon = wrf_model.get_lats(), wrf_model.get_lons()
  dom_shape = lat.shape
  T2 = wrf_model['T2']
  Q2 = wrf_model['Q2']
  PSFC = wrf_model['PSFC']
  hgt = wrf_model['HGT']
  rain = wrf_model['RAIN']
  rain = np.log(rain + 1.0)
  constant = np.ones_like(T2)
  Ed,Ew = compute_equilibria(T2,PSFC,Q2)
  E = 0.5 * (Ed[0,:,:] + Ew[0,:,:])
  P0 = np.diag(cfg['P0'])

  Tk = np.array([1.0, 10.0, 100.0]) * 3600
  model = GridMoistureModel(E[:,:,np.newaxis][:,:,np.zeros((3,),dtype=np.int)], Tk, P0)

  # try to find a stored covariance matrix
  cov_path = os.path.join(os.path.dirname(cfg['wrf_output_prev'], 'P.nc'))
  if os.path.exists(cov_path):
    print('INFO: found stored covariance matrix, loading for init (also FMC_GC)...')
    model.get_state()[:,:,:3] = wrf_model['FMC_GC'][0,:3,:,:].transpose((1,2,0))
    model.get_state()[:,:,3:5] = wrf_model['FMEP'][0,:,:,:].transpose((1,2,0))
    d = netCDF4.Dataset(cov_path)
    model.get_state_covar()[:,:,:,:] = d.variables['P'][:,:,:,:]
  else:
    print('INFO: no covariance matrix found, intializing with background covariance.')

  return 0


def run_module():
  # read in configuration file to execute run
  print("Reading configuration from [%s]" % sys.argv[1])

  with open(sys.argv[1]) as f:
      cfg = eval(f.read())

  # init diagnostics
  init_diagnostics(os.path.join(cfg['output_dir'], 'moisture_model_v1_diagnostics.txt'))
  diagnostics().configure_tag("s2_eta_hat", True, True, True)
  diagnostics().configure_tag("kriging_rmse", True, True, True)
  diagnostics().configure_tag("kriging_beta", True, True, True)
  diagnostics().configure_tag("kriging_iters", False, True, True)
  diagnostics().configure_tag("kriging_subzero_s2_estimates", False, True, True)

  # load the wrfinput file
  wrfin = WRFModelData(cfg['wrf_input'], ['T2', 'Q2', 'PSFC', 'HGT', 'FMC_GC', 'FMEP'])
  lat, lon = wrfin.get_lats(), wrfin.get_lons()
  ts_now = wrfin['GMT'][0]
  dom_shape = lat.shape
  print('INFO: domain size is %d x %d grid points, wrfinput timestamp %s' % (dom_shape[0], dom_shape[1], str(ts_now)))
  print('INFO: domain extent is lats (%g to %g) lons (%g to %g).' % (np.amin(lat),np.amax(lat),np.amin(lon),np.amax(lon)))

  # compute the diagonal distance between grid points
  grid_dist_km = great_circle_distance(lon[0,0], lat[0,0], lon[1,1], lat[1,1])
  print('INFO: diagonal distance in grid is %g' % grid_dist_km)
 
  # load observations but discard those too far away from the grid nodes
  obss = load_raws_observations(cfg['observations'], lat, lon, grid_dist_km)
  fm10 = build_observation_data(obss)
  print('INFO: %d different time instances found in observations' % len(fm10))

  # if a previous cycle is available (i.e. the wrfoutput is a valid file)
  if os.path.exists(cfg['wrf_output_prev']) and check_overlap(cfg['wrf_output_prev'],ts_now):

    # load the model as a wrfout with all default variables
    wrfout = WRFModelData(cfg['wrf_output_prev'])
    outts = wrfout['GMT']
    print("INFO: previous forecast [%s - %s] exists, running DA till %s" % (str(outts[0]),str(outts[-1]),str(ts_now)))

    # run from the start until now (retrieve fuel moisture, extended parameters, covariance matrix)
    model =  run_data_assimilation(wrfout, fm10, ts_now, cfg)
    # store this for the current time instance (fm, ep in the wrfinput, P next to it)
    d = netCDF4.Dataset(cfg['wrf_input'], 'r+')
    d.variables['FMC_GC'] = fm
    d.variables['FMEP'] = ep
    d.close()

    # store the covariance matrix alongside the wrfinput file
    dir = os.path.dirname(wrfin)
    store_covariance_matrix(P, os.path.join(dir, 'P.nc'))

  else:

    print("INFO: no previous forecast found, running DA from equilibrium at %s" % (str(ts_now)))
    # initialize from weather equilibrium and perform one DA step
    model = init_from_equilibrium(wrfin, fm10, ts_now, cfg)

    # store result in wrfinput dataset
    d = netCDF4.Dataset(cfg['wrf_input'], 'r+')
    fmcep = model.get_state()
    d.variables['FMC_GC'][0,:3,:,:] = fmcep[:,:,:3].transpose((2,0,1))
    d.variables['FMEP'][0,:,:,:] = fmcep[:,:,3:5].transpose((2,0,1))
    d.close()
    store_covariance_matrix(model.get_state_covar(), os.path.join(os.path.dirname(cfg['wrf_input']), 'P.nc'))

  return 0


if __name__ == '__main__':
#    import profile
#    import pstats
#    profile.run('run_module(); print', 'spatial_model.stats')

#    stats = pstats.Stats('spatial_model.stats')
#    stats.strip_dirs()
#    stats.sort_stats('cumulative')
#    stats.print_stats()

  if len(sys.argv) != 2:
    print('USAGE: fmda_cycle.py <cfg-file>')
    sys.exit(1)

  run_module()
  sys.exit(0)

