
from datetime import datetime
import netCDF4
import sys
import numpy as np

from spatial_model_utilities import find_closest_grid_point, great_circle_distance


def get_ngp(lat,lon,ncpath):
  """
  Finds the closest grid point to lat/lon, prints the indices and shows the distance to the point.
  """
  d = netCDF4.Dataset(ncpath)

  # find closest grid point
  glat,glon = d.variables['XLAT'][0,:,:], d.variables['XLONG'][0,:,:]
  i,j = find_closest_grid_point(lon,lat,glon,glat)

  dist = great_circle_distance(lon,lat,glon[i,j],glat[i,j])

  print('GP closest to lat/lon %g,%g is %d,%d with distance %g km.' % (lat,lon,i,j,dist))


def extract_nearest_time_series(varnames,lat,lon,ncpath):
  """
  Extract the time series in varnames for the grid point that is closest to the given lat/lon
  and store them in a CSV file in the same order as varnames.
  This assumes that the time dimension is first.  Each variable name is either a name or a tuple
  with name and index (for 4d vars).
  """

  d = netCDF4.Dataset(ncpath)

  # find closest grid point
  glat,glon = d.variables['XLAT'][0,:,:], d.variables['XLONG'][0,:,:]
  i,j = find_closest_grid_point(lon,lat,glon,glat)

  # retrieve all time series
  tss = []
  for vi in varnames:
    if isinstance(vi,tuple):
      vn,ndx = vi
      tss.append(d.variables[vn][:,ndx,i,j])
    else:
      tss.append(d.variables[vi][:,i,j])

  # extract time strings (example 2012-09-14_23:00:00) & convert to datetime
  esmft = d.variables['Times'][:,:]
  times = map(lambda x: ''.join(x), esmft)
  dts = map(lambda x: datetime(int(x[:4]),int(x[5:7]),int(x[8:10]),int(x[11:13]), int(x[14:16]), 0), times)

  d.close()

  for i in range(len(dts)):
    dt = dts[i]
    toks = map(lambda x: str(x), [dt.year,dt.month,dt.day,dt.hour,dt.minute,0])
    for ts in tss:
      toks.append(str(ts[i]))
    print(','.join(toks))


def display_help():
  print('usage: %s <command> <nc-file> <arguments>' % sys.argv[0])
  print('       varts <nc-file> lat lon varspec ...  -> extract variables from grid point closest to lat/lon')
  print('                        varspec is either a variable name or varname/index for 4d variables')
  print('       fmts <nc-file> lat lon -> extract fm model variables from grid point closest to lat/lon')
  print('       ngp <nc-file> lat lon -> find closest grid point and print it together with distance')
  print('')
  sys.exit(1)

def extract_varspec(v):
  vs = v.split('/')
  if len(vs) == 1:
    return vs[0]
  elif len(vs) == 2:
    return (vs[0],int(vs[1]))
  else:
    print('FATAL: varspec %s not understood.' % v)
    sys.exit(2)

if __name__ == '__main__':

  # if no arguments given then display help & exit
  if len(sys.argv) < 2:
    display_help()

  if sys.argv[1] == 'varts':
    if len(sys.argv) < 6:
      display_help()
    ncpath = sys.argv[2]
    lat,lon = float(sys.argv[3]),float(sys.argv[4])
    vs = map(extract_varspec, sys.argv[5:])
    extract_nearest_time_series(vs,lat,lon,ncpath)
  elif sys.argv[1] == 'fmts':
    ncpath = sys.argv[2]
    lat,lon = float(sys.argv[3]),float(sys.argv[4])
    # this ordering of variables matches load_wrf_data_csv.m in fmda_matlab
    vs = [ 'T2', 'Q2', 'PSFC', 'RAINC', 'RAINNC', ('FMC_GC',0), ('FMC_GC',1), ('FMC_GC',2)]
    extract_nearest_time_series(vs,lat,lon,ncpath)
  elif sys.argv[1] == 'ngp':
    ncpath = sys.argv[2]
    lat,lon = float(sys.argv[3]),float(sys.argv[4])
    get_ngp(lat,lon,ncpath)
  else:
    display_help()

  sys.exit(0)
