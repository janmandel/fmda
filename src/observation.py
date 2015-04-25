


class Observation:
  """
  An observation of a field value at a certain time.
  """

  def __init__(self,tm,lat,lon,elev,var_id,obs,var,ngp):
    """
    Constructs an observation packet from the info dictionary.
    """
    self.tm = tm
    self.lon = lon
    self.lat = lat
    self.elevation = elev
    self.obs_val = obs
    self.obs_var = var
    self.varid = var_id
    self.ngp = ngp


  def get_value(self):
    """
    Return the observation value.
    """
    return self.obs_val


  def get_variance(self):
    """
    Return the variance of the measurement of the given field.
    """
    return self.obs_var


  def get_elevation(self):
    """
    Return the elevation, where the observation was taken.
    """
    return self.elevation


  def get_position(self):
    """
    Longitude and lattitude of the originating station (shortcut).
    """
    return (self.lat, self.lon)


  def get_nearest_grid_point(self):
    """
    Return the indices that identify the nearest grid point.
    """
    return self.ngp

  
  def get_time(self):
    """
    Return the GMT time of the observations.
    """
    return self.tm

  
  def get_variable(self):
    """
    Returns the variable which was observed.
    """
    return self.var_id

  def __str__(self):
    """
    Returns a string representation.
    """
    return "%s %s pos: [%g,%g] val: %g (%g)" % (str(self.tm),self.varid,self.lat,self.lon,self.obs_val,self.obs_var)

