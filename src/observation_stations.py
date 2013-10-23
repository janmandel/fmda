
from spatial_model_utilities import find_closest_grid_point, great_circle_distance
import pytz
from datetime import datetime, timedelta
import string


def readline_skip_comments(f):
    """
    Read a new line while skipping comments.
    """
    l = f.readline().strip()
    while len(l) > 0 and l[0] == '#':
        l = f.readline().strip()
    return l


class Observation:
    """
    An observation of a field value at a certain time.
    """

    def __init__(self, s, tm, obs, var, field_name):
        """
        Constructs an observation packet from the info dictionary.
        """
        self.s = s
        self.tm = tm
        self.obs_val = obs
        self.field_name = field_name
        self.obs_var = var
        self.nearest_grid_pt_override = None


    def get_time(self):
        """
        Return time of observation.
        """
        return self.tm


    def get_value(self):
        """
        Return the observation value.
        """
        return self.obs_val


    def get_measurement_variance(self):
        """
        Return the variance of the measurement of the given field.
        """
        return self.obs_var


    def get_position(self):
        """
        Longitude and lattitude of the originating station (shortcut).
        """
        return self.s.get_position()


    def get_nearest_grid_point(self):
        """
        Return the indices that identify the nearest grid point.
        """
        if self.nearest_grid_pt_override is None:
            return self.s.get_nearest_grid_point()
        else:
            return self.nearest_grid_pt_override


    def set_nearest_grid_point(self, ngp):
        """
        Override the nearest grid point - used for TSM testing.
        """
        self.nearest_grid_pt_override = ngp


    def get_station(self):
        """
        Return the station from which this observation originates 
        """
        return self.s



class Station:
    """
    An observation station which stores and yields observations.
    All times must be in GMT.
    """
    def __init__(self):
        """
        Load a station from data file.
        """
        # array of observation times and dictionary of obs variables
        self.tm = []
        self.obs_vars = {}

        # no co-registration by default
        self.grid_pt = None
        self.dist_grid_pt = None



    def register_to_grid(self, wrf_data):
        """
        Find the nearest grid point to the current location.
        """
        # only co-register to grid if required
        mlon, mlat = wrf_data.get_lons(), wrf_data.get_lats()
        self.grid_pt = find_closest_grid_point(self.lon, self.lat, mlon, mlat)
        self.dist_grid_pt =  great_circle_distance(self.lon, self.lat,
                                                   mlon[self.grid_pt], mlat[self.grid_pt])


    def get_id(self):
        """
        Returns the id of the station.
        """
        return self.id


    def get_name(self):
        """
        Returns the name of the station.
        """
        return self.name
    
    
    def get_position(self):
        """
        Get geographical position of the observation station as a (lon, lat) tuple.
        """
        return self.lon, self.lat

    
    def get_nearest_grid_point(self):
        """
        Returns the indices of the nearest grid point.
        """
        return self.grid_pt
    
    
    def get_dist_to_grid(self):
        """
        Returns the distance in kilometers to the nearest grid point.
        """
        return self.dist_grid_pt
    
        
    def get_obs_times(self):
        """
        Return observatuion times.
        """
        return self.tm
    
    
    def get_elevation(self):
        """
        Return the elevation in meters above sea level.
        """
        return self.elevation
    
    

    def get_observations(self, obs_type):
        """
        Returns a list of Observations for given observation type (var name).
        """
        return self.obs[obs_type] if obs_type in self.obs else []





class MesoWestStation(Station):
    """
    An observation station with data downloaded from the MesoWest website in xls format.
    """
    
    def __init__(self, name):
        """
        Initialize the station using an info_string that is written by the scrape_stations.py
        script into the 'station_infos' file.
        """
        # parse the info_string
        self.name = name

        Station.__init__(self)


    def load_station_info(self, station_info):
        """
        Load station information from an .info file.                                                                  """
        with open(station_info, "r") as f:

            # read station id
            self.id = readline_skip_comments(f)

            # read station name
            self.name = readline_skip_comments(f)

            # read station geo location
            loc_str = readline_skip_comments(f).split(",")
            self.lat, self.lon = float(loc_str[0]), float(loc_str[1])

            # read elevation
            self.elevation = float(readline_skip_comments(f))

            # read sensor types
            self.sensors = map(lambda x: x.strip(), readline_skip_comments(f).split(","))

            # create empty lists for observations
            self.obs = {}
            for s in self.sensors:
                self.obs[s] = []



    def load_station_data(self, station_file):
        """
        Load all available fuel moisture data from the station measurement file
        in an obs file.
        """
        gmt_tz = pytz.timezone('GMT')

        with open(station_file, "r") as f:

            while True:

                # read in the date or exit if another packet is not found
                tm_str = readline_skip_comments(f)
                if len(tm_str) == 0:
                    break

                tstamp = gmt_tz.localize(datetime.strptime(tm_str, '%Y-%m-%d_%H:%M %Z'))

                # read in the variable names
                var_str = map(string.strip, readline_skip_comments(f).split(","))

                # read in observations
                vals = map(lambda x: float(x), readline_skip_comments(f).split(","))

                # read in variances
                variances = map(lambda x: float(x), readline_skip_comments(f).split(","))

                # construct observations
                for vn,val,var in zip(var_str, vals, variances):
                    self.obs[vn].append(Observation(self, tstamp, val, var, vn))



if __name__ == '__main__':

    o = MesoWestStation('../real_data/colorado_stations/BAWC2.xls', 'BAWC2,39.3794,-105.3383,2432.9136') 
    print(o.get_observations('fm10'))

