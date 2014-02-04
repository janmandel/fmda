
from spatial_model_utilities import great_circle_distance
from observation_stations import MesoWestStation

import string
import sys
import os


if __name__ == '__main__':

    with open(sys.argv[1]) as f:
        cfg = eval(f.read())
    
    with open(cfg['station_list_file'], 'r') as f:
        si_list = f.read().split('\n')

    si_list = filter(lambda x: len(x) > 0 and x[0] != '#', map(string.strip, si_list))

    # for each station id, load the station
    stations = []
    for code in si_list:
        mws = MesoWestStation(code)
        mws.load_station_info(os.path.join(cfg["station_info_dir"], "%s.info" % code))
        stations.append(mws)


    cmd = sys.argv[2]
    if cmd == 'find_closest':
        lat = float(sys.argv[3])
        lon = float(sys.argv[4])
        min_dist = 1000000 # that should be more [km] than anything we will find :)
        closest = None
        for st in stations:
            slon,slat = st.get_position()
            d = great_circle_distance(lon,lat,slon,slat)
            if d < min_dist:
                closest = st
                min_dist = d

        clon,clat = closest.get_position()
        print('Closest station is %s at %g,%g with dist %g km' % (closest.get_id(), clat, clon, min_dist))
                
