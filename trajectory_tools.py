import numpy as np
from scipy import interpolate
import datetime
from glob import glob
import netCDF4 as nc
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import html_animation
import pandas as pd

# New trajectory object
"""
new_trajectory class:
Defines a pandas dataframe that holds information about a trajectory object,
as well as associated variables and functions
"""
def new_trajectory(dates=[], lon=[], lat=[], height=[], data_dict={}):
    if type(lon) is np.ndarray:
        in_lon = lon.flatten()
    else:
        in_lon = np.array(lon)
    if type(lat) is np.ndarray:
        in_lat = lat.flatten()
    else:
        in_lat = np.array(lat)
    if type(dates) is np.ndarray:
        in_dates = dates.flatten()
    else:
        in_dates = np.array(dates)
    if (len(in_lon) != len(in_lat) != len(in_dates)):
        raise Exception("""Input arrays for dates, lat, lon have different
                        lengths""")
    else:
        length = len(in_dates)
    if len(height) == 0:
        in_height = np.full_like(in_lon, 0.)
    else:
        if type(height) is np.ndarray:
            in_height = height.flatten()
        else:
            in_height = np.array(height)
    keys = ['dates', 'time', 'lon', 'lat', 'height']
    time_offset = in_dates-in_dates[0]
    values = [in_dates, time_offset, in_lon, in_lat, in_height]
    if len(data_dict) > 0:
        keys.extend(data_dict.keys())
        values.extend(data_dict.values())
        values = np.stack(values, axis=1)
    else:
        values = np.stack(values, axis=1)
    if time_offset[1].total_seconds() < 0:
        index = np.arange(0,-length,-1).astype(int)
    else:
        index = np.arange(length).astype(int)
    return pd.DataFrame(values, columns=keys, index=index)

# Define a trajectories object
class trajectory:
    """
    trajectory class:
    Defines an object holding information about a HySplit style trajectory, and
    methods to extract the location at a specific time.
    """
    def __init__(self, lon=[], lat=[], year=[], month=[], day=[], hour=[]):
        if (len(lon) != len(lat) != len(year) != len(month) != len(day)
            != len(hour)):
            raise Exception("Input arrays have different lengths")
        # Check type of all inputs, make sure ndarrays are flat
        if type(lon) is np.ndarray:
            self.lon = lon.flatten()
        elif type(lon) is list:
            self.lon = lon
        else:
            raise Exception('lon input must be a list or ndarray type')
        if type(lat) is np.ndarray:
            self.lat = lat.flatten()
        elif type(lat) is list:
            self.lat = lat
        else:
            raise Exception('lat input must be a list or ndarray type')
        if type(year) is np.ndarray:
            self.year = year.flatten()
        elif type(year) is list:
            self.year = year
        else:
            raise Exception('year input must be a list or ndarray type')
        if type(month) is np.ndarray:
            self.month = month.flatten()
        elif type(month) is list:
            self.month = month
        else:
            raise Exception('month input must be a list or ndarray type')
        if type(day) is np.ndarray:
            self.day = day.flatten()
        elif type(day) is list:
            self.day = day
        else:
            raise Exception('day input must be a list or ndarray type')
        if type(hour) is np.ndarray:
            self.hour = hour.flatten()
        elif type(hour) is list:
            self.hour = hour
        else:
            raise Exception('hour input must be a list or ndarray type')
        self.dates = [ datetime.datetime(year=int(year[i]), month=int(month[i]),
                                        day=int(day[i]), hour=int(hour[i]))
                      for i in range(len(year)) ]
        self.dates = np.array(self.dates)
        # Check if trajectory is moving forwards or backwards in time
        if (self.dates[-1]-self.dates[0]).total_seconds() < 0:
            self.direction = -1
        else:
            self.direction = 1

    def __len__(self):
        return len(self.dates)

    def get_loc_at_t(self,date_time=None, bounds_error=False, fill_value=np.nan,
                     extrapolate=False, **kwargs):
        """
        Routine to linearly interpolate the location along the trajectory at a
        specific time. Accepts a datetime object or year, month, day (,hour,
        minute, ...) keywors to generate a datetime object. Returns a (lon, lat)
        tuple of the interpolated location along the trajectory. The
        bounds_error keyword controls whether the interpolation throws an error
        if outside of the trajectory or returns a fill value. The fill_value
        keyword sets the fill value to return (note, bounds_error must also be
        set to false). The extrapolate keyword determines if the trajectory
        shoulde be extrapolated beyond its bounds if set to true.
        """
        array_flag = False
        # Check date_time input is present and valid

        if hasattr(date_time, '__iter__'):
            for dt in date_time:
                if type(dt) is not datetime.datetime:
                    raise Exception("""date_time input must be an array of
                                    datetime.datetime objects""")
            if type(date_time) is not np.array:
                if type(date_time) is not list:
                    date_time = np.array([dt for dt in date_time])
                else:
                    date_time = np.array(date_time)
            array_flag = True
        elif isintance(date_time, datetime.datetime):
            array_flag = False
        elif date_time == None:
            kw_array_flags = [hasattr(kw, '__iter__') for kw in
                                                                kwargs.values()]
            if np.all(kw_array_flags):
                kw_lengths = np.array([len(kw) for kw in kwargs.values()])
                if not np.all(kw_lengths == kw_lengths[0]):
                    raise Exception("""year, month, day, ... keywords must all
                                       have the same length""")
                else:
                    kw_values = np.array(kwargs.values())
                    kw_keys = kwargs.keys()
                    date_time = np.array([
                        datetime.datetime(**dict(zip(kw_keys,kw_values[...,i])))
                        for i in range(kw_lengths[0])])
                    array_flag = True
            elif np.any(kw_array_flags):
                raise Exception("""mixing of array-like and scalar year, month,
                                   day, ... keywords not allowed""")
            else:
                date_time = datetime.datetime(**kwargs)
                array_flag = False
        else:
            raise Exception("""date_time input must be a
                            datetime.datetime object""")

        # Find time difference between input points and start of trajectory
        if array_flag:
            delta_t = np.array([(dt - self.dates[0]).total_seconds()
                                 for dt in date_time])
        else:
            delta_t = (date_time - self.dates[0]).total_seconds()
        # Get time difference along trajectory
        t = np.array([(date - self.dates[0]).total_seconds()
                       for date in self.dates])
        if extrapolate == True: fill_value = 'extrapolate'
        interp_lon = interpolate.interp1d(t, self.lon,
                                          bounds_error=bounds_error,
                                          fill_value=fill_value)(delta_t)
        interp_lat = interpolate.interp1d(t, self.lat,
                                          bounds_error=bounds_error,
                                          fill_value=fill_value)(delta_t)
        return interp_lon, interp_lat

def setup_hysplit(control_filename, start_lat, start_lon, start_height,
                  start_date=None, start_year=2017, start_month=9, start_day=2,
                  start_hour=0, duration=24, vert_motion=0, domain_height=15000,
                  metdata_filenames='/group_workspaces/cems2/nceo_generic/model_data/ARL_data/ERA5*.arl',
                  endpoint_filename='./tdump'):
    """
    Creates a HySPLIT CONTROL file that can be used to run the trajectory model
    """
    # Setup start date and times
    if start_date == None:
        start_date = datetime.datetime(start_year, start_month, start_day,
                                       start_hour)
    year = start_date.year % 100 # mod 100 to reduce to two digits
    month = start_date.month
    day = start_date.day
    hour = start_date.hour
    # Find metadata files
    metdata_filespaths = glob(metdata_filenames)
    n_metdata_files = len(metdata_filespaths)
    metdata_files = [f.split('/')[-1] for f in metdata_filespaths]
    metdata_paths = [metdata_filespaths[i][:-len(metdata_files[i])]
                        for i in range(n_metdata_files)]
    # Separate endpoint file and path
    endpoint_file = endpoint_filename.split('/')[-1]
    endpoint_path = endpoint_filename[:-len(endpoint_file)]
    # Check input lat, lon, heights input
    if (hasattr(start_lat, '__iter__') and hasattr(start_lon, '__iter__')
        and hasattr(start_height, '__iter__')):
        if not (len(start_lat) == len(start_lon) == len(start_height)):
            raise Exception("""setup_hysplit: arguments start_lat, start_lon and
                            start_height must either all be scalar or all be
                            lists of the same length""")
        else:
            n_startpoints = len(start_lat)
    elif (hasattr(start_lat, '__iter__') or hasattr(start_lon, '__iter__')
        or hasattr(start_height, '__iter__')):
        raise Exception("""setup_hysplit: arguments start_lat, start_lon and
                        start_height must either all be scalar or all be
                        lists of the same length""")
    else:
        n_startpoints = 1
        start_lat, start_lon, start_height = [start_lat], [start_lon], [start_height]
    # Check valid vertical motion
    if vert_motion not in range(6):
        raise Exception("""steup_hysplit: vertical motion must be one of the
                        following: 0:data, 1:isobaric, 2:isentropic, 3:constant
                        density, 4: sigma level, 5: divergence""")
    with open(control_filename,'w') as control:
        control.write('{year:02d} {month:02d} {day:02d} {hour:02d}\n'.format(
                                year=year, month=month, day=day, hour=hour))
        control.write('{n_startpoints:d}\n'.format(n_startpoints=n_startpoints))
        for i in range(n_startpoints):
            control.write('{lat:f} {lon:f} {height:f}\n'.format(
                    lat=start_lat[i], lon=start_lon[i], height=start_height[i]))
        control.write('{n_hours:d}\n'.format(n_hours=duration))
        control.write('{vert:d}\n'.format(vert=vert_motion))
        control.write('{domain_height:.1f}\n'.format(domain_height=domain_height))
        control.write('{n_met:d}\n'.format(n_met=n_metdata_files))
        for j in range(n_metdata_files):
            control.write('{met_path}\n'.format(met_path=metdata_paths[j]))
            control.write('{met_file}\n'.format(met_file=metdata_files[j]))
        control.write('{tdump_path}\n'.format(tdump_path=endpoint_path))
        control.write('{tdump_file}\n'.format(tdump_file=endpoint_file))

def parse_a_traj(traj_data, i, diagnostic_vars=[]):
    """
    parses an array of data from a hysplit output file and returns a single
    trajectory object
    """
    if i not in set(traj_data[:,0]):
        raise ValueError("""Input trajectory index: """+str(i)+"""not present in
                         trajectory data file""")
    wh = traj_data[:,0]==i
    traj_length = np.sum(wh)
    var_map = {'index':0, 'grid':1, 'year':2, 'month':3, 'day':4, 'hour':5,
               'minute':6, 'forecast_hour':7, 'trajectory_hour':8,
               'lat':9, 'lon':10, 'height':11}
    for i, var in enumerate(diagnostic_vars):
        var_map[var] = 12+i
    dates = [datetime.datetime(year=int(traj_data[wh,var_map['year']][i]),
                      month=int(traj_data[wh,var_map['month']][i]),
                      day=int(traj_data[wh,var_map['day']][i]),
                      hour=int(traj_data[wh,var_map['hour']][i]),
                      minute=int(traj_data[wh,var_map['minute']][i]))
                      for i in range(traj_length)]
    diagnostic_data = {}
    for var in diagnostic_vars:
        diagnostic_data[var] = traj_data[wh,var_map[var]]
    traj = new_trajectory(dates, traj_data[wh,var_map['lon']],
                          traj_data[wh,var_map['lat']],
                          traj_data[wh,var_map['height']],
                          diagnostic_data)
    return traj

def new_parse_hysplit(text_file):
    """
    parse_hysplit function
    Parses a HySPLIT output text file containing 1 or multiple trajectories.
    In: Text file to parse
    Out: Trajectory object for a single trajectory or a list of multiple
        trajectory objects for a file containing multiple trajectories
    """
    with open(text_file) as f:
        # Nested comprehension takes each line, cuts off the last character
        #  (newline), splits each line by spaces and removes the null characters
        #  left between successive spaces.
        traj_lines = [[s for s in line[:-1].split(' ') if s != '']
                                                      for line in f.readlines()]
    f.close()
    # Now parse through the list to find the trajectory info
    # First record = number of meteorology files
    n_datasets = int(traj_lines[0][0])
    n_traj = int(traj_lines[n_datasets+1][0])
    # Get the names of any diagnostic variables
    diagnostic_vars = traj_lines[n_datasets+n_traj+2][1:]
    # Get the data record of the trajectories
    traj_data = np.array(traj_lines[n_datasets+n_traj+3:]).astype(float)
    # Year is formatted YY, so need to add 2000/1900 as appropriate
    traj_data[...,2][traj_data[...,2]<50] += 100.
    traj_data[...,2] += 1900.
    # If more than 1 trajectory split into separate arrays
    traj = [parse_a_traj(traj_data, i, diagnostic_vars)
            for i in set(traj_data[:,0])]
    return traj

def parse_hysplit(text_file):
    """
    parse_hysplit function
    Parses a HySPLIT output text file containing 1 or multiple trajectories.
    In: Text file to parse
    Out: Trajectory object for a single trajectory or a list of multiple
        trajectory objects for a file containing multiple trajectories
    """
    with open(text_file) as f:
        # Nested comprehension takes each line, cuts off the last character
        #  (newline), splits each line by spaces and removes the null characters
        #  left between successive spaces.
        traj_lines = [[s for s in line[:-1].split(' ') if s != '']
                                                      for line in f.readlines()]
    f.close()
    # Now parse through the list to find the trajectory info
    # First record = number of meteorology files
    n_datasets = int(traj_lines[0][0])
    n_traj = int(traj_lines[n_datasets+1][0])
    # Get the data record of the trajectories
    traj_data = np.array(traj_lines[n_datasets+n_traj+3:]).astype(float)
    # Year is formatted YY, so need to add 2000/1900 as appropriate
    traj_data[...,2][traj_data[...,2]<50] += 100.
    traj_data[...,2] += 1900.
    # If more than 1 trajectory split into separate arrays
    if n_traj > 1:
        traj = []
        for i in set(traj_data[:,0]):
            wh = traj_data[:,0]==i
            # Create a trajectory object for each trajectory
            # Columns:
            # Year: 2, month: 3, day:4, hour:5, lon:10, lat:9
            # need to convert datetime inputs to int and flatten all inputs
            traj.append(trajectory(traj_data[wh,10].flatten(),
                                   traj_data[wh,9].flatten(),
                                   traj_data[wh,2].astype(int).flatten(),
                                   traj_data[wh,3].astype(int).flatten(),
                                   traj_data[wh,4].astype(int).flatten(),
                                   traj_data[wh,5].astype(int).flatten()))
            traj[-1].height=traj_data[wh,11].flatten()
    else:
        traj = trajectory(traj_data[...,10].flatten(),
                          traj_data[...,9].flatten(),
                          traj_data[...,2].astype(int).flatten(),
                          traj_data[...,3].astype(int).flatten(),
                          traj_data[...,4].astype(int).flatten(),
                          traj_data[...,5].astype(int).flatten())
        traj.height=traj_data[...,11].flatten()
    return traj

def map_traj(trajectory, out_path):
    """
    Produces an animated plot of the trajectory on seviri imagery
    """
    # TODO: make a globber for a range of dates
    """
    uyear = list(set(trajectory.year))
    umonth = list(set(trajectory.month))
    uday = list(set(trajectory.day))
    uhour = list(set(trajectory.hour))
    if len(uyear) > 1:
        uyear_str = [[n for n in str(y)] for y in uyear]
    """
    files = glob('/group_workspaces/jasmin2/acpc/Data/ORAC/clarify/merged/*SEVIRI*2017090[23]*.merged.nc')
    files.sort()
    out_files = []
    latrange = [np.floor(np.min(trajectory.lats)/5)*5,
                np.ceil(np.max(trajectory.lats)/5)*5]
    lonrange = [np.floor(np.min(trajectory.lon)/5)*5,
                np.ceil(np.max(trajectory.lon)/5)*5]
    # Get every 4th file for brevity
    for file in files[:]:
        nc_sev = nc.Dataset(file)
        if file == files[0]:
            lat_orac = nc_sev.variables['lat'][:]
            lon_orac = nc_sev.variables['lon'][:]
            wh = np.where(
                    np.logical_and(
                    np.logical_and(lat_orac>=latrange[0],lat_orac<=latrange[1]),
                    np.logical_and(lon_orac>=lonrange[0],lon_orac<=lonrange[1])))
            irange = [np.min(wh[0]), np.max(wh[0])+1]
            jrange = [np.min(wh[1]), np.max(wh[1])+1]
            lat_orac = lat_orac[irange[0]:irange[1], jrange[0]:jrange[1]]
            lon_orac = lon_orac[irange[0]:irange[1], jrange[0]:jrange[1]]
        ref_sev = nc_sev.variables['reflectance_in_channel_no_1'][irange[0]:irange[1], jrange[0]:jrange[1]]
        ir_sev = nc_sev.variables['brightness_temperature_in_channel_no_10'][irange[0]:irange[1], jrange[0]:jrange[1]]
        datestr = file.split('_')[-2]
        date = datetime.datetime(int(datestr[:4]), int(datestr[4:6]),
                                 int(datestr[6:8]), int(datestr[8:10]),
                                 int(datestr[10:]))
        traj_lon, traj_lat = trajectory.get_loc_at_t(date)
        # Now plot
        fig = plt.figure()
        fig.suptitle('SEVIRI Imagery along trajectory'+str(date))
        parallels = np.arange(latrange[0],latrange[1]+1,5.)
        meridians = np.arange(lonrange[0],lonrange[1]+1,5.)
        ax1 = fig.add_subplot(1,2,1)
        m1 = Basemap(projection='merc',llcrnrlat=latrange[0],urcrnrlat=latrange[1],llcrnrlon=lonrange[0],urcrnrlon=lonrange[1])
        m1.drawparallels(parallels,labels=[1,0,0,0],fontsize=10)
        m1.drawmeridians(meridians,labels=[0,0,0,1],fontsize=10)
        x,y = m1(lon_orac,lat_orac)
        img1 = m1.pcolormesh(x,y,ref_sev,cmap='gray',vmin=0.,vmax=1.)
        cbar1 = m1.colorbar(img1,location='right',pad="5%")
        ax1.set_title('SEVIRI reflectance')
        x,y = m1(traj_lon,traj_lat)
        m1.plot([x],[y],marker='o',markersize=5,color='blue')

        ax2 = fig.add_subplot(1,2,2)
        m2 = Basemap(projection='merc',llcrnrlat=latrange[0],urcrnrlat=latrange[1],llcrnrlon=lonrange[0],urcrnrlon=lonrange[1])
        m2.drawparallels(parallels,labels=[1,0,0,0],fontsize=10)
        m2.drawmeridians(meridians,labels=[0,0,0,1],fontsize=10)
        x,y = m2(lon_orac,lat_orac)
        img2 = m2.pcolormesh(x,y,ir_sev,cmap='inferno',vmin=275.,vmax=300.)
        cbar2 = m2.colorbar(img2,location='right',pad="5%", label='/K')
        ax2.set_title('SEVIRI brightness temperature')
        x,y = m2(traj_lon,traj_lat)
        m2.plot([x],[y],marker='o',markersize=5,color='blue')

        savename = out_path+'/trajectory2'+datestr+'.png'
        print 'Plotting to: '+savename
        plt.savefig(savename)
        out_files.append(savename)
        plt.close()

    # Make animation
    out_files.sort()
    html_animation.html_script([p.split('/')[-1] for p in out_files],
                                        'trajectory2',out_path,make_gif='yes')

def plot_traj(traj_list):
    if isinstance(traj_list, trajectory):
        lons = traj_list.lon
        lats = traj_list.lats
    else:
        traj = merge_traj(traj_list)
        lats = np.stack([t.lat for t in traj.flatten()], axis=-1).reshape((-1,traj.size))
        lons = np.stack([t.lon for t in traj.flatten()], axis=-1).reshape((-1,traj.size))
    # Now plot
    fig = plt.figure()
    latrange = [np.floor(np.min(lats)/5)*5,
                np.ceil(np.max(lats)/5)*5]
    lonrange = [np.floor(np.min(lons)/5)*5,
                np.ceil(np.max(lons)/5)*5]
    parallels = np.arange(latrange[0],latrange[1]+1,5.)
    meridians = np.arange(lonrange[0],lonrange[1]+1,5.)
    ax = fig.add_subplot(1,1,1)
    m = Basemap(projection='merc',llcrnrlat=latrange[0],urcrnrlat=latrange[1],llcrnrlon=lonrange[0],urcrnrlon=lonrange[1])
    m.drawparallels(parallels,labels=[1,0,0,0],fontsize=10)
    m.drawmeridians(meridians,labels=[0,0,0,1],fontsize=10)
    m.drawcoastlines()
    x,y = m(lons, lats)
    m.plot(x,y)
    plt.show()

def merge_traj(in_list):
    """
    Merges a list of trajectories or list of list of trajectories into a 2d
    numpy array.
    """
    traj, lons, lats = _parse_lists(in_list, target_type=trajectory,
                                        traj_list=[], lon=[], lat=[])
    n_traj = len(traj)
    lon_set = set(lons)
    lat_set = set(lats)
    if len(lon_set)*len(lat_set) != n_traj:
        raise Exception("""input list must be a matrix of trajectories""")
        # TODO: Make an alternative output for an irregular grid
    out_list = [[t for t in traj if t.lat[0]==l] for l in lat_set]
    # now sort by lon in place
    keyfun= lambda x: x.lon[0] # use a lambda function to get lons
    for l in out_list:
        l.sort(key=keyfun)
    out_array = np.array(out_list)
    return out_array

def _parse_lists(in_list, target_type=trajectory, traj_list=None, lon=None, lat=None):
    if traj_list == None:
        traj_list = []
    if lon == None:
        lon = []
    if lat == None:
        lat = []
    if isinstance(in_list, target_type):
        traj_list.append(in_list)
        lon.append(in_list.lon[0])
        lat.append(in_list.lat[0])
    elif hasattr(in_list, '__iter__'):
        for i in in_list:
            traj_list, lon, lat = _parse_lists(i, target_type,
                                                traj_list, lon, lat)
    else:
        raise Exception("""input is not an interable object or target_type""")
    return traj_list, lon, lat
