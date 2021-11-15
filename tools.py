from pyproj import Proj, transform
from netCDF4 import Dataset
import numpy as np
import h5py
import os
from pyproj import Proj, Transformer

def get_EASE_grid(input_data_dir):

    # data = Dataset(f'{input_data_dir}icemotion_weekly_nh_25km_19820101_19821231_v4.1.nc')
    data = Dataset('/media/robbie/TOSHIBA_EXT/weekly_NSIDC_IMV/icemotion_weekly_nh_25km_19820101_19821231_v4.1.nc')
    lons = np.array(data['longitude'])
    lats = np.array(data['latitude'])

    return(lons, lats)

def remove_dead_tracks(tracks_array,
                       save_key,
                       day_num,
                       start_days,
                       save_file_name,
                       printer,
                       override=False):

    if override:

        dead_cols = list(np.arange( tracks_array[day_num + 1, :, 0].shape[0]))
        print(f'Saving surviving {dead_cols[-1]} tracks')

    else:
        dead_cols = [index[0] for index in np.argwhere(np.isnan(tracks_array[day_num + 1, :, 0]))]

    # deadcols is a list of column indexes that have died.

    # Save dead tracks

    for column_no in dead_cols:

        # Find number of non-zero entries in array of x coords

        track_length = np.count_nonzero(~np.isnan(tracks_array[:, column_no, 0]))

        if track_length > 1:
            # Start day can be calculated from subtracting the number of extant days from day of death
            start_day = day_num - track_length + 1

            if start_day < 0:
                raise

            select_and_save_track(tracks_array[start_day:day_num + 1, column_no, :],
                                  save_key,
                                  save_file_name)

            start_days[save_key] = {'start_day': start_day,
                                    'day_num': day_num}

            save_key += 1

    # Remove dead tracks

    tracks_array = np.delete(tracks_array, dead_cols, axis=1)

    if printer: print(f'Tracks killed: {len(dead_cols)}')

    return(tracks_array, save_key, start_days)

def calculate_div_from_velocities(velocities):

    """ Takes in a velocity distribution and calculates the divergence distribution using np.gradient

    Args:
        velocities: an array of velocities

    Returns:
        div: an array of the divergence
    """

    dudx = np.gradient(velocities[:,:,:,0], axis=1)

    dvdy = np.gradient(velocities[:,:,:,1], axis=2)

    div = np.add(dudx,dvdy)

    return(div)


def get_vectors_for_year(data_dir,year,hemisphere):

    data_for_year = Dataset(f'{data_dir}icemotion_weekly_{hemisphere}h_25km_{year}0101_{year}1231_v4.1.nc')

    all_u, all_v = np.array(data_for_year['u']), np.array(data_for_year['v'])

    velocities = np.stack((all_u, all_v), axis=3)

    velocities[velocities == -9999.0] = np.nan

    velocities = velocities/100 #Convert cm/s to m/s

    return(velocities)

def check_output_file_empty(save_file_name):

    """ Checks that the specified locations to save output data are free, and not already occupied by old results.

    Args:
        save_file_name: the file name without extensions. The extensions .h5 and .p are then added.

    Returns:
        zero for satisfactory result. Otherwise raises OSError.
    """

    if os.path.exists(save_file_name+'.p'): raise OSError(f'The file {save_file_name}.p already exists')
    elif os.path.exists(save_file_name + '.h5'): raise OSError(f'The file {save_file_name}.h5 already exists')
    else: return 0

def check_input_files_exist(start_year,no_years,input_data_dir,hemisphere):

    """ Checks that all the files required for your run exist

    Args:
        start_year: start year of your run. First year for which data is checked
        no_years: number of years to run. If 5 and start_year =2000 then 2000,2001,2002,2003,2004 are checked
        input_data_dir: the directory in which the data are stored. Must be appended by / character
        hemisphere: the hemisphere of your run

    Returns:
        zero for satisfactory result. Otherwise an exception is raised.
    """

    for year in range(no_years):
        assert os.path.exists(f'{input_data_dir}icemotion_weekly_{hemisphere}h_25km_{start_year+year}0101_{start_year+year}1231_v4.1.nc')

    return 0

def select_and_save_track(track, key, f_name):

    """ Writes floe trajectory to hdf5 file in append mode

    Args:
        track: track coords
        track_no: int representing track number (for later data retrieval)
        f_name: file name of hdf5 storage file

    Returns:
        no return, writes to file.

    """

    with h5py.File(f_name+'.h5', 'a') as hf:
        hf[f't{key}'] = track

def iterate_points(array,
                   velocities_on_day,
                   EASE_tree,
                   timestep):

    """ Takes in an array of positions and some velocity vectors, and moves the positions based on the velocities.

    Args:
        array: an n x m array. n represents the number of extant tracks. m = 2 if no divergence (x & y dims). m = 3 if
        make_divergence_series == True, with the div value being the third dimension.
        velocities_on_day: 361x361xm array. m defined as above.
        EASE_tree: A scipy.KDTree decision tree calculated with the EASE x & y coordinates
        timestep: the number of seconds to integrate over.
        make_divergence_series (bool): true if tracks accumulate divergence

    Returns:
        new_positions: n x m array with n being the number of extant tracks and m representing x, y and cumulative div.

    """


    distances, indexs = EASE_tree.query(array[:,:2])

    velocities_of_interest = np.array([velocities_on_day[:,:,0].ravel()[indexs],
                                       velocities_on_day[:,:,1].ravel()[indexs]]).T

    displacements = velocities_of_interest * timestep

    new_positions = array + displacements

    return (new_positions)


def lonlat_to_xy(coords_1, coords_2, hemisphere, inverse=False):
    """Converts between longitude/latitude and EASE xy coordinates.

    Args:
        lon (float): WGS84 longitude
        lat (float): WGS84 latitude
        hemisphere (string): 'n' or 's'
        inverse (bool): if true, converts xy to lon/lat

    Returns:
        tuple: pair of xy or lon/lat values
    """

    EASE_Proj = {'n': 'EPSG:3408',
                 's': 'EPGS:3409'}

    WGS_Proj = 'EPSG:4326'

    for coords in [coords_1, coords_2]: assert isinstance(coords, (np.ndarray, list))

    if inverse == False:  # lonlat to xy

        lon, lat = coords_1, coords_2

        transformer = Transformer.from_crs(WGS_Proj, EASE_Proj[hemisphere])

        x, y = transformer.transform(lat, lon)

        return (x, y)

    else:  # xy to lonlat

        x, y = coords_1, coords_2

        transformer = Transformer.from_crs(EASE_Proj[hemisphere], WGS_Proj)

        lat, lon = transformer.transform(x, y)

        return (lon, lat)
