import datetime
from scipy.spatial import KDTree
import numpy as np
from tools import get_vectors_for_year, get_EASE_grid, iterate_points,\
    remove_dead_tracks
import tools
from tqdm import trange
from tools import lonlat_to_xy
import pickle

dist_threshold = 25_000 #12_500
res_factor = 1
hemisphere = 'n'
start_year = 1979
no_years = 5 #41
printer = True
save_file_name = f'long_tracks_{start_year}_{start_year+no_years}'
input_data_dir = '/media/robbie/TOSHIBA_EXT/weekly_NSIDC_IMV/'

def make_weekly_tracks():


    ################################################################################

    tools.check_output_file_empty(save_file_name)
    tools.check_input_files_exist(start_year,no_years,input_data_dir,hemisphere)

    now = datetime.datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Start Time =", current_time)

    # Get EASE_lons & EASE_lats

    EASE_lons, EASE_lats = get_EASE_grid(input_data_dir)

    EASE_xs, EASE_ys = lonlat_to_xy(EASE_lons.ravel(), EASE_lats.ravel(), hemisphere=hemisphere)

    EASE_tree = KDTree(list(zip(EASE_xs, EASE_ys)))

    start_x, start_y = EASE_xs.ravel()[::res_factor], EASE_ys.ravel()[::res_factor]

    x_index = np.indices(EASE_lons.ravel()[::res_factor].shape)

    # Get dataset for first year

    # This is a 52x361x361x2 array
    # 0 axis is time
    # 1 & 2 axes are ease grid x & y dims
    # 3 axis just has length two, 0 for u vectors, 1 for v vecotrs

    velocities = get_vectors_for_year(data_dir = input_data_dir,
                                        year = start_year,
                                        hemisphere='n')

    #######################################################

    # Initialise week 1

    # Select first row (wk 1), and x and y slices

    data_for_start_week = {'u': velocities[0,:,:,0],
                          'v': velocities[0,:,:,1]}

    u_field = data_for_start_week['u'].ravel()[::res_factor]

    # Select points on ease grid with valid u velocity data on wk 0

    valid_start_x = start_x[~np.isnan(u_field)]
    valid_start_y = start_y[~np.isnan(u_field)]

    tracks_array = np.full((velocities.shape[0]+50,
                            valid_start_x.shape[0],
                            velocities.shape[3]), # Space for x and y coords and potentially div values
                            np.nan)

    tracks_array[0, :, 0] = valid_start_x
    tracks_array[0, :, 1] = valid_start_y

    save_key = 0
    start_weeks = {}
    week_num = 0

    for year in range(start_year, start_year + no_years + 1):
        print(year)

        if year != start_year: #We've already initialised using start_year data, no need to do it again

            velocities = get_vectors_for_year(data_dir=input_data_dir,
                                                year=year,
                                                hemisphere='n')

            # Make the tracks_array longer with each year to accomodate longer and longer tracks

            time_booster = np.full((velocities.shape[0], tracks_array.shape[1], velocities.shape[3]), np.nan)

            tracks_array = np.concatenate((tracks_array, time_booster), axis = 0)


        weeks_in_year = velocities.shape[0]

        for doy in range(0, weeks_in_year):

            if printer: print(f'Week_num: {week_num}, Extant tracks: {tracks_array.shape[1]}')

            # Get the ice motion field for that week

            u_data_for_week = velocities[doy,:,:,0]

            # Update points

            timestep = 24 * 60 * 60 * 7 # Number of seconds in a week

            updated_points = iterate_points(tracks_array[week_num,:, :],
                                               velocities[doy],
                                               EASE_tree,
                                               timestep)

            # Save these updated points to the numpy array

            tracks_array[week_num + 1, :, :] = updated_points

            # Identify index of dead tracks

            # Take the x coordinates of tracks_array for the week and look at ones where there's a nan

            tracks_array, save_key, start_weeks = remove_dead_tracks(tracks_array,
                                                                    save_key,
                                                                    week_num,
                                                                    start_weeks,
                                                                    save_file_name,
                                                                    printer)

            # Create new parcels in gaps
            # Make a decision tree for the track field

            # Identify all points of ease_grid with valid values

            u_field = u_data_for_week.ravel()[::res_factor]

            valid_grid_points = np.array([start_x[~np.isnan(u_field)],
                                          start_y[~np.isnan(u_field)]]).T


            # Iterate through all valid points to identify gaps using the tree

            # if tracks_array.shape[2]: # Seems like this is always true - test code without?

            track_tree = KDTree(tracks_array[week_num+1,:,:2])


            distance, index = track_tree.query(valid_grid_points)

            # Select rows of valid_grid_points where corresponding value in distance array is > dist_threshold

            # Remember 'valid_grid_points' is a 2D array of all x, y coordinates that have valid values on weeknum

            new_track_initialisations = valid_grid_points[distance>dist_threshold]

            additional_array = np.full((tracks_array.shape[0],
                                        new_track_initialisations.shape[0],
                                        velocities.shape[3]), np.nan)

            additional_array[week_num+1,:,:2] = new_track_initialisations

            if printer: print(f'Tracks added: {new_track_initialisations.shape[0]}')

            # Add newly intitiated tracks to other tracks

            tracks_array = np.concatenate((tracks_array, additional_array), axis = 1)

            week_num +=1

        with open(save_file_name+'.p', 'wb') as f:
            pickle.dump(start_weeks, f)


    # Save final array even though the tracks aren't dead

    tracks_array, save_key, start_weeks = remove_dead_tracks(tracks_array,
                                                             save_key,
                                                             week_num,
                                                             start_weeks,
                                                             save_file_name,
                                                             printer,
                                                             override=True)


    print(f'Tracks array shape: {tracks_array.shape}')
    now = datetime.datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("End Time =", current_time)


if __name__ == '__main__':
    make_weekly_tracks()