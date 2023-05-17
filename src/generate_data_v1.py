from utils import dataset_reader
from utils.grid_utils import SceneObjects, AllObjects, generateLabelGrid, generateSensorGrid
from utils.dataset_types import Track
import numpy as np
import os
from datetime import datetime
import glob
from tqdm import tqdm
import pandas as pd
import re

np.random.seed(123)


def getstate(timestamp, track_dict, id):
    for key, value in track_dict.items():
        if key==id:
            return value.motion_states[timestamp]

def getNumScene(folder):
    # List all .csv files in the directory
    csv_files = [f for f in os.listdir(folder) if f.endswith('.csv')]

    # Use regular expressions to extract the number from each file name
    numbers = [int(re.search(r'(\d+)', f).group(1)) for f in csv_files]

    # Get the maximum number
    max_number = max(numbers)
    return max_number

def Dataprocessing():
    global vis_ids, vis_ax, vis_ay, ref_ax, ref_ay, ego_id, res

    main_folder = '/home/kin/workspace/MultiAgentVariationalOcclusionInference/data/INTERACTION-Dataset-DR-v1_1'
    reformat_folder = '/home/kin/workspace/MultiAgentVariationalOcclusionInference/data/INTERACTION-Dataset-DR-v1_1/occ_files_all_n1/'
    scenarios = ['DR_USA_Intersection_GL']
    # all data
    # scenarios = os.listdir(os.path.join(main_folder, 'recorded_trackfiles/'))
    num_files = []
    for scene in scenarios:
        num_files.append(getNumScene(os.path.join(main_folder, 'recorded_trackfiles/'+scene)))

    for scene, nth_scene in zip(scenarios, num_files):
        
        full_folder = reformat_folder + scene+'/'
        if not os.path.exists(full_folder):
            os.makedirs(full_folder)

        for i in tqdm(range(nth_scene)):
            print("Processing scene:", scene, "file:", i)
            i_str = ['%03d' % i][0]


            car_file_name = 'vehicle_tracks_'+ i_str +'.csv'
            
            filename = os.path.join(main_folder, 'recorded_trackfiles/'+scene+'/'+'vehicle_tracks_'+ i_str +'.csv')
            df_car = pd.read_csv(filename)
            df_car["occluded"] = -1
            track_dict = dataset_reader.read_tracks(filename)
            filename_pedes = os.path.join(main_folder, 'recorded_trackfiles/'+scene+'/'+'pedestrian_tracks_'+ i_str +'.csv')

            if os.path.exists(filename_pedes):
                track_pedes_dict = dataset_reader.read_pedestrian(filename_pedes)
                df_pedes = pd.read_csv(filename_pedes)
                df_pedes["occluded"] = -1
                df_pedes["psi_rad"] = 0
                df_pedes["length"] = 0
                df_pedes["width"] = 0
                pedes_file_name = 'pedestrian_tracks_'+ i_str +'.csv'
            else:
                track_pedes_dict = None
            
            run_count = 0

            res = 1.
            vehobjects, _ = AllObjects(track_dict, track_pedes_dict)
            
            processed_file = glob.glob(os.path.join(main_folder, '/Processed_data_new_goal/pkl/DR_USA_Intersection_GL_'+i_str+'*_ego_*'))
            processed_id = [ int(file.split('_')[-1][:-4]) for file in processed_file]

            num = 0
            sampled_key = [id for id in vehobjects if id not in processed_id][num:]
            run_count = num
            sampled_key = np.random.choice(sampled_key,np.minimum(100, len(sampled_key)),replace=False)
            
            for key, value in tqdm(track_dict.items()):
                assert isinstance(value, Track)
                if key in sampled_key:
                    ego_id = int(key)
                    vis_ids = []

                    start_timestamp = value.time_stamp_ms_first
                    last_timestamp = value.time_stamp_ms_last
                    vis_ax = []
                    vis_ay = []

                    # Get accelerations.
                    for stamp in range(start_timestamp, last_timestamp, 100):
                        object_id, pedes_id = SceneObjects(track_dict, stamp, track_pedes_dict)
                        ego_ms = getstate(stamp, track_dict, ego_id)

                        # Get label grid.
                        label_grid_ego, _, _, _, pre_local_x_ego, pre_local_y_ego = generateLabelGrid(stamp, track_dict, ego_id, object_id, ego_flag=True, res=res, track_pedes_dict=track_pedes_dict, pedes_id=pedes_id)

                        # Get sensor grid.
                        width = value.width
                        length = value.length
                        _, occluded_id, visible_id = generateSensorGrid(label_grid_ego, pre_local_x_ego, pre_local_y_ego, ego_ms, width, length, res=res, ego_flag=True)
                        
                        # print("occluded_id", occluded_id)
                        # print("visible_id", visible_id)
                        for v_id in visible_id:
                            if v_id >0:
                                df_car.loc[(df_car["timestamp_ms"]==stamp) & (df_car["track_id"]==int(v_id)), "occluded"] = 0
                            elif track_pedes_dict is not None:
                                id_str = 'P' + str(abs(int(v_id)))
                                df_pedes.loc[(df_pedes["timestamp_ms"]==stamp) & (df_pedes["track_id"]==id_str), "occluded"] = 0
                        for o_id in occluded_id:
                            if o_id >0:
                                df_car.loc[(df_car["timestamp_ms"]==stamp) & (df_car["track_id"]==int(o_id)), "occluded"] = 1
                            elif track_pedes_dict is not None:
                                id_str = 'P' + str(abs(int(o_id)))
                                df_pedes.loc[(df_pedes["timestamp_ms"]==stamp) & (df_pedes["track_id"]==id_str), "occluded"] = 1

                    run_count += 1

            # save new csv with occluded column
            df_car.to_csv(full_folder+car_file_name, index=False)
            if track_pedes_dict is not None:
                df_pedes.to_csv(full_folder+pedes_file_name, index=False)
            print("Done, check folder:", full_folder)
            # break
if __name__ == "__main__":
    start_time = datetime.now()
    Dataprocessing()
    end_time = datetime.now()
    print("Full path execution time:", end_time - start_time)
