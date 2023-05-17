from utils import dataset_reader
from utils.grid_utils import SceneObjects, AllObjects, generateLabelGrid, generateSensorGrid
import numpy as np
import os
from datetime import datetime
from tqdm import tqdm
import pandas as pd

np.random.seed(123)

def getstate(timestamp, track_dict, id):
    for key, value in track_dict.items():
        if key==id:
            return value.motion_states[timestamp]

def Dataprocessing():
    global vis_ids, vis_ax, vis_ay, ref_ax, ref_ay, ego_id, res

    main_folder = '/home/kin/DATA_HDD/yy/INTERACTION-Dataset-DR-multi-v1_2/'
    reformat_folder = '/home/kin/DATA_HDD/yy/INTERACTION-Dataset-DR-multi-v1_2/occ_files_all_n1/'

    for data_type in ['val']:
        data_folder = os.path.join(main_folder, data_type)
        reformat_data_folder = os.path.join(reformat_folder, data_type)
        if not os.path.exists(reformat_data_folder):
            os.makedirs(reformat_data_folder)
        for scene in os.listdir(data_folder):
            full_data_path = os.path.join(data_folder, scene)
            full_reformat_data_path = os.path.join(reformat_data_folder, scene)

            all_data = pd.read_csv(full_data_path)
            all_data["occluded"] = -1 # init occluded

            new_data = pd.DataFrame()
            case_ids = all_data['case_id'].unique()
            for cnt, case_id in tqdm(enumerate(case_ids), total=len(case_ids)):
                all_data_in_one_case = all_data.loc[(all_data['case_id'] == case_id) & (all_data['agent_type'] == 'car')]
                val_data = pd.DataFrame()

                # val data based on Autobot
                for agent_id in list(set(all_data_in_one_case['track_id'].tolist())):
                    agent_data = all_data_in_one_case.loc[(all_data_in_one_case['track_id'] == agent_id)]
                    if len(agent_data['frame_id'].to_list()) < 40:  # only take complete trajectories
                        continue
                    val_data = pd.concat([val_data, agent_data])
                all_data_in_one_case = val_data


                ego_id = np.random.choice(all_data_in_one_case['track_id'].unique().tolist())
                car_track_dict = dataset_reader.read_tracks_v2(all_data_in_one_case)

                # cal occlusion
                track_pedes_dict = None # no need track_pedes_dict
                res = 1. # resolution in grid

                for key, value in car_track_dict.items():
                    if key == ego_id:
                        continue

                    vis_ids = []

                    start_timestamp = value.time_stamp_ms_first
                    last_timestamp = value.time_stamp_ms_last
                    vis_ax = []
                    vis_ay = []
                    # Get accelerations.

                    for stamp in range(start_timestamp, last_timestamp, 100):
                        object_id, pedes_id = SceneObjects(car_track_dict, stamp, track_pedes_dict)
                        ego_ms = getstate(stamp, car_track_dict, ego_id)

                        # Get label grid.
                        label_grid_ego, _, _, _, pre_local_x_ego, pre_local_y_ego = \
                            generateLabelGrid(stamp, car_track_dict, ego_id, object_id, ego_flag=True, res=res, \
                                              track_pedes_dict=track_pedes_dict, pedes_id=pedes_id)

                        # Get sensor grid.
                        width = value.width
                        length = value.length

                        _, occluded_id, visible_id = \
                            generateSensorGrid(label_grid_ego, pre_local_x_ego, pre_local_y_ego, \
                                               ego_ms, width, length, res=res, ego_flag=True)
                        
                        # print("occluded_id", occluded_id)
                        # print("visible_id", visible_id)
                        for v_id in visible_id:
                            if v_id >0:
                                all_data_in_one_case.loc[(all_data_in_one_case["timestamp_ms"]==stamp) \
                                                         & (all_data_in_one_case["track_id"]==int(v_id)), "occluded"] = 0
                        for o_id in occluded_id:
                            if o_id >0:
                                all_data_in_one_case.loc[(all_data_in_one_case["timestamp_ms"]==stamp) \
                                                         & (all_data_in_one_case["track_id"]==int(o_id)), "occluded"] = 1

                # if no occluded in this case, skip
                if len(all_data_in_one_case.loc[all_data_in_one_case["occluded"]==1]) < 1:
                    continue

                # replace ego_id with 0
                all_data_in_one_case.loc[(all_data_in_one_case["track_id"]==ego_id), "track_id"] = 0
                all_data_in_one_case["case_id"] = case_id
                all_data_in_one_case["agent_type"] = "car"
                new_data = pd.concat([new_data, all_data_in_one_case])
            # save all
            new_data.to_csv(full_reformat_data_path, index=False)
if __name__ == "__main__":
    start_time = datetime.now()
    Dataprocessing()
    end_time = datetime.now()
    print("Full path execution time:", end_time - start_time)
