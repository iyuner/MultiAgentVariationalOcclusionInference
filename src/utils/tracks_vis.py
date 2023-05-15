# Plot the full pipeline occlusion inference. Code is adapted from: https://github.com/interaction-dataset/interaction-dataset.

import matplotlib
import matplotlib.patches
import matplotlib.transforms
from matplotlib.offsetbox import OffsetImage, AnnotationBbox, AnchoredOffsetbox
from scipy import ndimage
import skimage.transform
from PIL import Image
import pdb
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from copy import deepcopy

import os
import time

seed = 123

import numpy as np
np.random.seed(seed)
from matplotlib import pyplot as plt
import torch
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
import torch.nn as nn
torch.autograd.set_detect_anomaly(True)

import io
from tqdm import tqdm
import time

import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import OrderedDict, defaultdict
import torch._utils

from src.utils.dataset_types import Track, MotionState
from src.utils.grid_utils import *
from src.utils.grid_fuse import *
from src.utils.utils_model import to_var
from src.driver_sensor_model.models_cvae import VAE
from src.utils.interaction_utils import *

try:
    torch._utils._rebuild_tensor_v2
except AttributeError:
    def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks):
        tensor = torch._utils._rebuild_tensor(storage, storage_offset, size, stride)
        tensor.requires_grad = requires_grad
        tensor._backward_hooks = backward_hooks
        return tensor
    torch._utils._rebuild_tensor_v2 = _rebuild_tensor_v2

def rotate_around_center(pts, center, yaw):
    return np.dot(pts - center, np.array([[np.cos(yaw), np.sin(yaw)], [-np.sin(yaw), np.cos(yaw)]])) + center

def polygon_xy_from_motionstate(ms, width, length):
    assert isinstance(ms, MotionState)
    lowleft = (ms.x - length / 2., ms.y - width / 2.)
    lowright = (ms.x + length / 2., ms.y - width / 2.)
    upright = (ms.x + length / 2., ms.y + width / 2.)
    upleft = (ms.x - length / 2., ms.y + width / 2.)
    return rotate_around_center(np.array([lowleft, lowright, upright, upleft]), np.array([ms.x, ms.y]), yaw=ms.psi_rad)


def polygon_xy_from_motionstate_pedest(ms, width, length):
    assert isinstance(ms, MotionState)
    lowleft = (ms.x - length / 2., ms.y - width / 2.)
    lowright = (ms.x + length / 2., ms.y - width / 2.)
    upright = (ms.x + length / 2., ms.y + width / 2.)
    upleft = (ms.x - length / 2., ms.y + width / 2.)
    return np.array([lowleft, lowright, upright, upleft])

def update_objects_plot(timestamp, patches_dict, text_dict, axes, track_dict=None, pedest_dict=None,
 data=None, car_ids=None, sensor_grids=None, id_grids=None, label_grids=None, grids_dict=None,
 driver_sensor_data=None, driver_sensor_state_data=None, driver_sensor_state=None, driver_sensor_state_dict=None, endpoint=None, models=None, mode='evidential', model='vae'):

    ego_id = data[1][1]

    if track_dict is not None:

        # Plot and save the ego-vehicle first.
        assert isinstance(track_dict[ego_id], Track)
        value = track_dict[ego_id]
        if (value.time_stamp_ms_first <= timestamp <= value.time_stamp_ms_last):
            
            ms_ego = value.motion_states[timestamp]
            assert isinstance(ms_ego, MotionState)

            # Check if the ego vehicle already exists.
            if ego_id not in patches_dict:
                width = value.width
                length = value.length

                rect = matplotlib.patches.Polygon(polygon_xy_from_motionstate(ms_ego, width, length), closed=True,
                                              zorder=40, color='green')

                axes.add_patch(rect)
                patches_dict[ego_id] = rect
                text_dict[ego_id] = axes.text(ms_ego.x, ms_ego.y + 3, str(ego_id), horizontalalignment='center', zorder=50, fontsize='xx-large')
                
            else:
                width = value.width
                length = value.length
                patches_dict[ego_id].set_xy(polygon_xy_from_motionstate(ms_ego, width, length))

                text_dict[ego_id].set_position((ms_ego.x, ms_ego.y + 3))
        
                # Set colors of the ego vehicle.
                patches_dict[ego_id].set_color('green')

            if ego_id in grids_dict.keys():
                grids_dict[ego_id].remove()

            # Plot the ego vehicle's vanilla grid.
            res = 1.0
            colormap = ListedColormap(['white','black','gray'])  
            object_id, pedes_id = SceneObjects(track_dict, timestamp, track_pedes_dict=pedest_dict)
            local_x_ego, local_y_ego, _, _ = local_grid(ms_ego, width, length, res=res, ego_flag=True)
            label_grid_ego, center, x_local, y_local, pre_local_x, pre_local_y = generateLabelGrid(timestamp, track_dict, ego_id, object_id, ego_flag=True, res=res, track_pedes_dict=pedest_dict, pedes_id=pedes_id)
            start = time.time()
            sensor_grid_ego, occluded_id, visible_id = generateSensorGrid(label_grid_ego, pre_local_x, pre_local_y, ms_ego, width, length, ego_flag=True, res=res)
            visible_id += [ego_id]
            # sensor_grid_ego = 2.0*np.ones((sensor_grid_ego.shape))
            
            full_sensor_grid_dst, mask_unk = get_belief_mass(sensor_grid_ego, ego_flag=True)
            full_sensor_grid = pignistic(full_sensor_grid_dst)
            full_sensor_grid_dst[0,label_grid_ego[0] == 2] = 0.0
            full_sensor_grid_dst[1,label_grid_ego[0] == 2] = 1.0
            full_sensor_grid[label_grid_ego[0] == 2] = 0.0
            sensor_grid_ego[label_grid_ego[0] == 2] = 0.0

            box = axes.pcolormesh(local_x_ego, local_y_ego, full_sensor_grid, cmap='gray_r', zorder=30, alpha=0.7, vmin=0, vmax=1)
            grids_dict[ego_id] = box

        # Initialize the variables to keep for later computation.
        if model != 'kmeans':
            all_latent_classes = []
            ref_local_xy_list = []
            ego_local_xy_list = []
            alpha_p_list = []
        sensor_grid_ego_dst = [full_sensor_grid_dst]

        if mode == 'average':
            average_mask = np.zeros(full_sensor_grid.shape)
            driver_sensor_grid = np.zeros(full_sensor_grid.shape)

        # Plot the rest of the agents.
        for key, value in track_dict.items():
            assert isinstance(value, Track)
            if ((value.time_stamp_ms_first <= timestamp <= value.time_stamp_ms_last)):

                ms = value.motion_states[timestamp]
                assert isinstance(ms, MotionState)

                if key not in patches_dict:
                    width = value.width
                    length = value.length

                    if ((key in visible_id) and (key != ego_id)):
                        rect = matplotlib.patches.Polygon(polygon_xy_from_motionstate(ms, width, length), closed=True,
                                                      zorder=40, color='cyan')
                        colormap = ListedColormap(['white','black','gray'])
                        res = 1.0            
                        if key in driver_sensor_state_data.keys():
                            for state in driver_sensor_state_data[key]:
                                if state[0] == timestamp:
                                    driver_sensor_state[key] = np.reshape(state, (1,-1))
                                    break

                    elif ((key not in visible_id) and (key != ego_id)) :    
                        rect = matplotlib.patches.Polygon(polygon_xy_from_motionstate(ms, width, length), closed=True,
                                                      zorder=20)
                    patches_dict[key] = rect
                    axes.add_patch(rect)
                    text_dict[key] = axes.text(ms.x, ms.y + 3, str(key), horizontalalignment='center', zorder=50, fontsize='xx-large')
                    
                else:
                    width = value.width
                    length = value.length
                    patches_dict[key].set_xy(polygon_xy_from_motionstate(ms, width, length))
                    if (key in visible_id):
                        patches_dict[key].set_zorder(40)   
                    elif (key not in visible_id):
                        patches_dict[key].set_zorder(20)    
                    text_dict[key].set_position((ms.x, ms.y + 3))
            
                    # Consider all the visible drivers.
                    if ((key in visible_id) and (key != ego_id)):
                        patches_dict[key].set_color('cyan')

                        colormap = ListedColormap(['white','black','gray'])
                        res = 1.0            
                        if key in driver_sensor_state_data.keys():
                            for state in driver_sensor_state_data[key]:
                                if state[0] == timestamp:
                                    if key in driver_sensor_state.keys():

                                        # Make sure that the states are contiguous.
                                        if state[0] - driver_sensor_state[key][-1,0] == 100:
                                            driver_sensor_state[key] = np.concatenate((driver_sensor_state[key], np.reshape(state, (1,-1))))
                                        else:
                                            driver_sensor_state[key] = np.reshape(state, (1,-1))
                                    else:
                                        driver_sensor_state[key] = np.reshape(state, (1,-1))
                                    break
                    elif ((key not in visible_id) and (key != ego_id)):
                        patches_dict[key].set_color('#1f77b4')
                        if key in driver_sensor_state_dict.keys():
                            driver_sensor_state_dict[key].remove()
                            del driver_sensor_state_dict[key]
            
            else:
                if key in patches_dict:
                    patches_dict[key].remove()
                    patches_dict.pop(key)
                    text_dict[key].remove()
                    text_dict.pop(key)
                    if key in driver_sensor_state_dict.keys():
                        driver_sensor_state_dict[key].remove()
                        del driver_sensor_state_dict[key]

    # Plot the pedestrians.
    if pedest_dict is not None:

        for key, value in pedest_dict.items():
            assert isinstance(value, Track)
            if value.time_stamp_ms_first <= timestamp <= value.time_stamp_ms_last:
                ms = value.motion_states[timestamp]
                assert isinstance(ms, MotionState)

                if key not in patches_dict:
                    width = 1.5
                    length = 1.5

                    rect = matplotlib.patches.Polygon(polygon_xy_from_motionstate_pedest(ms, width, length), closed=True,
                                                      zorder=20, color='red')
                    patches_dict[key] = rect
                    axes.add_patch(rect)
                    text_dict[key] = axes.text(ms.x, ms.y + 3, str(key), horizontalalignment='center', zorder=50, fontsize='xx-large')
                else:
                    width = 1.5
                    length = 1.5
                    patches_dict[key].set_xy(polygon_xy_from_motionstate_pedest(ms, width, length))
                    text_dict[key].set_position((ms.x, ms.y + 3))
            else:
                if key in patches_dict:
                    patches_dict[key].remove()
                    patches_dict.pop(key)
                    text_dict[key].remove()
                    text_dict.pop(key)

