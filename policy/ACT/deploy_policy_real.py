import sys
import numpy as np
import torch
import os
import pickle
import cv2
import time  # Add import for timestamp
import h5py  # Add import for HDF5
from datetime import datetime  # Add import for datetime formatting
from .act_policy import ACT
import copy
from argparse import Namespace


def encode_obs(observation):

    fisheye_cam = observation["image"]["fisheye_rgb"]
    fisheye_cam = np.moveaxis(fisheye_cam, -1, 0) / 255.0
    qpos = np.concatenate([observation["state"]["arm_end_pose"], observation["state"]["arm_joint"], observation["state"]["gripper_rad"]], axis=-1).astype(np.float32)
    return {
        "fisheye_rgb": fisheye_cam,
        "qpos": qpos,
    }


def get_model(usr_args):
    return ACT(usr_args, Namespace(**usr_args))


def eval(TASK_ENV, model, observation):
    next_time = time.time()
    obs = encode_obs(observation)
    # instruction = TASK_ENV.get_instruction()

    # Get action from model
    actions = model.get_action(obs)
    for action in actions:
        TASK_ENV.take_action(action)
        observation = TASK_ENV.get_obs()
        
        next_time += TASK_ENV.target_interval
        sleep_time = next_time - time.time()
        if sleep_time > 0:
            time.sleep(sleep_time)
        else:
            TASK_ENV.logger.warning("Warning: Loop overrun, running behind schedule")
            next_time = time.time()
    return observation


def reset_model(model):
    # Reset temporal aggregation state if enabled
    if model.temporal_agg:
        model.all_time_actions = torch.zeros([
            model.max_timesteps,
            model.max_timesteps + model.num_queries,
            model.state_dim,
        ]).to(model.device)
        model.t = 0
        print("Reset temporal aggregation state")
    else:
        model.t = 0
