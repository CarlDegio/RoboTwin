import sys
import os
import subprocess

sys.path.append("./")
sys.path.append(f"./policy")
sys.path.append("./description/utils")
from envs import CONFIGS_PATH
from envs.utils.create_actor import UnStableError

import numpy as np
from pathlib import Path
from collections import deque
import traceback

import yaml
from datetime import datetime
import importlib
import argparse
import pdb

current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(current_file_path)
from hardware_interface.common_curobo import pika_arm, pika_sense, calibrate_arm, disable_piper_arm
import hardware_interface.common_curobo as common
import hardware_interface.math_utils as math_utils
import time
import logging
import cv2


class PikaEnv():
    def __init__(self):
        self.logger = logging.getLogger("pika_env")
        self.target_interval = 0.1
        self.take_action_cnt = 0
        self.step_lim = 300
        
        self.arm = pika_arm(fisheye_camera_index=8, gripper_port='/dev/ttyUSB0')
        self.arm.reset_arm_and_gripper_record()
        
        time.sleep(3)
        print("arm pose", self.arm.get_arm_pose())
        print("arm pose offset", self.arm.get_arm_pose_offset())
        
        
        for _ in range(5):
            fish_image = self.arm.get_fisheye_rgb()
            realsense_image = self.arm.get_realsense_rgb()
        print("camera check done, image shape:", fish_image.shape, realsense_image.shape)
        
        self.logger.info("\033[0;32m"+"waiting for tele start..."+"\033[0m")
            
        time.sleep(0.5)
        
    def get_obs(self):
        fish_image = self.arm.get_fisheye_rgb()
        fish_image = cv2.resize(fish_image, (320, 240))
        arm_joint_state = self.arm.get_joint_position()
        gripper_rad_state = self.arm.get_gripper_msg()[1].reshape(-1, 1)
        arm_end_pose_state = np.concatenate(self.arm.get_arm_pose())
        dict_obs = {
            "image": fish_image,
            "state": {
                "arm_joint": arm_joint_state,
                "gripper_rad": gripper_rad_state,
                "arm_end_pose": arm_end_pose_state,
            }
        }
        return dict_obs
    
    def take_action(self, action):
        assert action.shape == (8,), "action shape must be (8,)"
        command_position, command_rotation_quat, command_gripper_rad = action[:3], action[3:7], action[7]
        command_rotation_quat = command_rotation_quat / np.linalg.norm(command_rotation_quat)
        
        self.arm.control_arm_end_pose(command_position, command_rotation_quat)
        self.arm.control_gripper(command_gripper_rad)
        self.take_action_cnt += 1
        
    def safe_check(self, command_position, command_rotation_quat, command_gripper_rad):
        pass
        
    def close_env(self):
        self.arm.disconnect()
        
    def __del__(self):
        self.close_env()
        
    


def eval_function_decorator(policy_name, model_name):
    try:
        policy_model = importlib.import_module(policy_name)
        return getattr(policy_model, model_name)
    except ImportError as e:
        raise e

def get_camera_config(camera_type):
    camera_config_path = os.path.join(parent_directory, "../task_config/_camera_config.yml")

    assert os.path.isfile(camera_config_path), "task config file is missing"

    with open(camera_config_path, "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)

    assert camera_type in args, f"camera {camera_type} is not defined"
    return args[camera_type]


def main(usr_args):
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    task_name = usr_args["task_name"]
    task_config = usr_args["task_config"]
    ckpt_setting = usr_args["ckpt_setting"]
    # checkpoint_num = usr_args['checkpoint_num']
    policy_name = usr_args["policy_name"]
    instruction_type = usr_args["instruction_type"]
    save_dir = None
    video_save_dir = None
    video_size = None

    get_model = eval_function_decorator(policy_name, "get_model")

    with open(f"./task_config/{task_config}.yml", "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)

    args['task_name'] = task_name
    args["task_config"] = task_config
    args["ckpt_setting"] = ckpt_setting

    save_dir = Path(f"eval_result/{task_name}/{policy_name}/{task_config}/{ckpt_setting}/{current_time}")
    save_dir.mkdir(parents=True, exist_ok=True)

    if args["eval_video_log"]:
        video_save_dir = save_dir
        camera_config = get_camera_config(args["camera"]["head_camera_type"])
        video_size = str(camera_config["w"]) + "x" + str(camera_config["h"])
        video_save_dir.mkdir(parents=True, exist_ok=True)
        args["eval_video_save_dir"] = video_save_dir

    TASK_ENV = PikaEnv()
    args["policy_name"] = policy_name

    seed = usr_args["seed"]

    st_seed = 100000 * (1 + seed)
    suc_nums = []
    test_num = 1

    model = get_model(usr_args)
    st_seed = eval_policy(task_name,
                                   TASK_ENV,
                                   args,
                                   model,
                                   st_seed,
                                   test_num=test_num,
                                   video_size=video_size,
                                   instruction_type=instruction_type)
    # return task_reward


def eval_policy(task_name,
                TASK_ENV:PikaEnv,
                args,
                model,
                st_seed,
                test_num=1,
                video_size=None,
                instruction_type=None):
    print(f"\033[34mTask Name: {args['task_name']}\033[0m")
    print(f"\033[34mPolicy Name: {args['policy_name']}\033[0m")

    succ_seed = 0

    policy_name = args["policy_name"]
    eval_func = eval_function_decorator(policy_name, "eval")
    reset_func = eval_function_decorator(policy_name, "reset_model")

    now_seed = st_seed

    args["eval_mode"] = True

    while succ_seed < test_num:
        render_freq = args["render_freq"]
        args["render_freq"] = 0
        args["render_freq"] = render_freq

        succ = False
        reset_func(model)
        
        try:
            while TASK_ENV.take_action_cnt < TASK_ENV.step_lim:
                start_time = time.time()
                observation = TASK_ENV.get_obs()
                eval_func(TASK_ENV, model, observation)
                
                next_time = start_time + TASK_ENV.target_interval
                sleep_time = next_time - time.time()
                if sleep_time > 0:
                    time.sleep(sleep_time)
                else:
                    TASK_ENV.logger.warning("Warning: Loop overrun, running behind schedule")
                    next_time = time.time()
        except Exception as e:
            TASK_ENV.logger.error(f"try-catch Error: {e}")
            TASK_ENV.arm.reset_arm_and_gripper_record()
        finally:
            TASK_ENV.close_env()

        now_seed += 1

    return now_seed


def parse_args_and_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--overrides", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Parse overrides
    def parse_override_pairs(pairs):
        override_dict = {}
        for i in range(0, len(pairs), 2):
            key = pairs[i].lstrip("--")
            value = pairs[i + 1]
            try:
                value = eval(value)
            except:
                pass
            override_dict[key] = value
        return override_dict

    if args.overrides:
        overrides = parse_override_pairs(args.overrides)
        config.update(overrides)

    return config


if __name__ == "__main__":

    usr_args = parse_args_and_config()

    main(usr_args)
