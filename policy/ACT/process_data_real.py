import sys

sys.path.append("./policy/ACT/")

import os
import h5py
import numpy as np
import pickle
import cv2
import argparse
import pdb
import json


def load_hdf5(dataset_path):
    if not os.path.isfile(dataset_path):
        print(f"Dataset does not exist at \n{dataset_path}\n")
        exit()

    with h5py.File(dataset_path, "r") as root:
        action_gripper_rad, action_arm_end_pose = (
            root["/action/gripper_rad"][()],
            root["/action/arm_end_pose"][()],
        )
        action_gripper_rad = action_gripper_rad.reshape(-1, 1)
        obs_gripper_rad, obs_arm_joint, obs_arm_end_pose = (
            root["/state/gripper_rad"][()],
            root["/state/arm_joint"][()],
            root["/state/arm_end_pose"][()],
        )
        obs_gripper_rad = obs_gripper_rad.reshape(-1, 1)
        image_dict = {'fisheye_rgb': root["/image/fisheye_rgb"][()], 'left_rgb': root["/image/left_rgb"][()], 'front_rgb': root["/image/front_rgb"][()]}

    return action_gripper_rad, action_arm_end_pose, obs_gripper_rad, obs_arm_joint, obs_arm_end_pose ,image_dict



def data_transform(path, episode_num, save_path):
    begin = 0
    floders = os.listdir(path)
    assert episode_num <= len(floders), "data num not enough"

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for i in range(episode_num):
        action_gripper_rad, action_arm_end_pose, obs_gripper_rad, obs_arm_joint, obs_arm_end_pose ,image_dict = \
        (load_hdf5(os.path.join(path, f"episode_{i}.hdf5")))
        qpos = np.concatenate([obs_arm_end_pose, obs_arm_joint, obs_gripper_rad], axis=-1).astype(np.float32)
        actions = np.concatenate([action_arm_end_pose, action_gripper_rad], axis=-1).astype(np.float32)
        cam_fisheye = []
        cam_left = []
        cam_front = []
        # left_arm_dim = []
        # right_arm_dim = []

        for j in range(0, action_gripper_rad.shape[0]):
            camera_fisheye_bits = image_dict["fisheye_rgb"][j]
            camera_fisheye = cv2.imdecode(np.frombuffer(camera_fisheye_bits, np.uint8), cv2.IMREAD_COLOR)
            camera_fisheye_resized = cv2.resize(camera_fisheye, (640, 480))
            cam_fisheye.append(camera_fisheye_resized)

            camera_left_bits = image_dict["left_rgb"][j]
            camera_left = cv2.imdecode(np.frombuffer(camera_left_bits, np.uint8), cv2.IMREAD_COLOR)
            camera_left_resized = cv2.resize(camera_left, (640, 480))
            cam_left.append(camera_left_resized)

            camera_front_bits = image_dict["front_rgb"][j]
            camera_front = cv2.imdecode(np.frombuffer(camera_front_bits, np.uint8), cv2.IMREAD_COLOR)
            camera_front_resized = cv2.resize(camera_front, (640, 480))
            cam_front.append(camera_front_resized)

        hdf5path = os.path.join(save_path, f"episode_{i}.hdf5")

        with h5py.File(hdf5path, "w") as f:
            f.create_dataset("action", data=np.array(actions))
            obs = f.create_group("observations")
            obs.create_dataset("qpos", data=np.array(qpos))
            # obs.create_dataset("left_arm_dim", data=np.array(left_arm_dim))
            # obs.create_dataset("right_arm_dim", data=np.array(right_arm_dim))
            image = obs.create_group("images")
            image.create_dataset("fisheye_rgb", data=np.stack(cam_fisheye), dtype=np.uint8)
            image.create_dataset("left_rgb", data=np.stack(cam_left), dtype=np.uint8)
            image.create_dataset("front_rgb", data=np.stack(cam_front), dtype=np.uint8)

        begin += 1
        print(f"proccess {i} success!")

    return begin


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some episodes.")
    parser.add_argument(
        "task_name",
        type=str,
        default="stack_block",
        help="The name of the task (e.g., adjust_bottle)",
    )
    parser.add_argument("task_config", type=str, default="demo_randomized")
    parser.add_argument("expert_data_num", type=int, default=5)

    args = parser.parse_args()

    task_name = args.task_name
    task_config = args.task_config
    expert_data_num = args.expert_data_num

    begin = 0
    begin = data_transform(
        os.path.join("../../data/", task_name, task_config, 'data'),
        expert_data_num,
        f"processed_data/sim-{task_name}/{task_config}-{expert_data_num}",
    )

    SIM_TASK_CONFIGS_PATH = "./SIM_TASK_CONFIGS.json"

    try:
        with open(SIM_TASK_CONFIGS_PATH, "r") as f:
            SIM_TASK_CONFIGS = json.load(f)
    except Exception:
        SIM_TASK_CONFIGS = {}

    SIM_TASK_CONFIGS[f"sim-{task_name}-{task_config}-{expert_data_num}"] = {
        "dataset_dir": f"./processed_data/sim-{task_name}/{task_config}-{expert_data_num}",
        "num_episodes": expert_data_num,
        "episode_len": 300,
        "camera_names": ["fisheye_rgb", "left_rgb", "front_rgb"],
    }

    with open(SIM_TASK_CONFIGS_PATH, "w") as f:
        json.dump(SIM_TASK_CONFIGS, f, indent=4)
