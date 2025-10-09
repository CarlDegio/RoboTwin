import pickle, os
import numpy as np
import pdb
from copy import deepcopy
import zarr
import shutil
import argparse
import yaml
import cv2
import h5py


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
        # 代码中用for key按fisheye fron left遍历相机
    return action_gripper_rad, action_arm_end_pose, obs_gripper_rad, obs_arm_joint, obs_arm_end_pose ,image_dict


def main():
    parser = argparse.ArgumentParser(description="Process some episodes.")
    parser.add_argument(
        "task_name",
        type=str,
        help="The name of the task (e.g., beat_block_hammer)",
    )
    parser.add_argument("task_config", type=str)
    parser.add_argument(
        "expert_data_num",
        type=int,
        help="Number of episodes to process (e.g., 50)",
    )
    args = parser.parse_args()

    task_name = args.task_name
    num = args.expert_data_num
    task_config = args.task_config

    load_dir = "../../data/" + str(task_name) + "/" + str(task_config)

    total_count = 0

    save_dir = f"./data/{task_name}-{task_config}-{num}.zarr"

    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)

    current_ep = 0

    zarr_root = zarr.group(save_dir)
    zarr_data = zarr_root.create_group("data")
    zarr_meta = zarr_root.create_group("meta")

    fisheye_camera_arrays = []
    left_camera_arrays = []
    front_camera_arrays = []
    episode_ends_arrays, state_arrays, action_arrays, val_arrays = (
        [],
        [],
        [],
        [],
    )

    while current_ep < num:
        print(f"processing episode: {current_ep + 1} / {num}", end="\r")

        load_path = os.path.join(load_dir, f"data/episode_{current_ep}.hdf5")
        action_gripper_rad, action_arm_end_pose, obs_gripper_rad, obs_arm_joint, obs_arm_end_pose ,image_dict_all= load_hdf5(load_path)

        for j in range(0, action_gripper_rad.shape[0]):

            fisheye_img_bit = image_dict_all["fisheye_rgb"][j]
            left_img_bit = image_dict_all["left_rgb"][j]
            front_img_bit = image_dict_all["front_rgb"][j]
            
            joint_state = np.concatenate([obs_arm_end_pose[j], obs_arm_joint[j], obs_gripper_rad[j]], axis=-1)
            action = np.concatenate([action_arm_end_pose[j], action_gripper_rad[j]], axis=-1)

            fisheye_img = cv2.imdecode(np.frombuffer(fisheye_img_bit, np.uint8), cv2.IMREAD_COLOR)
            left_img = cv2.imdecode(np.frombuffer(left_img_bit, np.uint8), cv2.IMREAD_COLOR)
            front_img = cv2.imdecode(np.frombuffer(front_img_bit, np.uint8), cv2.IMREAD_COLOR)
            
            # fisheye_img = cv2.resize(fisheye_img, (224, 224))
            # left_img = cv2.resize(left_img, (224, 224))
            # front_img = cv2.resize(front_img, (224, 224))
            
            fisheye_camera_arrays.append(fisheye_img)
            left_camera_arrays.append(left_img)
            front_camera_arrays.append(front_img)
            
            state_arrays.append(joint_state)
            action_arrays.append(action)
            val_arrays.append(np.array([j/action_gripper_rad.shape[0]]))

        current_ep += 1
        total_count += action_gripper_rad.shape[0]
        episode_ends_arrays.append(total_count)

    print()
    episode_ends_arrays = np.array(episode_ends_arrays)
    # action_arrays = np.array(action_arrays)
    state_arrays = np.array(state_arrays)
    fisheye_camera_arrays = np.array(fisheye_camera_arrays)
    left_camera_arrays = np.array(left_camera_arrays)
    front_camera_arrays = np.array(front_camera_arrays)
    action_arrays = np.array(action_arrays)
    val_arrays = np.array(val_arrays)
    
    fisheye_camera_arrays = np.moveaxis(fisheye_camera_arrays, -1, 1)  # NHWC -> NCHW
    left_camera_arrays = np.moveaxis(left_camera_arrays, -1, 1)  # NHWC -> NCHW
    front_camera_arrays = np.moveaxis(front_camera_arrays, -1, 1)  # NHWC -> NCHW
    
    compressor = zarr.Blosc(cname="zstd", clevel=3, shuffle=1)
    # action_chunk_size = (100, action_arrays.shape[1])
    state_chunk_size = (100, state_arrays.shape[1])
    joint_chunk_size = (100, action_arrays.shape[1])
    fisheye_camera_chunk_size = (100, *fisheye_camera_arrays.shape[1:])
    left_camera_chunk_size = (100, *left_camera_arrays.shape[1:])
    front_camera_chunk_size = (100, *front_camera_arrays.shape[1:])
    val_chunk_size = (100, val_arrays.shape[1])
    zarr_data.create_dataset(
        "fisheye_camera",
        data=fisheye_camera_arrays,
        chunks=fisheye_camera_chunk_size,
        overwrite=True,
        compressor=compressor,
    )
    zarr_data.create_dataset(
        "left_camera",
        data=left_camera_arrays,
        chunks=left_camera_chunk_size,
        overwrite=True,
        compressor=compressor,
    )
    zarr_data.create_dataset(
        "front_camera",
        data=front_camera_arrays,
        chunks=front_camera_chunk_size,
        overwrite=True,
        compressor=compressor,
    )
    zarr_data.create_dataset(
        "state",
        data=state_arrays,
        chunks=state_chunk_size,
        dtype="float32",
        overwrite=True,
        compressor=compressor,
    )
    zarr_data.create_dataset(
        "action",
        data=action_arrays,
        chunks=joint_chunk_size,
        dtype="float32",
        overwrite=True,
        compressor=compressor,
    )
    zarr_meta.create_dataset(
        "episode_ends",
        data=episode_ends_arrays,
        dtype="int64",
        overwrite=True,
        compressor=compressor,
    )
    zarr_data.create_dataset(
        "val",
        data=val_arrays,
        chunks=val_chunk_size,
        dtype="float32",
        overwrite=True,
        compressor=compressor,
    )


if __name__ == "__main__":
    main()
