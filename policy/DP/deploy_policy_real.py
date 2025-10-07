import numpy as np
from .dp_model import DP
import yaml
import time

def encode_obs(observation):
    # fisheye_camera = (np.moveaxis(observation["image"]["fisheye_rgb"], -1, 0) / 255)
    left_camera = (np.moveaxis(observation["image"]["left_rgb"], -1, 0) / 255)
    front_camera = (np.moveaxis(observation["image"]["front_rgb"], -1, 0) / 255)
    obs = dict(
        # fisheye_camera=fisheye_camera,
        left_camera=left_camera,
        front_camera=front_camera,
    )
    qpos = np.concatenate([observation["state"]["arm_end_pose"], observation["state"]["arm_joint"], observation["state"]["gripper_rad"]], axis=-1).astype(np.float32)
    obs["agent_pos"] = qpos
    return obs


def get_model(usr_args):
    ckpt_file = f"./policy/DP/checkpoints/{usr_args['task_name']}-{usr_args['ckpt_setting']}-{usr_args['expert_data_num']}-{usr_args['seed']}/{usr_args['checkpoint_num']}.ckpt"
    action_dim = 8 # 2 gripper
    
    load_config_path = f'./policy/DP/diffusion_policy/config/robot_dp_{action_dim}.yaml'
    with open(load_config_path, "r", encoding="utf-8") as f:
        model_training_config = yaml.safe_load(f)
    
    n_obs_steps = model_training_config['n_obs_steps']
    n_action_steps = model_training_config['n_action_steps']
    
    return DP(ckpt_file, n_obs_steps=n_obs_steps, n_action_steps=n_action_steps)


def eval(TASK_ENV, model, observation):
    """
    TASK_ENV: Task Environment Class, you can use this class to interact with the environment
    model: The model from 'get_model()' function
    observation: The observation about the environment
    """
    obs = encode_obs(observation)
    # instruction = TASK_ENV.get_instruction()

    # ======== Get Action ========
    actions = model.get_action(obs)
    next_time = time.time()
    for action in actions:
        TASK_ENV.take_action(action)
        observation = TASK_ENV.get_obs()
        obs = encode_obs(observation)
        model.update_obs(obs)
        
        
        next_time += TASK_ENV.target_interval-0.0
        sleep_time = next_time - time.time()
        if sleep_time > 0:
            time.sleep(sleep_time)
        else:
            TASK_ENV.logger.warning("Warning: Loop overrun, running behind schedule")
            next_time = time.time()

def reset_model(model):
    model.reset_obs()