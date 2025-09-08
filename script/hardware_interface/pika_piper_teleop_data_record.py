from hardware_interface.common_curobo import pika_arm, pika_sense, calibrate_arm
import numpy as np
import hardware_interface.common_curobo as common
import hardware_interface.math_utils as math_utils
import time
import logging
import cv2

from curobo.util import logger
logger.setup_logger(level="error", logger_name="curobo")

logger = logging.getLogger("pika_teleop")

def close_arm():
    for _ in range(10):
        logger.warning("\033[0;33m"+"close arm"+"\033[0m")
        time.sleep(0.5)
    common.disable_piper_arm()
    
def calibrate_arm():
    common.calibrate_arm()
    
def main():
    arm = pika_arm(fisheye_camera_index=8, gripper_port='/dev/ttyUSB0')
    arm.reset_arm_and_gripper_record()
    time.sleep(3)
    print("arm pose", arm.get_arm_pose())
    print("arm pose offset", arm.get_arm_pose_offset())
    
    
    for _ in range(5):
        fish_image = arm.get_fisheye_rgb()
        realsense_image = arm.get_realsense_rgb()
    print("camera check done, image shape:", fish_image.shape, realsense_image.shape)
    
    logger.info("\033[0;32m"+"waiting for tele start..."+"\033[0m")
        
    time.sleep(0.5)
    
    data_dict = {}
    data_dict["image"] = {"fisheye_rgb": [], "realsense_rgb": []}
    data_dict["action"] = {"arm_end_pose": [], "gripper_rad": []}
    data_dict["state"] = {"arm_joint": [], "gripper_rad": [], "arm_end_pose": []}
    
    target_interval = 0.1
    next_time = time.time()
    for i in range(300):
        
        fish_image = arm.get_fisheye_rgb()
        realsense_image = arm.get_realsense_rgb()
        fish_image = cv2.resize(fish_image, (320, 240))
        realsense_image = cv2.resize(realsense_image, (320, 240))
        arm_joint_state = arm.get_joint_position()
        gripper_rad_state = arm.get_gripper_msg()[1]
        arm_end_pose_state = np.concatenate(arm.get_arm_pose())
        
        command_gripper_rad = sense.get_sense_msg()[1]
        command_position, command_rotation_quat = sense.get_vive_relative_pose()
        
        fix_rotation_quat = [np.cos(np.pi/4), 0, np.sin(np.pi/4), 0]
        
        command_position = np.matmul(math_utils.make_matrix_from_quaternion(math_utils.quaternion_inv(fix_rotation_quat)), command_position)
        command_position = command_position
        
        arm_end_position_command, arm_end_quat_command, gripper_rad_command = arm.control_by_tele(command_position, command_rotation_quat, command_gripper_rad)
        
        arm_end_pose_command = np.concatenate([arm_end_position_command, arm_end_quat_command])
        
        
        data_dict["image"]["fisheye_rgb"].append(fish_image)
        data_dict["image"]["realsense_rgb"].append(realsense_image)
        data_dict["action"]["arm_end_pose"].append(arm_end_pose_command)
        data_dict["action"]["gripper_rad"].append(gripper_rad_command)
        data_dict["state"]["arm_joint"].append(arm_joint_state)
        data_dict["state"]["gripper_rad"].append(gripper_rad_state)
        data_dict["state"]["arm_end_pose"].append(arm_end_pose_state)
        
        next_time += target_interval
        sleep_time = next_time - time.time()
        if sleep_time > 0:
            time.sleep(sleep_time)
        else:
            logger.warning("Warning: Loop overrun, running behind schedule")
            next_time = time.time()  # 重置时间戳，避免累积误差
        
    
    arm.disconnect()


if __name__ == "__main__":
    main()
    # close_arm()
    # calibrate_arm()