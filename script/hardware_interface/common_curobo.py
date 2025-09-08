from pika.gripper import Gripper
import cv2
from piper_sdk import C_PiperInterface_V2
import time
import numpy as np
from pika import sense
from .curobo_planner import CuroboPlanner
from .math_utils import make_matrix_from_quaternion, quaternion_inv, quaternion_mul
import copy


def enable_piper_arm(arm_port="can0"):
    piper_arm = C_PiperInterface_V2(arm_port)
    piper_arm.ConnectPort()
    while( not piper_arm.EnablePiper()):
        time.sleep(0.01)
    print("成功连接到 Pika Arm 设备")
    
def disable_piper_arm(arm_port="can0"):
    piper_arm = C_PiperInterface_V2(arm_port)
    piper_arm.ConnectPort()
    while(piper_arm.DisablePiper()):
        pass
        time.sleep(0.01)
    print("成功断开 Pika Arm 设备")

def calibrate_arm(arm_port="can0"):
    piper_arm = C_PiperInterface_V2(arm_port)
    piper_arm.ConnectPort()
    piper_arm.JointConfig(7,set_zero=0xAE,clear_err=0xAE)
    print("成功校准 Pika Arm 设备")
    

class pika_arm:
    def __init__(self, arm_port="can0", gripper_port="/dev/ttyUSB0", camera_param=(640, 480, 30, 100), fisheye_camera_index=0, realsense_serial_number='230322275842'):
        self.arm_port = arm_port
        self.gripper_port = gripper_port
        self.camera_param = camera_param
        self.fisheye_camera_index = fisheye_camera_index
        self.realsense_serial_number = realsense_serial_number
        self.gripper_offset = np.array([0.0, 0.0, 190.34])/1000 # m
        
        self.gripper = Gripper(self.gripper_port)
        if not self.gripper.connect():
            raise Exception("连接 Pika Gripper 设备失败，请检查设备连接和串口路径")
        print("成功连接到 Pika Gripper 设备")
        
        if not self.gripper.enable():
            raise Exception("启用 Pika Gripper 设备失败，请检查设备连接和串口路径")
        print("成功启用 Pika Gripper 设备")
        
        self.piper_arm = C_PiperInterface_V2(self.arm_port)
        self.piper_arm.ConnectPort()
        while( not self.piper_arm.EnablePiper()):
            time.sleep(0.01)
        print("成功连接到 Pika Arm 设备")
        
        self.gripper.set_camera_param(*self.camera_param)
        self.gripper.set_fisheye_camera_index(self.fisheye_camera_index)
        self.gripper.set_realsense_serial_number(self.realsense_serial_number)
        self.fisheye_camera = self.gripper.get_fisheye_camera()
        self.realsense_camera = self.gripper.get_realsense_camera()

        self.scale_factor = 1000
        self.position0 = None
        self.rotation0_quat = None
        
        self.smooth_buffer = []
        self.smooth_weight = np.ones(10) / 10
        
        self.planner = CuroboPlanner(active_joints_name=["joint1","joint2","joint3","joint4","joint5","joint6"], yml_path='/home/lzh/PycharmProjects/pika_frame/urdf/piper/curobo_tmp.yml')
        self.last_command_quat = None
        
        time.sleep(1)
        print("Pika系统初始化完成")
    
    def init_curobo_planner(self):
        init_input = np.array([0.0,0.0,0.0,0.0,0.0,0.0])
        init_fk_result = self.planner.fk(init_input)
        init_ik_pose = np.concatenate([init_fk_result.ee_position.cpu().numpy()[0], init_fk_result.ee_quaternion.cpu().numpy()[0]])
        init_ik_result = self.planner.ik(init_ik_pose, current_joint_angle=init_input)
    
    def on_record_start(self):
        self.position0, self.rotation0_quat = self.get_arm_pose_offset()
    
    def get_fisheye_rgb(self):
        success, frame = self.fisheye_camera.get_frame()
        if success == False or frame is None:
            raise Exception("获取鱼眼相机图像失败")
        
        return frame

    def get_realsense_rgb(self):
        success, frame = self.realsense_camera.get_color_frame()
        if success == False or frame is None:
            raise Exception("获取Realsense相机图像失败")
        
        return frame

    def get_realsense_depth(self):
        success, frame = self.realsense_camera.get_depth_frame()
        if success == False or frame is None:
            raise Exception("获取Realsense深度图像失败")
        
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(frame, alpha=0.03), cv2.COLORMAP_JET)
        return depth_colormap

    def get_joint_position(self):
        # 返回弧度
        joint_msgs = self.piper_arm.GetArmJointMsgs()
        joint1 = joint_msgs.joint_state.joint_1/self.scale_factor
        joint2 = joint_msgs.joint_state.joint_2/self.scale_factor
        joint3 = joint_msgs.joint_state.joint_3/self.scale_factor
        joint4 = joint_msgs.joint_state.joint_4/self.scale_factor
        joint5 = joint_msgs.joint_state.joint_5/self.scale_factor
        joint6 = joint_msgs.joint_state.joint_6/self.scale_factor
        joints_deg = np.array([joint1, joint2, joint3, joint4, joint5, joint6])
        joints_rad = np.deg2rad(joints_deg)
        return joints_rad

    def get_arm_pose(self):
        """
        以夹爪朝前为Z系,朝下为X系, 即最上面时与世界系对齐
        """
        joints_rad = self.get_joint_position()
        fk_result = self.planner.fk(joints_rad)
        position = fk_result.ee_position.cpu().numpy()[0]
        quat = fk_result.ee_quaternion.cpu().numpy()[0]
        return position, quat
    
    def get_arm_pose_offset(self):
        position, quat = self.get_arm_pose()
        rotation_matrix = make_matrix_from_quaternion(quat)
        position = np.matmul(rotation_matrix, self.gripper_offset) + position
        
        return position, quat

    def control_arm_joint(self, joint1, joint2, joint3, joint4, joint5, joint6):
        factor = 57295.7795 #1000*180/3.1415926
        joint1 = round(joint1*factor)
        joint2 = round(joint2*factor)
        joint3 = round(joint3*factor)
        joint4 = round(joint4*factor)
        joint5 = round(joint5*factor)
        joint6 = round(joint6*factor)
        self.piper_arm.MotionCtrl_2(0x01, 0x01, 10, 0x00)
        self.piper_arm.JointCtrl(joint1, joint2, joint3, joint4, joint5, joint6)
        
    
    def control_arm_end_pose(self, position, quat):
        joints_rad = self.get_joint_position()
        target_pose = np.concatenate([position, quat])
        
        ik_result = self.planner.ik(target_pose, current_joint_angle=joints_rad)
        joints_rad = ik_result.js_solution.position.cpu().numpy()[0,0]
        if not np.isnan(joints_rad[0]):
            self.control_arm_joint(joints_rad[0], joints_rad[1], joints_rad[2], joints_rad[3], joints_rad[4], joints_rad[5])
        
        
    def get_gripper_msg(self):
        current_gripper_distance  = self.gripper.get_gripper_distance()
        current_pos_rad = self.gripper.get_motor_position()
        current_pos_deg = current_pos_rad * 180 / np.pi
        return current_gripper_distance, current_pos_rad, current_pos_deg
        
    def control_gripper(self, gripper_rad):
        max_gripper_rad = 90*np.pi/180
        gripper_rad = np.clip(gripper_rad, 0.0, max_gripper_rad)
        self.gripper.set_motor_angle(gripper_rad)
        
    def control_arm_gripper_end_pose(self, position, quat, gripper_rad):    
        self.control_arm_end_pose(position, quat)
        self.control_gripper(gripper_rad)
        
    def reset_arm_and_gripper_zero(self):
        factor = 57295.7795 #1000*180/3.1415926
        position = [0,0,0,0,0,0]
        gripper_rad = 90*np.pi/180
        
        joint_0 = round(position[0]*factor)
        joint_1 = round(position[1]*factor)
        joint_2 = round(position[2]*factor)
        joint_3 = round(position[3]*factor)
        joint_4 = round(position[4]*factor)
        joint_5 = round(position[5]*factor)
        self.piper_arm.ModeCtrl(0x01, 0x01, 20, 0x00)
        self.piper_arm.JointCtrl(joint_0, joint_1, joint_2, joint_3, joint_4, joint_5)
        self.gripper.set_motor_angle(gripper_rad)
        
    def reset_arm_and_gripper_record(self):
        factor = 57295.7795 #1000*180/3.1415926
        position = [0.0,0.5,-0.6,0.0,0.5,0.0,0.0]
        gripper_rad = 90*np.pi/180
        
        joint_0 = round(position[0]*factor)
        joint_1 = round(position[1]*factor)
        joint_2 = round(position[2]*factor)
        joint_3 = round(position[3]*factor)
        joint_4 = round(position[4]*factor)
        joint_5 = round(position[5]*factor)
        self.piper_arm.ModeCtrl(0x01, 0x01, 10, 0x00)
        self.piper_arm.JointCtrl(joint_0, joint_1, joint_2, joint_3, joint_4, joint_5)
        self.gripper.set_motor_angle(gripper_rad)
    
    def smooth_command(self, tele_position, tele_rotation_quaternion):
        if len(self.smooth_buffer) < len(self.smooth_weight):
            self.smooth_buffer= []
            for _ in range(len(self.smooth_weight)):
                self.smooth_buffer.append([tele_position, tele_rotation_quaternion])
        else:
            self.smooth_buffer.pop(0)
            self.smooth_buffer.append([tele_position, tele_rotation_quaternion])
        
        tele_position = 0
        tele_rotation_quaternion = 0
        for i in range(len(self.smooth_weight)):
            tele_position += self.smooth_weight[i]*self.smooth_buffer[i][0]
            tele_rotation_quaternion += self.smooth_weight[i]*self.smooth_buffer[i][1]
        return tele_position, tele_rotation_quaternion
    
    def control_by_tele(self, tele_position, tele_rotation_quaternion, gripper_rad):
        if self.position0 is None or self.rotation0_quat is None:
            raise Exception("请先调用on_record_start()方法")
        
        tele_position, tele_rotation_quaternion = self.smooth_command(tele_position, tele_rotation_quaternion)
        
        command_quat = quaternion_mul(tele_rotation_quaternion, self.rotation0_quat)
        if self.last_command_quat is not None:
            if np.dot(self.last_command_quat, command_quat) < 0:
                command_quat = quaternion_mul(command_quat, [-1, 0, 0, 0])
        self.last_command_quat = command_quat
        
        command_rotation_matrix = make_matrix_from_quaternion(command_quat)
        
        command_position = np.matmul(make_matrix_from_quaternion(self.rotation0_quat), tele_position) + self.position0
        
        # 以夹爪的offset做控制
        command_position = command_position - np.matmul(command_rotation_matrix, self.gripper_offset)
        
        # print("command_quat", command_quat)
        # print("command_position", command_position)
        self.control_arm_end_pose(command_position, command_quat)
        self.control_gripper(gripper_rad)
        
        return command_position, command_quat, gripper_rad
        
    def disconnect(self):
        self.gripper.disable()
        self.fisheye_camera.disconnect()
        self.realsense_camera.disconnect()


class pika_sense:
    def __init__(self, gripper_port="/dev/ttyUSB0", camera_param=(320, 240, 10), fisheye_camera_index=0, realsense_serial_number='230322271819'):
        self.gripper_port = gripper_port
        self.camera_param = camera_param
        self.fisheye_camera_index = fisheye_camera_index
        self.realsense_serial_number = realsense_serial_number
        
        self.sense = sense(self.gripper_port)
        if not self.sense.connect():
            raise Exception("连接 Pika Sense 设备失败，请检查设备连接和串口路径")
        
        self.sense.set_vive_tracker_config(config_path="~/.config/libsurvive/config.json")
        print("成功连接到 Pika Sense 设备")
        
        tracker = self.sense.get_vive_tracker()
        if not tracker:
            raise Exception("获取Vive Tracker对象失败，请确保已安装pysurvive库")
        
        devices = self.sense.get_tracker_devices()
        self.target_device = "WM0"
        retry_find_vive_count = 0
        max_retries = 10
        while self.target_device not in devices and retry_find_vive_count < max_retries:
            print(f"未检测到{self.target_device}设备，等待并重试 ({retry_find_vive_count+1}/{max_retries})...")
            time.sleep(1.0)
            devices = self.sense.get_tracker_devices()
            print(f"检测到的设备: {devices}")
            retry_find_vive_count += 1
        
        if self.target_device not in devices:
            print(f"经过多次尝试，仍未检测到{self.target_device}设备")
            print("请确保设备已连接并被正确识别")
            raise Exception("未检测到Vive Tracker设备")
        
        print(f"成功检测到{self.target_device}设备！")
        
        self.sense.set_camera_param(*self.camera_param)
        self.sense.set_fisheye_camera_index(self.fisheye_camera_index)
        self.sense.set_realsense_serial_number(self.realsense_serial_number)
        self.fisheye_camera = self.sense.get_fisheye_camera()
        self.realsense_camera = self.sense.get_realsense_camera()
        
        self.abs_pose0 = None
        
    def on_record_start(self):
        self.abs_pose0 = copy.deepcopy(self.get_vive_abs_pose())
        self.abs_pose0.rotation = [self.abs_pose0.rotation[3], self.abs_pose0.rotation[0], self.abs_pose0.rotation[1], self.abs_pose0.rotation[2]] # 转换为 [w, x, y, z] 格式
        
    def get_sense_msg(self):
        sense_msg = self.sense.get_encoder_data()
        return sense_msg['angle'], sense_msg['rad']
        
    def get_tele_state(self):
        tele_state = self.sense.get_command_state()
        return tele_state
    
    def get_fisheye_rgb(self):
        success, frame = self.fisheye_camera.get_frame()
        if success == False or frame is None:
            raise Exception("获取鱼眼相机图像失败")
        
        return frame
    
    def get_realsense_rgb(self):
        success, frame = self.realsense_camera.get_color_frame()
        if success == False or frame is None:
            raise Exception("获取Realsense相机图像失败")
        
        return frame
    
    def get_realsense_depth(self):
        success, frame = self.realsense_camera.get_depth_frame()
        if success == False or frame is None:
            raise Exception("获取Realsense深度图像失败")
        
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(frame, alpha=0.03), cv2.COLORMAP_JET)
        return depth_colormap
    
    def get_vive_abs_pose(self):
        pose = self.sense.get_pose(self.target_device)
        return pose
    
    def get_vive_relative_pose(self):
        """
        获取相对位姿,以pos, [w, x, y, z] 格式返回
        """
        pose = self.get_vive_abs_pose()
        rotation = [pose.rotation[3], pose.rotation[0], pose.rotation[1], pose.rotation[2]] # 转换为 [w, x, y, z] 格式
        relative_rotation_quaternion = quaternion_mul(quaternion_inv(self.abs_pose0.rotation), rotation)
        relative_position = np.matmul(make_matrix_from_quaternion(self.abs_pose0.rotation).transpose(), np.array(pose.position)-np.array(self.abs_pose0.position))
        return relative_position, relative_rotation_quaternion
    
    
    def disconnect(self):
        self.sense.disconnect()