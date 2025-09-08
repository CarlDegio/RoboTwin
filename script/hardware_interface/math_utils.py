import numpy as np


def make_matrix_from_quaternion(q):
        """将四元数转换为旋转矩阵"""
        w, x, y, z = q
        
        xx = x * x
        xy = x * y
        xz = x * z
        xw = x * w
        
        yy = y * y
        yz = y * z
        yw = y * w
        
        zz = z * z
        zw = z * w
        
        rotation_matrix = np.array([
            [1 - 2 * (yy + zz), 2 * (xy - zw), 2 * (xz + yw)],
            [2 * (xy + zw), 1 - 2 * (xx + zz), 2 * (yz - xw)],
            [2 * (xz - yw), 2 * (yz + xw), 1 - 2 * (xx + yy)]
        ])
        
        return rotation_matrix

def quaternion_inv(q):
    """将四元数转换为逆四元数, w,x,y,z"""
    return [q[0], -q[1], -q[2], -q[3]]

def quaternion_mul(q1, q2):
    """将两个四元数相乘, w,x,y,z"""
    return np.array([q1[0] * q2[0] - q1[1] * q2[1] - q1[2] * q2[2] - q1[3] * q2[3],
            q1[0] * q2[1] + q1[1] * q2[0] + q1[2] * q2[3] - q1[3] * q2[2],
            q1[0] * q2[2] - q1[1] * q2[3] + q1[2] * q2[0] + q1[3] * q2[1],
            q1[0] * q2[3] + q1[1] * q2[2] - q1[2] * q2[1] + q1[3] * q2[0]])
    
def make_quat_from_rpy(rx, ry, rz):
    """将rpy转换为四元数, rpy是绕原始系旋转的, 顺序wxyz"""
    rx_quat = [np.cos(rx/2), np.sin(rx/2), 0, 0]
    ry_quat = [np.cos(ry/2), 0, np.sin(ry/2), 0]
    rz_quat = [np.cos(rz/2), 0, 0, np.sin(rz/2)]
    quat=quaternion_mul(rz_quat, quaternion_mul(ry_quat, rx_quat))
    return quat

def make_matrix_from_rpy(rx, ry, rz):
    """将rpy转换为旋转矩阵"""
    quat = make_quat_from_rpy(rx, ry, rz)
    return make_matrix_from_quaternion(quat)

def make_rpy_from_matrix(matrix):
    """将旋转矩阵转换为rpy, rpy是绕原始系旋转的"""
    if np.sqrt(matrix[2, 1]**2 + matrix[2, 2]**2) < 1e-3:
        ry = np.pi/2
        rx = 0.0
        rz = np.arctan2(-matrix[0, 1], matrix[1, 1])
    else:
        ry = np.arctan2(-matrix[2, 0], np.sqrt(matrix[2, 1]**2 + matrix[2, 2]**2))
        rx = np.arctan2(matrix[2, 1], matrix[2, 2])
        rz = np.arctan2(matrix[1, 0], matrix[0, 0])
    return rx, ry, rz

if __name__ == "__main__":
    quat = np.array([0.5, 0.0 , 0.86, 0])
    quat = quat/np.linalg.norm(quat)
    quat0 = np.array([0.5, 0.0 , 0.87, 0])
    quat0 = quat0/np.linalg.norm(quat0)
    
    relative_rotation_quaternion = quaternion_mul(quaternion_inv(quat0), quat)
    relative_position = np.matmul(make_matrix_from_quaternion(quat0).transpose(), np.array([0.0, 0.0, 0.0]))
    print(relative_rotation_quaternion)
    print(relative_position)