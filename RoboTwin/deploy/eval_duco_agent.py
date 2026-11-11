"""
Evaluation Agent for Duco Cobot with High Frequency Force Sensor.
"""
import time
import numpy as np
import threading
import socket
import struct
import collections
from scipy.spatial.transform import Rotation as R

from device.robot.duco import DucoRobot
from utils.transformation import xyz_rot_transform
from device.camera.realsense import RealSenseRGBDCamera
from dataset.constants import INTRINSICS
import utils.constants as constants  # 假设你的IP配置在这里

# Quaternion convention used throughout the system for storage and interchange:
#   "xyzw" = scalar-last  = [x, y, z, qx, qy, qz, qw]
# PyTorch3D uses scalar-first [w, x, y, z] internally; all conversion
# functions in utils.transformation use PyTorch3D as backend, so the raw
# output of xyz_rot_transform(..., to_rep="quaternion") is [x,y,z,w,x,y,z].
# We reorder to scalar-last for storage and Agent API.
QUATERNION_CONVENTION = "xyzw"

# =============================================================================
# UDP Client & Sensor Logic (提取自你的 HighFreqDataRecorder)
# =============================================================================

class UDPClient:
    """
    专门用于机械臂2011高频配方端口，接收并解析UDP力传感器数据的客户端。
    (直接复用你提供的代码逻辑)
    """
    def __init__(self, robot_ip, robot_port=2011, pc_ip="0.0.0.0", pc_port=45678, timeout=2.0):
        self.robot_ip = robot_ip
        self.robot_port = robot_port
        self.pc_addr = (pc_ip, pc_port)
        self.sock = None
        self.timeout = timeout
        self._connect()

    def _connect(self):
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.sock.settimeout(self.timeout)
            # 设置 SO_REUSEADDR 以便快速重启时能绑定端口
            self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.sock.bind(self.pc_addr)
            print(f"[ForceSensor] UDP监听器绑定到 {self.pc_addr}")

            # 发送启动指令
            self.start_stream()
        except OSError as e:
            print(f"[ForceSensor] 端口绑定失败: {e}")
            self.sock = None

    def start_stream(self):
        if self.sock:
            try:
                msg = b'start_force_stream'
                self.sock.sendto(msg, (self.robot_ip, self.robot_port))
                print(f"[ForceSensor] 发送启动指令到 {self.robot_ip}:{self.robot_port}")
            except Exception as e:
                print(f"[ForceSensor] 发送启动指令失败: {e}")

    def get_high_freq_data(self):
        if not self.sock: return None
        try:
            data, addr = self.sock.recvfrom(1024)
            if addr[0] != self.robot_ip: return None

            # 解析逻辑保持你提供的原样
            if len(data) == 24:
                # 仅力数据: 6 floats
                force_tuple = struct.unpack('<6f', data)
                # 返回 [Fx, Fy, Fz, Tx, Ty, Tz]
                return np.array(force_tuple)
            elif len(data) == 48:
                # 力 + TCP: 12 floats
                data_tuple = struct.unpack('<12f', data)
                # 我们主要需要前6个 (力/力矩)
                return np.array(data_tuple[0:6])
            else:
                return None
        except socket.timeout:
            # 超时重发指令
            self.start_stream()
            return None
        except Exception as e:
            print(f"[ForceSensor] 解析异常: {e}")
            return None

    def close(self):
        if self.sock:
            self.sock.close()

class AsyncForceSensor:
    """
    异步力传感器包装类。
    在后台线程中持续读取 UDP 数据，并维护一个固定长度的历史队列。
    """
    def __init__(self, robot_ip, pc_port=45678, history_len=200):
        self.client = UDPClient(robot_ip=robot_ip, pc_port=pc_port)

        # 线程安全锁
        self.lock = threading.Lock()
        # 历史数据队列 (maxlen 自动处理溢出)
        self.history = collections.deque(maxlen=history_len)
        self.history_with_timestamps = collections.deque()
        # 最新一帧数据
        self.latest_ft = np.zeros(6)

        self.running = True
        self.thread = threading.Thread(target=self._update_loop, daemon=True)
        self.thread.start()

        # 等待收到第一帧数据，避免初始全0
        print("[ForceSensor] Waiting for first packet...")
        for _ in range(20):
            time.sleep(0.1)
            if np.any(self.latest_ft != 0):
                print("[ForceSensor] Connection Established.")
                break

    def _update_loop(self):
        while self.running:
            ft_data = self.client.get_high_freq_data()
            if ft_data is not None:
                timestamp_ms = int(time.time() * 1000)
                with self.lock:
                    self.latest_ft = ft_data
                    self.history.append(ft_data)
                    self.history_with_timestamps.append((timestamp_ms, ft_data.copy()))
            else:
                # 避免空转占用过多CPU
                time.sleep(0.001)

    def get_latest(self):
        with self.lock:
            return self.latest_ft.copy()

    def get_history(self, n=None):
        """获取最近 n 帧历史数据"""
        with self.lock:
            data = np.array(self.history)
        if len(data) == 0:
            return np.zeros((1, 6))
        if n is None:
            return data
        # 如果请求长度超过历史记录，进行填充或返回全部
        if n > len(data):
            # padding if needed, or just return what we have
            return data
        return data[-n:]

    def get_history_samples(self):
        with self.lock:
            return [(ts, ft.copy()) for ts, ft in self.history_with_timestamps]

    def reset_history(self):
        with self.lock:
            self.history.clear()
            self.history_with_timestamps.clear()

    def stop(self):
        self.running = False
        self.thread.join(timeout=1.0)
        self.client.close()


# =============================================================================
# Main Agent Class
# =============================================================================

class Agent:
    """
    Evaluation agent with Duco Cobot, UDP Force Sensor, and RealSense.
    """
    def __init__(
        self,
        camera_serial='243222076209',
        num_obs_force=100, # 观测需要的历史长度
        robot_ip=constants.ROBOT_IP,
        pc_force_port=45678, # 接收UDP数据的本地端口
        initial_gripper_closed=False,
        move_home_on_init=True,
        init_camera=True,
        **kwargs
    ):
        self.camera_serial = camera_serial
        self.num_obs_force = num_obs_force
        self.camera = None
        self._camera_lock = threading.Lock()
        self._robot_lock = threading.RLock()

        print("Init robot, gripper, sensor, and camera.")

        # 1. 初始化机器人本体 (TCP/Modbus)
        self.robot = DucoRobot(robot_ip=robot_ip)
        if initial_gripper_closed:
            print("Closing gripper before moving to home pose.")
            with self._robot_lock:
                self.robot.close_gripper()
            time.sleep(0.5)
        if move_home_on_init:
            with self._robot_lock:
                self.robot.return_home_pose()
            time.sleep(1.0)
        else:
            print("Skipping return_home_pose on init; using current robot TCP as rollout start.")

        # 2. 初始化高频力传感器 (UDP)
        # 注意：确保防火墙允许该端口 UDP 通信
        self.sensor = AsyncForceSensor(
            robot_ip=robot_ip,
            pc_port=pc_force_port,
            history_len=max(int(num_obs_force), 200)
        )

        # 3. 初始化相机
        if init_camera:
            self.camera = RealSenseRGBDCamera(serial=camera_serial)
            with self._camera_lock:
                for _ in range(30):
                    self.camera.get_rgbd_image()
        else:
            print("Skipping RealSense camera init; external global camera recorder owns the device.")

        print("Initialization Finished.")

    @property
    def intrinsics(self):
        return INTRINSICS[self.camera_serial]

    @property
    def ready_pose(self):
        # [x, y, z, rx, ry, rz]
        return constants.HOME_POSE

    @property
    def ready_rot_6d(self):
        return constants.HOME_POSE

    @property
    def ready_rot_7d(self):
        # HOME_POSE is [x, y, z, rx, ry, rz] (Axis-Angle)
        # Convert to quaternion scalar-last [x, y, z, qx, qy, qz, qw]
        pose_6d = np.array(constants.HOME_POSE)
        pose_7d = xyz_rot_transform(
            pose_6d,
            from_rep="axis_angle",
            to_rep="quaternion"
        )
        # PyTorch3D → scalar-last: indices [0,1,2,4,5,6,3]
        pose_7d = pose_7d[[0, 1, 2, 4, 5, 6, 3]]
        return pose_7d

    def get_observation(self):
        with self._camera_lock:
            colors, depths = self.camera.get_rgbd_image()
        return colors, depths

    # --- 适配后的力控接口 ---

    def get_force_torque_history(self, freq=100):
        """
        获取历史力/力矩数据。
        参数 freq 在原始代码中似乎表示获取的历史长度(steps)，而不仅是频率。
        这里将其解释为“获取最近 freq 帧的数据”。
        """
        history = self.sensor.get_history(n=freq)

        # 确保返回的数据形状是 (freq, 6)，如果历史数据不足，进行零填充
        if len(history) < freq:
            padding = np.zeros((freq - len(history), 6))
            history = np.vstack([padding, history])

        return history

    def reset_runtime_buffers(self):
        """Clear runtime histories that must not leak across pre-rollout robot actions."""
        if hasattr(self.sensor, "reset_history"):
            self.sensor.reset_history()
            print("[Agent] Cleared force history buffer.")

    def get_force_torque(self):
        """获取当前最新的力/力矩 [Fx, Fy, Fz, Tx, Ty, Tz]"""
        return self.sensor.get_latest()

    def get_force(self):
        return self.get_force_torque()[:3]

    def get_torque(self):
        return self.get_force_torque()[3:]

    def get_force_torque_value(self):
        ft = self.get_force_torque()
        # 计算力的大小和力矩的大小 (L2 Norm)
        f_norm = np.linalg.norm(ft[:3])
        t_norm = np.linalg.norm(ft[3:])
        return f_norm, t_norm

    def get_force_value(self):
        return np.linalg.norm(self.get_force())

    def get_torque_value(self):
        return np.linalg.norm(self.get_torque())

    # --- 机器人控制接口 ---

    def get_tcp_pose(self):
        """
        Get current TCP pose.

        Duco SDK returns: [x, y, z, rx, ry, rz] (6-dim, Axis-Angle).
        Converts to quaternion scalar-last [x, y, z, qx, qy, qz, qw].
        """
        if self.robot.duco_robot:
            # 1. Get raw 6D pose from Duco
            with self._robot_lock, self.robot._sdk_lock:
                pose_6d = np.array(self.robot.duco_robot.get_tcp_pose())

            # 2. Convert to quaternion via PyTorch3D
            #    xyz_rot_transform returns PyTorch3D scalar-first: [x, y, z, w, x, y, z]
            pose_7d = xyz_rot_transform(
                pose_6d,
                from_rep="axis_angle",
                to_rep="quaternion"
            )
            # 3. Reorder to scalar-last: [x, y, z, qx, qy, qz, qw]
            #    Indices: 0=x, 1=y, 2=z, 4=qx, 5=qy, 6=qz, 3=qw
            pose_7d = pose_7d[[0, 1, 2, 4, 5, 6, 3]]
            return pose_7d

        # Return 7-dim zeros on failure to avoid dimension errors
        return np.zeros(7)

    def get_robot_joint_position(self):
        with self._robot_lock:
            return self.robot.get_robot_joint_position()

    def get_gripper_position_raw(self):
        with self._robot_lock:
            return self.robot.get_gripper_position()

    def get_gripper_width(self):
        """Return the current gripper opening in metres."""
        return float(self.get_gripper_position_raw()) / 1000.0 * 0.095

    def set_tcp_pose(self, pose, rotation_rep, rotation_rep_convention=None, blocking=False):
        # Duco 需要 axis_angle (旋转向量)
        tcp_pose = xyz_rot_transform(
            pose,
            from_rep=rotation_rep,
            to_rep="axis_angle",
            from_convention=rotation_rep_convention
        )
        with self._robot_lock:
            self.robot.move_robot_cartesian(tcp_pose)
        if blocking:
            time.sleep(0.1)

    def move_to_pose_safe(self, pose, rotation_rep="axis_angle", rotation_rep_convention=None):
        """
        使用 movej_pose2 安全地移动到指定位置（阻塞式）。
        
        Args:
            pose: TCP pose in [x, y, z] + rotation_rep.
                  注意：请确保传入的是正确的完整位姿。
        """
        # Duco movej_pose2 only accepts [x, y, z, rx, ry, rz] axis-angle poses.
        target_pose_axis = xyz_rot_transform(
            pose,
            from_rep=rotation_rep,
            to_rep="axis_angle",
            from_convention=rotation_rep_convention
        )

        # 2. 调用底层的 movej_pose2
        if self.robot.duco_robot:
            # movej_pose2(pose, v, a, r, q_near, tool, wobj, block)
            # 注意：Duco SDK 的 movej_pose2 接受的是 [x,y,z,rx,ry,rz]
            q_near = []
            try:
                current_joints = self.robot.get_robot_joint_position()
                q_near = np.asarray(current_joints, dtype=np.float64).reshape(-1).tolist()
                if len(q_near) < 6 or not np.all(np.isfinite(q_near[:6])):
                    print(f"[SafeMove] q_near unavailable: invalid current joints {current_joints}")
                    q_near = []
                else:
                    q_near = q_near[:6]
                    print(f"[SafeMove] using current joints as q_near={np.array2string(np.asarray(q_near), precision=6)}")
            except Exception as e:
                print(f"[SafeMove] q_near unavailable: {e}")
            try:
                # 尝试使用关节空间规划到达笛卡尔目标（更不容易报奇异）
                with self._robot_lock:
                    self.robot.duco_robot.movej_pose2(
                        target_pose_axis,
                        0.5,  # velocity
                        0.5,  # acceleration
                        0,  # blend radius
                        q_near, "", "", True  # blocking=True
                    )
                    print(f"[SafeMove] movej_pose2 q_near_used={bool(q_near)}")
            except Exception as e:
                if q_near:
                    print(f"[SafeMove] movej_pose2 with q_near failed: {e}; retrying with empty q_near")
                    try:
                        with self._robot_lock:
                            self.robot.duco_robot.movej_pose2(
                                target_pose_axis,
                                0.5,
                                0.5,
                                0,
                                [], "", "", True
                            )
                            print("[SafeMove] movej_pose2 q_near_used=False fallback=True")
                    except Exception as fallback_e:
                        print(f"[SafeMove] Error: {fallback_e}")
                else:
                    print(f"[SafeMove] Error: {e}")

        time.sleep(0.5)  # 稍微停顿，让伺服稳定

    def set_gripper_width(self, width, blocking=False):
        # 映射: width(m) -> 0~1000
        target_val = int(np.clip(width / 0.095 * 1000., 0, 1000))

        with self._robot_lock:
            if target_val < 10:
                self.robot.close_gripper()
            else:
                self.robot.open_gripper(width=target_val)

        if blocking:
            time.sleep(0.5)

    def stop(self):
        self.sensor.stop()
        try:
            if getattr(self, "camera", None) is not None and getattr(self.camera, "pipeline", None) is not None:
                self.camera.pipeline.stop()
        except Exception as e:
            print(f"Camera stop failed: {e}")
        try:
            if getattr(self, "robot", None) is not None and hasattr(self.robot, "stop_servo"):
                self.robot.stop_servo()
        except Exception as e:
            print(f"Robot servo stop failed: {e}")
        # self.robot.stop() # 如果 DucoRobot 有 stop 方法
        print("Agent stopped.")
