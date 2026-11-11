# 文件名: duco.py
import numpy as np
import time
from copy import deepcopy as copy
# TODO：重新引入
from device.robot.DucoCobot import DucoCobot

# TODO：确定常量与配置
# 为机器人定义一个安全的初始/家庭位置 (单位: 弧度)
# 注意：这个位置需要根据您的实际应用场景进行调整
import utils.constants as constants
DUCO_HOME_JOINT_POS = constants.HOME_POSE

ROBOT_IP = constants.ROBOT_IP
ROBOT_PORT = constants.ROBOT_PORT


# =============================================================================
# Gripper Control Class (夹爪控制类)
# =============================================================================
class Gripper:
    """
    该类封装了通过 RS-485 (Modbus RTU) 协议与夹爪通信的底层逻辑。
    """

    def __init__(self, arm_sdk):
        """
        初始化夹爪，并执行两阶段的初始化指令序列。
        参数:
            arm_sdk: 已连接的 DucoCobot SDK 实例。
        """
        self.arm = arm_sdk
        print("正在初始化 Gripper...")

        # 发送第一阶段初始化指令
        data = [1, 0x06, 0x01, 0, 0, 1]
        data = self.append_crc_to_data(data)
        data = self.normalize_to_range(data)
        self.arm.tool_write_raw_data_485(data)
        time.sleep(0.1)
        response = self.arm.tool_read_raw_data_485(255)
        if not response:
            raise RuntimeError("夹爪第一阶段初始化失败：未收到响应")

        # 发送第二阶段初始化指令
        data = [1, 0x06, 0x01, 1, 0, 0x14]
        data = self.append_crc_to_data(data)
        data = self.normalize_to_range(data)
        self.arm.tool_write_raw_data_485(data)
        time.sleep(0.1)
        response = self.arm.tool_read_raw_data_485(255)
        if not response:
            raise RuntimeError("夹爪第二阶段初始化失败：未收到响应")

        print("Gripper 初始化完成！")

    def open(self, length=1000):
        print(f"夹爪打开到位置: {length}")
        if self.arm:
            data = [1, 0x06, 0x01, 3, int(length / 256), length & 0x00ff]
            data = self.append_crc_to_data(data)
            data = self.normalize_to_range(data)
            self.arm.tool_write_raw_data_485(data)
            time.sleep(0.1)

    def close(self):
        print("夹爪关闭")
        if self.arm:
            data = [1, 0x06, 0x01, 3, 0, 0]
            data = self.append_crc_to_data(data)
            data = self.normalize_to_range(data)
            self.arm.tool_write_raw_data_485(data)
            time.sleep(0.1)

    def get_pose(self, time_interval=0.2):
        if self.arm:
            data = [1, 0x03, 0x02, 2, 0, 1]
            data = self.append_crc_to_data(data)
            data = self.normalize_to_range(data)
            # 清空接收缓冲区
            while len(self.arm.tool_read_raw_data_485(255)) > 0:
                continue
            # 发送读取指令
            self.arm.tool_write_raw_data_485(data)
            response_data = self.arm.tool_read_raw_data_485(7)
            if len(response_data) == 7:
                response_data = self.range_to_normalize(response_data)
                pose = response_data[3] * 256 + response_data[4]
                return True, pose
            else:
                print("获取夹爪位置失败，响应数据长度不足。")
                return False, 0

    def append_crc_to_data(self, data):
        crc = 0xFFFF
        for pos in data:
            crc ^= pos
            for _ in range(8):
                if (crc & 0x0001) != 0:
                    crc >>= 1
                    crc ^= 0xA001
                else:
                    crc >>= 1
        crc_val = crc & 0xFFFF
        data.append(crc_val & 0xFF)
        data.append((crc_val >> 8) & 0xFF)
        return data

    def normalize_to_range(self, data):
        return [num - 256 if num > 127 else num for num in data]

    def range_to_normalize(self, data):
        return [num + 256 if num < 0 else num for num in data]


# =============================================================================
# Main Robot Control Wrapper (主机器人控制包装器)
# =============================================================================
class DucoRobot():
    """
    Duco 协作机器人的机器人包装器 (Robot Wrapper)，专为集成到 Open-Teach 框架而设计。
    这个类使用 Duco 机器人的原生 Python SDK 直接处理与机器人和夹爪的通信，
    从而将底层的控制细节与 Open-Teach 的主逻辑分离开来。
    """

    def __init__(self, robot_ip=ROBOT_IP, robot_port=ROBOT_PORT, need_enable=True):
        """
        初始化与 Duco 协作机器人的连接。

        参数:
            robot_ip (str): Duco 机器人控制器的 IP 地址。
            robot_port (int): 控制器上 RPC 服务的端口号。
        """
        # try:
        #     rospy.init_node("dex_arm", disable_signals=True, anonymous=True)
        #     rospy.loginfo("ROS 节点 'dex_arm' 已为 Duco Control 初始化。")
        # except rospy.exceptions.ROSException:
        #     pass

        self.robot_ip = robot_ip
        self.robot_port = robot_port
        self.duco_robot = None
        self.gripper = None  # 初始化夹爪对象为空

        self._init_robot_control(need_enable)
        self._init_servo_members()
        try:
            if self.duco_robot:
                cur_tcp = self.duco_robot.get_tcp_pose()
                if cur_tcp and len(cur_tcp) == 6:
                    self._servo_target = list(cur_tcp)  # 起步就以当前姿态为目标
        except Exception:
            pass

    def _init_robot_control(self, need_enable=True):
        """
        建立并初始化与 Duco 机器人和夹爪的连接。
        """
        print(f"正在连接 Duco 协作机器人，地址: {self.robot_ip}:{self.robot_port}...")
        self.duco_robot = DucoCobot(self.robot_ip, self.robot_port)

        try:

            # print("Duco 机器人断电重启中，请稍候...")
            # self.duco_robot.power_off(True)
            # print("Duco 机器人上电中，请稍候...")
            # self.duco_robot.power_on(True)
            # print("Duco 机器人使能中，请稍候...")
            # self.duco_robot.enable(True)

            if self.duco_robot.open() != 0:
                raise ConnectionError("打开与 Duco 机器人的连接失败。")

            print("连接成功。正在使能机器人...")
            if need_enable:
                self.duco_robot.enable(True)
            time.sleep(1)
            print("Duco 协作机器人已连接并使能。")

            # 初始化夹爪
            try:
                self.gripper = Gripper(self.duco_robot)
            except Exception as e:
                print(f"夹爪初始化失败: {e}")
                # 即使夹爪失败，也允许机械臂继续工作
                self.gripper = None

        except Exception as e:
            print(f"机器人初始化过程中发生错误: {e}")
            self.duco_robot = None
            raise

    # ------------------------------------------1----------------------------------

    def _stream_servoj(self, target_pose, rate_hz=50, duration_s=5, v=1.0, a=0.5, kp=200, kd=25):
        """
        以给定频率将同一目标位姿通过 servoj_pose 连续下发 duration_s 秒。
        这是高频伺服控制的基本使用方式：循环里不停发同一个目标。
        """
        if self.duco_robot is None:
            print("无法执行伺服：机器人未连接。")
            return
        import time
        dt = 1.0 / max(1, int(rate_hz))
        t_end = time.time() + float(duration_s)
        pose = list(np.array(target_pose, dtype=float).flatten())
        # non-blocking，高频循环
        while time.time() < t_end:
            try:
                self.duco_robot.servoj_pose(
                    pose_list=pose, v=v, a=a,
                    q_near=[], tool="", wobj="",
                    block=False, kp=kp, kd=kd
                )
            except Exception as e:
                print(f"servoj_pose 发送失败: {e}")
                break
            # 简单的定频睡眠
            next_t = time.time() + dt
            # 避免忙等
            while True:
                now = time.time()
                if now >= next_t: break
                time.sleep(min(0.0005, max(0.0, next_t - now)))

    # ==== Servoj 控制核心（新增） ====
    def _init_servo_members(self):
        import threading
        self._servo_running = False
        self._servo_thread = None
        self._servo_rate_hz = 60  # 默认 60Hz（更稳）
        self._servo_v = 1.0
        self._servo_a = 0.5
        self._servo_kp = 200
        self._servo_kd = 25
        self._servo_target = None  # [x,y,z,rx,ry,rz]
        self._servo_lock = threading.Lock()
        self._sdk_lock = threading.RLock()
        self._servo_qseed = None  # 关节种子

    def start_servo(self, rate_hz: int = 60, v: float = 1.0, a: float = 0.5, kp: float = 200, kd: float = 25):
        """启动高频 servoj 线程（不阻塞）。"""
        import threading
        if self.duco_robot is None:
            print("无法启动伺服：机器人未连接");
            return False
        if self._servo_running:
            print("伺服已运行");
            return True
        if not hasattr(self, "_servo_lock"):
            self._init_servo_members()
        self._servo_rate_hz = int(rate_hz)
        self._servo_v, self._servo_a = float(v), float(a)
        self._servo_kp, self._servo_kd = float(kp), float(kd)
        self._servo_running = True

        def _loop():
            import time, numpy as np
            dt = 1.0 / max(1, self._servo_rate_hz)
            t0 = time.perf_counter()
            i = 0
            last_q_refresh = 0
            successful_sends = 0
            last_rate_report = t0
            while self._servo_running:
                i += 1
                # 读取目标
                with self._servo_lock:
                    tgt = None if self._servo_target is None else np.array(self._servo_target, dtype=float).flatten()
                if tgt is not None and tgt.size == 6:
                    # qseed 只需低频刷新，避免状态 RPC 挤占 60 Hz 指令周期。
                    if i - last_q_refresh >= 30:
                        try:
                            with self._sdk_lock:
                                jq = np.asarray(
                                    self.duco_robot.get_actual_joints_position(),
                                    dtype=float,
                                ).reshape(-1)
                            if jq.size >= 6 and np.all(np.isfinite(jq[:6])):
                                self._servo_qseed = jq[:6].tolist()
                        except Exception:
                            pass
                        last_q_refresh = i
                    try:
                        with self._sdk_lock:
                            self.duco_robot.servoj_pose(
                                pose_list=tgt.tolist(),
                                v=self._servo_v, a=self._servo_a,
                                q_near=(self._servo_qseed if self._servo_qseed else []),
                                tool="", wobj="",
                                block=False, kp=self._servo_kp, kd=self._servo_kd
                            )
                        successful_sends += 1
                    except Exception as e:
                        print(f"servoj_pose 下发异常: {e}")
                now = time.perf_counter()
                if now - last_rate_report >= 2.0:
                    actual_rate = successful_sends / (now - last_rate_report)
                    print(f"[servoj] actual send rate={actual_rate:.1f} Hz")
                    successful_sends = 0
                    last_rate_report = now
                # 定频
                next_t = t0 + i * dt
                while True:
                    now = time.perf_counter()
                    if now >= next_t: break
                    time.sleep(min(0.0005, max(0.0, next_t - now)))
            print("伺服线程退出")

        self._servo_thread = threading.Thread(target=_loop, daemon=True)
        self._servo_thread.start()
        print(f"已启动 servoj 线程，{self._servo_rate_hz} Hz")
        return True

    def set_servo_target(self, pose_6):
        """更新伺服目标（绝对 TCP 位姿 [x,y,z,rx,ry,rz]）。"""
        import numpy as np

        # —— 懒初始化保护（避免还没建锁就被调用）——
        if not hasattr(self, "_servo_lock"):
            self._init_servo_members()

        p = np.array(pose_6, dtype=float).flatten()
        if p.size != 6:
            print(f"set_servo_target 需要 6 元素，得到 {p.size}")
            return
        # 可选的工作空间夹紧（避免越界/奇异）
        p[0] = np.clip(p[0], -0.40, +0.60)
        p[1] = np.clip(p[1], -0.50, +0.50)
        p[2] = max(p[2], 0.10)

        with self._servo_lock:
            self._servo_target = p.tolist()

        # —— 若伺服线程还没跑，自动以 60Hz 启动一次（兜底，防时序问题）——
        if not getattr(self, "_servo_running", False):
            print("servoj 线程未运行，自动以 60Hz 启动")
            try:
                self.start_servo(rate_hz=60, v=0.3, a=0.2, kp=200, kd=25)
            except Exception as e:
                print(f"自动启动 servoj 失败: {e}")

    def stop_servo(self):
        """停止 servoj 线程。"""
        if not hasattr(self, "_servo_running") or not self._servo_running:
            return
        self._servo_running = False
        try:
            if self._servo_thread: self._servo_thread.join(timeout=2.0)
        except Exception:
            pass
        self._servo_thread = None

    # def set_servo_target(self, pose_6): 第一版
    #     """更新伺服目标（绝对 TCP 位姿 [x,y,z,rx,ry,rz]）。"""
    #     import numpy as np
    #     p = np.array(pose_6, dtype=float).flatten()
    #     if p.size != 6:
    #         rospy.logerr(f"set_servo_target 需要 6 元素，得到 {p.size}")
    #         return
    #     # 可选的工作空间夹紧（避免越界/奇异）
    #     p[0] = np.clip(p[0], -0.40, +0.60)
    #     p[1] = np.clip(p[1], -0.50, +0.50)
    #     p[2] = max(p[2], 0.10)
    #     with self._servo_lock:
    #         self._servo_target = p.tolist()
    # ==== Servoj 控制核心（新增结束） ====

    # def _stream_servoj(self, target_pose, rate_hz=50, duration_s=5.0, v=1.0, a=0.5, kp=200, kd=25):
    #     if self.duco_robot is None:
    #         rospy.logerr("无法执行伺服：机器人未连接。"); return
    #     import time, numpy as np
    #     dt = 1.0 / max(1, int(rate_hz))
    #     t0 = time.perf_counter()
    #     t_end = t0 + float(duration_s)
    #     pose = list(np.array(target_pose, dtype=float).flatten())

    #     # 频率统计
    #     count = 0
    #     last_report = t0

    #     i = 0
    #     while True:
    #         now = time.perf_counter()
    #         if now >= t_end: break
    #         try:
    #             self.duco_robot.servoj_pose(
    #                 pose_list=pose, v=v, a=a,
    #                 q_near=[], tool="", wobj="",
    #                 block=False, kp=kp, kd=kd
    #             )
    #         except Exception as e:
    #             rospy.logerr(f"servoj_pose 发送失败: {e}")
    #             break
    #         count += 1; i += 1
    #         # 每秒输出一次实际Hz
    #         if now - last_report >= 1.0:
    #             hz = count / (now - last_report)
    #             rospy.loginfo(f"[servoj stream] target={rate_hz}Hz, actual≈{hz:.1f}Hz")
    #             count = 0
    #             last_report = now
    #         # 严格节拍：相对于 t0 的第 i 个 tick
    #         next_t = t0 + i * dt
    #         while True:
    #             now2 = time.perf_counter()
    #             if now2 >= next_t: break
    #             time.sleep(min(0.0005, max(0.0, next_t - now2)))

    def _offset_pose(self, base_pose, dx=0, dy=0, dz=0, drx=0, dry=0, drz=0):
        """
        在笛卡尔位姿 [x,y,z,rx,ry,rz] 上叠加一个小偏移，返回新位姿。
        """
        p = np.array(base_pose, dtype=float).flatten()
        p[:3] += np.array([dx, dy, dz], dtype=float)
        p[3:] += np.array([drx, dry, drz], dtype=float)
        return p.tolist()

    def _stream_servoj_path(self, start_pose, end_pose, rate_hz=50, duration_s=5.0, v=1.0, a=0.5, kp=200, kd=25):
        """
        在 duration_s 内，把目标从 start_pose 线性插值到 end_pose，
        每个 tick 下发一次 servoj_pose。频率不同 => 细腻程度不同（可视化更明显）。
        """
        if self.duco_robot is None:
            print("无法执行伺服：机器人未连接。");
            return
        import time
        import numpy as np
        start = np.array(start_pose, dtype=float).flatten()
        end = np.array(end_pose, dtype=float).flatten()
        assert start.size == 6 and end.size == 6

        dt = 1.0 / max(1, int(rate_hz))
        steps = max(1, int(duration_s / dt))
        t0 = time.perf_counter()
        for i in range(steps + 1):
            alpha = i / float(steps)
            pose = (1.0 - alpha) * start + alpha * end
            try:
                self.duco_robot.servoj_pose(
                    pose_list=pose.tolist(), v=v, a=a,
                    q_near=[], tool="", wobj="",
                    block=False, kp=kp, kd=kd
                )
            except Exception as e:
                print(f"servoj_pose 发送失败: {e}")
                break
            # 稳定节拍器（基于绝对时间，抑制漂移）
            next_t = t0 + (i + 1) * dt
            while True:
                now = time.perf_counter()
                if now >= next_t: break
                time.sleep(min(0.0005, max(0.0, next_t - now)))

    # ==================== Gripper Control Methods (夹爪控制方法) ====================

    def open_gripper(self, width=1000):
        """打开夹爪到指定宽度。"""
        if self.gripper:
            with self._sdk_lock:
                self.gripper.open(width)
        else:
            print("夹爪未初始化，无法执行打开操作。")

    def close_gripper(self):
        """完全关闭夹爪。"""
        if self.gripper:
            with self._sdk_lock:
                self.gripper.close()
        else:
            print("夹爪未初始化，无法执行关闭操作。")

    def get_gripper_position(self, time_interval=0.2):
        """
        获取夹爪当前的位置读数。
        返回:
            int: 夹爪的当前位置值，如果获取失败则返回 0。
        """
        if self.gripper:
            with self._sdk_lock:
                success, pose = self.gripper.get_pose(time_interval=time_interval)
            return pose if success else 0
        else:
            print("夹爪未初始化，无法获取位置。")
            return 0

    # ==================== Robot State Methods (机器人状态方法) ====================
    def set_gripper(self, status):
        """
        控制夹爪的开闭状态。
        参数:
            status (bool): True 打开夹爪，False 关闭夹爪。
        """
        if status:
            self.open_gripper(width=1000)
        else:
            self.close_gripper()

    def get_robot_state(self):
        """
        获取机器人关节的当前实际状态。
        """
        if self.duco_robot is None: return None
        return dict(
            position=np.array(self.duco_robot.get_actual_joints_position(), dtype=np.float32),
            velocity=np.array(self.duco_robot.get_actual_joints_speed(), dtype=np.float32),
            effort=np.array(self.duco_robot.get_actual_joints_torque(), dtype=np.float32),
            timestamp=time.time()
        )

    def get_commanded_robot_state(self):
        """
        获取机器人关节的最近一次指令（目标）状态。
        """
        if self.duco_robot is None: return None
        return dict(
            position=np.array(self.duco_robot.get_target_joints_position(), dtype=np.float32),
            velocity=np.array(self.duco_robot.get_target_joints_speed(), dtype=np.float32),
            effort=np.array(self.duco_robot.get_target_joints_torque(), dtype=np.float32),
            timestamp=time.time()
        )

    def get_robot_joint_position(self):
        """获取机器人当前的关节位置（单位：弧度）。"""
        return np.array(self.duco_robot.get_actual_joints_position()) if self.duco_robot else None

    def get_robot_cartesian_position(self):
        """获取机器人当前末端执行器 (TCP) 的位姿。"""
        # current_pos, current_quat = copy(self.duco_robot.get_tcp_pose())
        # print("机械臂TCP位姿形状:", self.duco_robot.get_tcp_pose())
        pose = self.duco_robot.get_tcp_pose()
        current_pos = pose[:3]
        current_quat = pose[3:]
        cartesian_state = dict(
            position=np.array(current_pos, dtype=np.float32).flatten(),
            orientation=np.array(current_quat, dtype=np.float32).flatten(),
            timestamp=time.time()
        )
        return cartesian_state if self.duco_robot else None

    def get_robot_velocity(self):
        """获取机器人当前的关节速度（单位：rad/s）。"""
        return np.array(self.duco_robot.get_actual_joints_speed()) if self.duco_robot else None

    def get_robot_torque(self):
        """获取机器人关节施加的力矩（单位：N.m）。"""
        return np.array(self.duco_robot.get_actual_joints_torque()) if self.duco_robot else None

    def get_commanded_robot_joint_position(self):
        """获取机器人被指令的目标关节位置。"""
        return np.array(self.duco_robot.get_target_joints_position()) if self.duco_robot else None

    # ==================== Robot Movement Methods (机器人运动方法) ====================

    # def move_robot_joint(self, joint_angles, velocity=1.0, acceleration=1.0, block=True):
    #     """
    #     将机器人移动到一组指定的关节角度。
    #     """
    #     if self.duco_robot is None:
    #         rospy.logerr("无法移动机器人，连接已断开。")
    #         return
    #     rospy.loginfo(f"正在移动到关节角度: {np.rad2deg(joint_angles).round(2)}")
    #     self.duco_robot.movej2(list(joint_angles), velocity, acceleration, 0, block)

    # move_robot = move_robot_joint

    # def move_robot_cartesian(self, cartesian_pose):
    #     """
    #     控制机器人进行笛卡尔空间运动到指定位姿。
    #     参数:
    #         cartesian_pose (list或np.ndarray): [x, y, z, rx, ry, rz]，单位米和弧度。
    #     """
    #     if self.duco_robot is None:
    #         rospy.logerr("无法执行笛卡尔运动，连接已断开。")
    #         return
    #     pose = np.array(cartesian_pose, dtype=float).flatten()
    #     if pose.size != 6:
    #         rospy.logerr(f"cartesian_pose 长度应为 6，当前为 {pose.size}")
    #         return
    #     velocity = 0.25  # m/s
    #     acceleration = 0.5  # m/s^2
    #     radius = 0.0
    #     q_near = None
    #     tool = None
    #     wobj = None
    #     block = True
    #     try:
    #         rospy.loginfo(f"笛卡尔移动到: {pose.round(4)}  vel={velocity} acc={acceleration}")
    #         # self.duco_robot.movel(list(pose), velocity, acceleration, radius, q_near, tool, wobj, block)
    #         self.duco_robot.movej_pose2(list(pose), 3, 1.0, 0.05, [], "", "", True)
    #         print("已经移动到" + pose)
    #         rospy.loginfo("笛卡尔运动完成")
    #     except Exception as e:
    #         rospy.logerr(f"执行笛卡尔运动失败: {e}")

    def move_robot_cartesian(self, cartesian_pose):
        """更新遥操作的笛卡尔伺服目标。

        Duco ``servoj_pose`` 需要持续的高频目标流。上层策略只以数 Hz
        更新目标，因此由 ``set_servo_target`` 启动的 60 Hz 线程重复下发。
        """
        if self.duco_robot is None:
            print("无法执行伺服：未连接")
            return

        pose = np.array(cartesian_pose, dtype=float).flatten()
        if pose.size != 6:
            print(f"cartesian_pose 长度应为 6，当前为 {pose.size}")
            return

        self.set_servo_target(pose)

    def return_home_pose(self):
        """将机器人移动到一个预定义的初始（Home）位置。"""
        print("机器人正在归位...")

        self.duco_robot.movej_pose2(
            constants.HOME_POSE, 3, 1.0, 0.05, [], "", "", True)


        print("机器人已回到初始位置。")

    def reset_robot(self):
        """通过移动到初始位置来重置机器人。"""
        print("正在重置机器人...")
        self.return_home_pose()
        print("机器人重置完成。")

    def __del__(self):
        """析构函数，确保在对象销毁时能干净地关闭与机器人的连接。"""
        if self.duco_robot:
            print("正在禁用并关闭与 Duco 协作机器人的连接。(暂时屏蔽)")
            # self.duco_robot.disable(True)
            # self.duco_robot.close()
