"""
URClient definition.
"""

import cv2
import zmq
import time
import pickle
import base64
import threading
import numpy as np
import rtde_control, rtde_receive
from scipy.spatial.transform import Rotation as R
import experiments.robot.snumpr.ur_controller.robotiq as robotiq


class ZMQCameraSubscriber:
    def __init__(self, host, port, topic_type):
        self._host, self._port, self._topic_type = host, port, topic_type
        self._init_subscriber()

    def _init_subscriber(self):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)
        self.socket.setsockopt(zmq.CONFLATE, 1)
        print("tcp://{}:{}".format(self._host, self._port))
        self.socket.connect("tcp://{}:{}".format(self._host, self._port))
        self.socket.setsockopt(zmq.SUBSCRIBE, b"rgb_image")

        poller = zmq.Poller()
        poller.register(self.socket, zmq.POLLIN)
        socks = poller.poll(500)

        if socks:
            print("[OK] Received a message from publisher!")
            return True
        else:
            print("[WARN] No message received (publisher not sending or wrong address?)")
            return False

    def recv_rgb_image(self, timeout_ms=1000):
        poller = zmq.Poller()
        poller.register(self.socket, zmq.POLLIN)

        socks = poller.poll(timeout_ms)
        if not socks:  # timeout
            return None, None

        raw_data = self.socket.recv()

        data = raw_data.lstrip(b"rgb_image ")
        data = pickle.loads(data)

        encoded_data = np.frombuffer(base64.b64decode(data["rgb_image"]), dtype=np.uint8)
        img = cv2.imdecode(encoded_data, cv2.IMREAD_COLOR)
        return img, data["timestamp"]

    def stop(self):
        print("Closing the subscriber socket in {}:{}.".format(self._host, self._port))
        self.socket.close()
        self.context.term()


class URClient:
    def __init__(self, host: str = "localhost", ur_ip: str = "localhost", port: int = 5556):
        """
        Args:
            host: camera publisher host ip
            ur_ip: UR controller ip
            port: (kept for interface compatibility, not used directly)
        """
        self.rtde_c = rtde_control.RTDEControlInterface(ur_ip)
        self.rtde_r = rtde_receive.RTDEReceiveInterface(ur_ip)

        self.gripper = robotiq.RobotiqGripper()
        self.gripper.connect(ur_ip, 63352)
        self.gripper.activate()

        self._pose_lock = threading.Lock()
        self.target_pose = None
        self._control_running = True

        # Flag to distinguish idle vs active servo control
        self._control_active = False

        # Servo parameters
        self.servo_accel = 0.0
        self.servo_speed = 0.0
        self.servo_lookahead = 0.1
        self.servo_gain = 100

        # Initialize target to actual pose once before starting control loop
        self._sync_target_to_actual()

        self._control_thread = threading.Thread(target=self._control_loop, daemon=True)
        self._control_thread.start()

        self.image_subscriber = ZMQCameraSubscriber(
            host=host,
            port=10005,
            topic_type="RGB",
        )

        print(f"URClient connected to {host}:{port}")

    def _flush_receive_buffer(self):
        print("Flushing receive buffer...")
        for _ in range(100):
            self.rtde_r.getActualTCPPose()
        print("Receive buffer flushed.")

    def _sync_target_to_actual(self, flush=True):
        """Read current TCP pose and set it as the servo target."""
        if flush:
            self._flush_receive_buffer()
            
        actual_pose = self.rtde_r.getActualTCPPose()
        with self._pose_lock:
            self.target_pose = list(actual_pose)
            
        return actual_pose

    def _control_loop(self, frequency=100.0):
        """
        High-frequency control loop to send servo commands to the robot.
        """
        dt = 1.0 / frequency

        print(f"[Control Loop] Started at {frequency}Hz")

        while self._control_running:
            # If not active, do nothing this cycle
            if not self._control_active:
                time.sleep(dt)
                continue

            start_time = time.time()
            current_pose = self.rtde_r.getActualTCPPose()

            with self._pose_lock:
                # Fallback to current pose if target is not initialized
                if self.target_pose is None:
                    cmd_pose = list(current_pose)
                else:
                    cmd_pose = list(self.target_pose)

            current_np = np.array(current_pose[:3], dtype=np.float32)
            cmd_np = np.array(cmd_pose[:3], dtype=np.float32)
            diff = np.linalg.norm(current_np - cmd_np)

            # Safety: if target drifts too far from actual, resync to actual pose
            if (0.15 < diff):
                print(f"[SafeGuard] Drift detected ({diff:.3f}m). Re-syncing.")
                current_pose = self._sync_target_to_actual(flush=False)
                cmd_pose = list(current_pose)

            try:
                self.rtde_c.servoL(
                    cmd_pose,
                    self.servo_speed,
                    self.servo_accel,
                    dt,
                    self.servo_lookahead,
                    self.servo_gain,
                )
            except Exception as e:
                print("[servo loop] error:", e)

            elapsed = time.time() - start_time
            if elapsed < dt:
                time.sleep(dt - elapsed)

    def init(self, env_params: dict, image_size: int = 224):
        """
        Initialize the environment.
            env_params: dict of env params
            image_size: the size of the cropped image to return
        """
        self.env_params = env_params
        self.image_size = image_size

    def get_observation(self):
        full_image, _ = self.image_subscriber.recv_rgb_image()

        if full_image is None:
            return None

        w = 250
        image = full_image[:, w : w + 720, :]
        image = cv2.resize(image, (self.image_size, self.image_size))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        return {"image": image, "full_image": image}
    
    def stop(self):
        print("Stopping URClient...")
        self._control_active = False
        self.rtde_c.servoStop()
        
        with self._pose_lock:
            self.target_pose = None
            
        time.sleep(0.5)
            
        print("URClient stopped.")

    def reset(self):
        # Put the control loop into idle state
        self._control_active = False
        with self._pose_lock:
            self.target_pose = None

    def _rotvec2rpy(self, rotvec):
        """Convert rotation vector to roll-pitch-yaw."""
        return R.from_rotvec(rotvec).as_euler("xyz")

    def _rpy2rotvec(self, rpy):
        """Convert roll-pitch-yaw to rotation vector."""
        return R.from_euler("xyz", rpy).as_rotvec()

    def step_action(
        self,
        action,
        speed=0.5,  # m/s (unused here, kept for interface compatibility)
        acceleration=0.5,  # m/s^2 (unused here)
        blocking=True,
    ):
        # If coming from idle, sync target to current pose and enable control
        if not self._control_active:
            print("[URClient] Waking up control loop from idle.")
            self._sync_target_to_actual()
            self._control_active = True

        action = np.asarray(action, dtype=np.float32)
        current_pose = np.array(self.rtde_r.getActualTCPPose(), dtype=np.float32)  # [x,y,z,rx,ry,rz]
        current_rpy = self._rotvec2rpy(current_pose[3:6])

        # Clip deltas to avoid overly large jumps per step
        delta_xyz = np.clip(action[0:3], -0.1, 0.1)
        delta_rpy = np.clip(action[3:6], -0.1, 0.1)

        new_rpy = current_rpy + delta_rpy
        new_rotvec = self._rpy2rotvec(new_rpy)

        new_pose = current_pose.copy()
        new_pose[0:3] += delta_xyz
        new_pose[3:6] = new_rotvec

        with self._pose_lock:
            self.target_pose = new_pose.tolist()

        gripper_command = (
            self.gripper.get_open_position() if (action[-1] > 0) else self.gripper.get_closed_position()
        )
        self.gripper.move(position=gripper_command, speed=64, force=1)

    def move(self, pose, speed=0.1, acceleration=0.1, blocking=True):
        print("[URClient] move command received.")
        pose = np.asarray(pose, dtype=np.float32)

        # Pause the control loop while doing a blocking moveL
        self._control_active = False
        
        try:
            self.rtde_c.servoStop()
        except Exception:
            pass

        try:
            self.rtde_c.moveL(pose.tolist(), speed, acceleration, asynchronous=(not blocking))
        except Exception as e:
            print("[move] moveL error:", e)

        try:
            self.gripper.move_and_wait_for_pos(
                position=self.gripper.get_open_position(),
                speed=64,
                force=1,
            )
        except Exception as e:
            print("[move] gripper error:", e)
            
        # After move, sync target to actual so the servo loop continues smoothly
        self._sync_target_to_actual()
        print(f"Target pose: {self.target_pose}")