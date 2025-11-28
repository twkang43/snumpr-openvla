"""
URClient definition.
"""

import cv2
import zmq
import pickle
import base64
import threading
import numpy as np
import rtde_control, rtde_receive
from scipy.spatial.transform import Rotation as R
import experiments.robot.snumpr.ur_controller.robotiq as robotiq

class ZMQCameraSubscriber(threading.Thread):
    def __init__(self, host, port, topic_type):
        self._host, self._port, self._topic_type = host, port, topic_type
        self._init_subscriber()

    def _init_subscriber(self):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)
        self.socket.setsockopt(zmq.CONFLATE, 1)
        print('tcp://{}:{}'.format(self._host, self._port))
        self.socket.connect('tcp://{}:{}'.format(self._host, self._port))
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
        if not socks: # timeout
            return None, None

        raw_data = self.socket.recv()

        data = raw_data.lstrip(b"rgb_image ")
        data = pickle.loads(data)

        encoded_data = np.frombuffer(base64.b64decode(data['rgb_image']), dtype=np.uint8)
        img = cv2.imdecode(encoded_data, cv2.IMREAD_COLOR)
        return img, data['timestamp']
            
    def stop(self):
        print('Closing the subscriber socket in {}:{}.'.format(self._host, self._port))
        self.socket.close()
        self.context.term()

class URClient:
    def __init__(self, host: str = "localhost", ur_ip: str = "localhost", port: int = 5556):
        """
        Args:
            :param host: the host ip address
            :param port: the port number
        """
        self.rtde_c = rtde_control.RTDEControlInterface(ur_ip)
        self.rtde_r = rtde_receive.RTDEReceiveInterface(ur_ip)
        
        self.gripper = robotiq.RobotiqGripper()
        self.gripper.connect(ur_ip, 63352)
        self.gripper.activate()
        
        self.image_subscriber = ZMQCameraSubscriber(
            host = host,
            port = 10005,
            topic_type = 'RGB'
        )
        
        print(f"URClient connected to {host}:{port}")
    
    def init(self, env_params: dict, image_size: int = 224):
        """
        Initialize the environment.
            :param env_params: a dict of env params
            :param image_size: the size of the image to return
        """
        self.env_params = env_params
        self.image_size = image_size
    
    def get_observation(self):
        full_image, _ = self.image_subscriber.recv_rgb_image()
        
        if full_image is None:
            return None
        
        w = 250
        image = full_image[:, w:w+720,:]
        image = cv2.resize(image, (224, 224))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        return {"image": image, "full_image": image}
    
    def reset(self):
        pass
    
    def _rpy2rotvec(self, rpy):
        """Convert roll-pitch-yaw to rotation vector."""
        return R.from_euler('xyz', rpy).as_rotvec()
    
    def step_action(
        self, 
        action, 
        spped = 0.5, # m/s
        acceleration = 0.5, # m/s^2
        blocking=True
    ):
        action = np.asarray(action, dtype=np.float32)
        current_pose = np.array(self.rtde_r.getActualTCPPose()) # [x,y,z,rx,ry,rz]
        
        delta_xyz = action[0:3]
        delta_rotvec = self._rpy2rotvec(action[3:6])
        
        new_pose = current_pose.copy()
        new_pose[0:3] += delta_xyz
        new_pose[3:6] += delta_rotvec
        
        self.rtde_c.moveL(
            new_pose.tolist(), spped, acceleration, asynchronous=(not blocking)
        )
        
        gripper_command = self.gripper.get_open_position() if (0 < action[-1]) else self.gripper.get_closed_position()
        self.gripper.move(position=gripper_command, speed=64, force=1)
        
    def move(self, pose, speed=0.1, acceleration=0.1, blocking=True):
        pose = np.asarray(pose, dtype=np.float32)
        self.rtde_c.moveL(
            pose.tolist(), speed, acceleration, asynchronous=(not blocking)
        )
        self.gripper.move_and_wait_for_pos(position=self.gripper.get_open_position(), speed=64, force=1)