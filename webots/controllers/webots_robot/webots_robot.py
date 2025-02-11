import asyncio
import os
from math import atan2
import numpy as np
import json
import socket
from datetime import datetime
from controller import Robot
from zeroros import Subscriber, Publisher
from zeroros.messages import geometry_msgs, sensor_msgs, nav_msgs
from zeroros.message_broker import MessageBroker
from zeroros.rate import Rate


# Check if the platform is windows
if os.name == "nt":
    # Set the event loop policy to avoid the following warning:
    # [...]\site-packages\zmq\_future.py:681: RuntimeWarning:
    # Proactor event loop does not implement add_reader family of methods required for
    # zmq. Registering an additional selector thread for add_reader support via tornado.
    #  Use `asyncio.set_event_loop_policy(WindowsSelectorEventLoopPolicy())` to avoid
    # this warning.
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


class UDPBroadcastServer:
    def __init__(self, ip, port):
        # -- Enable port reusage
        self.socket = socket.socket(
            socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP
        )
        # -- Enable broadcasting mode if feature is available
        if hasattr(socket, "SO_REUSEPORT"):
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
        # -- Enable broadcasting mode
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        self.socket.settimeout(None)
        self.ip = ip
        self.port = port
        

    def broadcast(self, message):
        # -- Broadcast the dictionary as a bytes-like object (string-like info) and empties
        broadcast_string = json.dumps(message, indent=3)
        self.socket.sendto(
            broadcast_string.encode("utf-8"),
            (self.ip, self.port),
        )


class WebotsController(Robot):
    def __init__(self):
        super(WebotsController, self).__init__()
        self.ip = "127.0.0.1"
        self.port = 5600
        self.wheel_distance = 0.162
        self.wheel_radius = 0.037
        self.marker_id = 0
        self.start_time = datetime.utcnow().timestamp()

        self.broker = MessageBroker()
        self.laserscan_pub = Publisher("/lidar", sensor_msgs.LaserScan)
        self.groundtruth_pub = Publisher("/groundtruth", geometry_msgs.Pose)
        self.rpm_sub = Subscriber(
            "/wheel_speeds_cmd", geometry_msgs.Vector3Stamped, self.wheel_speed_callback
        )
        self.true_rpm_pub = Publisher(
            "/true_wheel_speeds", geometry_msgs.Vector3Stamped
        )

        # UDP server to fake Aruco marker detection
        self.udp_server = UDPBroadcastServer("127.0.0.1", 50000)
        self.udp_rate = Rate(1)

        timestep = int(self.getBasicTimeStep())
        self.timeStep = timestep * 10
        print("Timestep:", timestep, "Setting controller timestep: ", self.timeStep)

        self.num_lidar_msgs = 0
        self.pose_msg = geometry_msgs.Pose()

        self.lidar = self.getDevice("lidar")
        self.lidar.enable(self.timeStep)
        self.lidar.enablePointCloud()
        self.last_lidar_timestamp = None
        self.lidar_rate = Rate(self.lidar.getFrequency())
        self.lidar_msg = sensor_msgs.LaserScan()
        self.lidar_msg.header.frame_id = "lidar"
        self.lidar_msg.angle_min = -self.lidar.getFov() / 2.0
        self.lidar_msg.angle_max = self.lidar.getFov() / 2.0
        self.lidar_msg.angle_increment = (
            self.lidar.getFov() / self.lidar.getHorizontalResolution()
        )
        self.lidar_msg.time_increment = self.lidar.getSamplingPeriod() / (
            1000.0 * self.lidar.getHorizontalResolution()
        )
        self.lidar_msg.scan_time = self.lidar.getSamplingPeriod() / 1000.0
        self.lidar_msg.range_min = self.lidar.getMinRange()
        self.lidar_msg.range_max = self.lidar.getMaxRange()
        self.lidar_msg.intensities = np.array([0] * len(self.lidar_msg.ranges))
        self.lidar_msg.angles = np.array(
            [
                self.lidar_msg.angle_min + i * self.lidar_msg.angle_increment
                for i in range(len(self.lidar_msg.ranges))
            ]
        )

        self.gps = self.getDevice("gps")
        self.gps.enable(self.timeStep)

        self.compass = self.getDevice("compass")
        self.compass.enable(self.timeStep)

        self.left_motor = self.getDevice("left wheel motor")
        self.right_motor = self.getDevice("right wheel motor")
        self.left_motor.setPosition(float("inf"))
        self.right_motor.setPosition(float("inf"))
        self.left_motor.setVelocity(0)
        self.right_motor.setVelocity(0)

    def wheel_speed_callback(self, wheel_speed_message):
        wheel_speed_right = wheel_speed_message.vector.x
        wheel_speed_left = wheel_speed_message.vector.y
        print("Received sensed wheel speeds: R=", wheel_speed_right, ", L=", wheel_speed_left, "rad/s")
        self.right_motor.setVelocity(wheel_speed_right)        
        self.left_motor.setVelocity(wheel_speed_left)
        

    def run(self):
        while self.step(self.timeStep) != -1:
            self.infinite_loop()

    def infinite_loop(self):
        if self.lidar_rate.remaining() <= 0:
            #self.last_lidar_timestamp = self.getTime()
            self.last_lidar_timestamp = datetime.utcnow().timestamp() - self.start_time
            msg = self.lidar_msg
            msg.ranges = np.array(self.lidar.getRangeImage())
            # Remove inf values and replace with max range
            msg.ranges[msg.ranges == float("inf")] = 0.0
            msg.angles = np.array(
                [
                    msg.angle_min + i * msg.angle_increment
                    for i in range(len(msg.ranges))
                ]
            )
            # msg.header.stamp = self.getTime()
            msg.header.stamp = datetime.utcnow().timestamp() - self.start_time
            msg.header.seq = self.num_lidar_msgs
            msg.intensities = np.array([0] * len(msg.ranges))
            # If ranges is not Inf, then set intensity to 100
            msg.intensities[msg.ranges != 0.0] = 100
            self.laserscan_pub.publish(msg)
            print("Published lidar message", self.num_lidar_msgs)
            self.num_lidar_msgs += 1
            self.lidar_rate.reset()

        # Publish true wheel speeds
        left_rpm = self.left_motor.getVelocity()
        right_rpm = self.right_motor.getVelocity()
        rpm_msg = geometry_msgs.Vector3Stamped()
        # rpm_msg.header.stamp = self.getTime()
        rpm_msg.header.stamp = datetime.utcnow().timestamp() - self.start_time
        rpm_msg.vector.x = right_rpm
        rpm_msg.vector.y = left_rpm
        self.true_rpm_pub.publish(rpm_msg)

        pose_val = self.gps.getValues()        
        pose_time = datetime.utcnow().timestamp() - self.start_time
        # Switch x and y
        pose_val = [pose_val[0], -pose_val[1], pose_val[2]]
        north = self.compass.getValues()
        # The Compass node returns a vector that indicates the north direction specified
        # by the coordinateSystem field of the WorldInfo node.
        # Transform from ENU to NED
        north = [north[1], north[0], -north[2]]

        ##change angle to clockwise
        angle = atan2(north[1], north[0])
        angle = ((np.pi*2) - angle) % (np.pi*2)
        if angle == (np.pi*2):
            angle = 0

        quaternion = geometry_msgs.Quaternion()
        quaternion.from_euler(0, 0, angle)

        # Publish groundtruth
        self.pose_msg.position.x = pose_val[0]
        self.pose_msg.position.y = pose_val[1]
        self.pose_msg.position.z = pose_val[2]
        self.pose_msg.orientation.x = quaternion.x
        self.pose_msg.orientation.y = quaternion.y
        self.pose_msg.orientation.z = quaternion.z
        self.pose_msg.orientation.w = quaternion.w
        self.groundtruth_pub.publish(self.pose_msg)

        if self.udp_rate.remaining() <= 0:
            # Fake Aruco marker detection
            # Broadcast the pose of the robot
            msg = {}
            msg[self.marker_id] = [
                pose_time,
                pose_time,
                pose_val[0],
                pose_val[1],
                None,
                None,
                None,
                angle,
            ]
            self.udp_server.broadcast(msg)
            self.udp_rate.reset()


wc = WebotsController()
wc.run()
