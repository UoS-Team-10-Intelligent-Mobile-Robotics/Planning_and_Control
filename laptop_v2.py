import numpy as np
import argparse
from datetime import datetime
import time
from drivers.aruco_udp_driver import ArUcoUDPDriver
from zeroros import Subscriber, Publisher
from zeroros.messages import LaserScan, Vector3Stamped, Pose, PoseStamped, Header, Quaternion
from zeroros.datalogger import DataLogger
from zeroros.rate import Rate
from model_feeg6043 import ActuatorConfiguration
from math_feeg6043 import Vector
from model_feeg6043 import rigid_body_kinematics
from model_feeg6043 import RangeAngleKinematics
from model_feeg6043 import TrajectoryGenerate
from math_feeg6043 import l2m
from model_feeg6043 import feedback_control
from math_feeg6043 import Inverse, HomogeneousTransformation

class LaptopPilot:
    def __init__(self, simulation):
        aruco_params = {
            "port": 50000,
            "marker_id": 25,
        }
        self.robot_ip = "192.168.90.1"
        self.sim_time_offset = 0
        self.sim_init = False
        self.simulation = simulation
        if self.simulation:
            self.robot_ip = "127.0.0.1"
            aruco_params['marker_id'] = 0
            self.sim_init = True

        print("Connecting to robot with IP", self.robot_ip)
        self.aruco_driver = ArUcoUDPDriver(aruco_params, parent=self)

        self.initialise_pose = True
        self.northings_path = [0, 0, 3, 3, 0]
        self.eastings_path = [0, -3, -3, 0, 0]
        self.relative_path = True # False if you want it to be absolute
        
        # control parameters
        self.tau_s = 1 # s to remove along track error
        self.L = 1 # m distance to remove normal and angular error
        self.v_max = 0.2 # fastest the robot can go (need to find out)
        self.w_max = np.deg2rad(30) # fastest the robot can turn (need to find out)

        self.initialise_control = True # False once control gains is initialised

        # Define velocity, acceleration, and turning radius parameters
        self.path_velocity = 0.1  # m/s (set based on robot's capabilities)
        self.path_acceleration = 0.1/3  # m/s² (acceleration limit)
        self.path_turning_radius = 0.375  # meters (radius of turns)
        self.path_dt = 0.1  # Sampling time for trajectory
        self.path = None
        self.est_pose_northings_m = None
        self.est_pose_eastings_m = None
        self.est_pose_yaw_rad = None

        self.measured_pose_timestamp_s = None
        self.measured_pose_northings_m = None
        self.measured_pose_eastings_m = None
        self.measured_pose_yaw_rad = None

        self.cmd_wheelrate_right = None
        self.cmd_wheelrate_left = None

        self.measured_wheelrate_right = None
        self.measured_wheelrate_left = None

        self.lidar_timestamp_s = None
        self.lidar_data = None
        lidar_xb = 0.0
        lidar_yb = 0.0
        self.lidar = RangeAngleKinematics(lidar_xb, lidar_yb)

        wheel_distance = 0.160
        wheel_diameter = 0.074
        self.ddrive = ActuatorConfiguration(wheel_distance, wheel_diameter)

        self.datalog = DataLogger(log_dir="logs")

        self.wheel_speed_pub = Publisher(
            "/wheel_speeds_cmd", Vector3Stamped, ip=self.robot_ip
        )

        self.true_wheel_speed_sub = Subscriber(
            "/true_wheel_speeds", Vector3Stamped, self.true_wheel_speeds_callback, ip=self.robot_ip,
        )
        self.lidar_sub = Subscriber(
            "/lidar", LaserScan, self.lidar_callback, ip=self.robot_ip
        )
        self.groundtruth_sub = Subscriber(
            "/groundtruth", Pose, self.groundtruth_callback, ip=self.robot_ip
        )

    def true_wheel_speeds_callback(self, msg):
        print("Received sensed wheel speeds: R=", msg.vector.x, ", L=", msg.vector.y)
        self.measured_wheelrate_right = msg.vector.x
        self.measured_wheelrate_left = msg.vector.y
        self.datalog.log(msg, topic_name="/true_wheel_speeds")

    def lidar_callback(self, msg):
        """Process Lidar data and convert to Earth frame."""
        print("Received lidar message", msg.header.seq)

        if self.sim_init:
            self.sim_time_offset = datetime.utcnow().timestamp() - msg.header.stamp
            self.sim_init = False

        msg.header.stamp += self.sim_time_offset
        self.lidar_timestamp_s = msg.header.stamp

        p_eb = Vector(3)
        p_eb[0] = self.est_pose_northings_m
        p_eb[1] = self.est_pose_eastings_m
        p_eb[2] = self.est_pose_yaw_rad

        self.lidar_data = np.zeros((len(msg.ranges), 2))
        z_lm = Vector(2)

        for i in range(len(msg.ranges)):
            z_lm[0] = msg.ranges[i]
            z_lm[1] = msg.angles[i]
            t_em = self.lidar.rangeangle_to_loc(p_eb, z_lm)
            self.lidar_data[i, 0] = t_em[0]
            self.lidar_data[i, 1] = t_em[1]

        self.lidar_data = self.lidar_data[~np.isnan(self.lidar_data).any(axis=1)]
        self.datalog.log(msg, topic_name="/lidar")

    def groundtruth_callback(self, msg):
        self.datalog.log(msg, topic_name="/groundtruth")

    def pose_parse(self, msg, aruco=False):
        time_stamp = msg[0]

        if aruco:
            if self.sim_init:
                self.sim_time_offset = datetime.utcnow().timestamp() - msg[0]
                self.sim_init = False

            print(
                "Received position update from",
                datetime.utcnow().timestamp() - msg[0] - self.sim_time_offset,
                "seconds ago",
            )
            time_stamp = msg[0] + self.sim_time_offset

        pose_msg = PoseStamped()
        pose_msg.header = Header()
        pose_msg.header.stamp = time_stamp
        pose_msg.pose.position.x = msg[1]
        pose_msg.pose.position.y = msg[2]
        pose_msg.pose.position.z = 0

        quat = Quaternion()
        if not self.simulation and aruco:
            quat.from_euler(0, 0, np.deg2rad(msg[6]))
        else:
            quat.from_euler(0, 0, msg[6])
        pose_msg.pose.orientation = quat
        return pose_msg
        
    ## Task 1
    def generate_trajectory(self):
        # pick waypoints as current pose relative or absolute northings and eastings
        if self.relative_path == True:
            for i in range(len(self.northings_path)):
                self.northings_path[i] += self.est_pose_northings_m   #offset by current northings
                self.eastings_path[i] += self.est_pose_eastings_m #offset by current eastings

            # convert path to matrix and create a trajectory class instance
            C = l2m([self.northings_path, self.eastings_path])
            self.path = TrajectoryGenerate(C[:,0],C[:,1])
            
            # set trajectory variables (velocity, acceleration and turning arc radius)
            self.path.path_to_trajectory(self.path_velocity, self.path_acceleration) #velocity and acceleration
            self.path.turning_arcs(self.path_turning_radius) #turning radius
            self.path.wp_id=self.path.wp_id #initialises the next waypoint
        
    def run(self, time_to_run=-1):
        self.start_time = datetime.utcnow().timestamp()

        try:
            r = Rate(10.0)
            while True:
                current_time = datetime.utcnow().timestamp()
                if time_to_run > 0 and current_time - self.start_time > time_to_run:
                    print("Time is up, stopping…")
                    break
                self.infinite_loop()
                r.sleep()
        except KeyboardInterrupt:
            print("KeyboardInterrupt received, stopping…")
        except Exception as e:
            print("Exception: ", e)
        finally:
            self.lidar_sub.stop()
            self.groundtruth_sub.stop()
            self.true_wheel_speed_sub.stop()

    def infinite_loop(self):
        aruco_pose = self.aruco_driver.read()

        if aruco_pose is not None:
            msg = self.pose_parse(aruco_pose, aruco=True)
            self.measured_pose_timestamp_s = msg.header.stamp
            self.measured_pose_northings_m = msg.pose.position.x
            self.measured_pose_eastings_m = msg.pose.position.y
            _, _, self.measured_pose_yaw_rad = msg.pose.orientation.to_euler()
            self.measured_pose_yaw_rad = self.measured_pose_yaw_rad % (np.pi * 2)

            if self.initialise_pose:
                self.est_pose_northings_m = self.measured_pose_northings_m
                self.est_pose_eastings_m = self.measured_pose_eastings_m
                self.est_pose_yaw_rad = self.measured_pose_yaw_rad
                self.t_prev = datetime.utcnow().timestamp()
                self.t = 0
                time.sleep(0.1)
                self.initialise_pose = False
                
                # Generate the trajectory after initializing the pose
                self.generate_trajectory()
                print("Trajectory generated after initializing pose.")

            q = Vector(2)
            if self.measured_wheelrate_right is not None and self.measured_wheelrate_left is not None:
                q[0] = self.measured_wheelrate_right
                q[1] = self.measured_wheelrate_left
                u = self.ddrive.fwd_kinematics(q)
                print("Twist motion (linear velocity, angular velocity):", u)
            else:
                print("Warning: Wheel rate measurements not available yet!")
                u = Vector(2)

            self.datalog.log(msg, topic_name="/aruco")

            if not self.initialise_pose:
                t_now = datetime.utcnow().timestamp()
                dt = t_now - self.t_prev
                self.t += dt
                self.t_prev = t_now

                p_robot = Vector(3)
                p_robot[0, 0] = self.est_pose_northings_m
                p_robot[1, 0] = self.est_pose_eastings_m
                p_robot[2, 0] = self.est_pose_yaw_rad

                p_robot = rigid_body_kinematics(p_robot, u, dt)
                p_robot[2] = p_robot[2] % (2 * np.pi)

                self.est_pose_northings_m = p_robot[0]
                self.est_pose_eastings_m = p_robot[1]
                self.est_pose_yaw_rad = p_robot[2]

        wheel_speed_msg = Vector3Stamped()
        wheel_speed_msg.vector.x = 2 * np.pi
        wheel_speed_msg.vector.y = 2 * np.pi

        self.cmd_wheelrate_right = wheel_speed_msg.vector.x
        self.cmd_wheelrate_left = wheel_speed_msg.vector.y

        self.wheel_speed_pub.publish(wheel_speed_msg)
        self.datalog.log(wheel_speed_msg, topic_name="/wheel_speeds_cmd")
        
        #################### Trajectory sample #################################

        # feedforward control: check wp progress and sample reference trajectory
        self.path.wp_progress(self.t, p_robot, self.path_turning_radius) # fill turning radius
        p_ref, u_ref = self.path.p_u_sample(self.t) #sample the path at the current elapsetime (i.e., seconds from start of motion modelling)
        
        # feedback control: get pose change to desired trajectory from body
        dp = p_ref - p_robot #compute difference between reference and estimated pose in the $e$-frame
        dp[2] = (dp[2] + np.pi) % (2 * np.pi) - np.pi # handle angle wrapping for yaw
        H_eb = HomogeneousTransformation(p_robot[0:2],p_robot[2])
        ds = Inverse(H_eb.H_R)@dp   # rotate the $e$-frame difference to get it in the $b$-frame (Hint: dp_b = H_be.H_R @ dp_e)

        # compute control gains for the initial condition (where the robot is stationalry)
        self.k_s = 1/tau_s #ks
        if self.initialise_control == True:
            self.k_n = 2*(u_ref[0])/(L**2) #kn
            self.k_g = u_ref[0]/L #kg
            self.initialise_control = False # maths changes a bit after the first iteration
            
        # update the controls
        du = feedback_control(ds, self.k_s, self.k_n, self.k_g)

        # total control
        u = u_ref + du # combine feedback and feedforward control twist components

        # update control gains for the next timestep
        self.k_n = 2*(u[0])/(L**2) #kn
        self.k_g = u[0]/L #kg
        
        # ensure within performance limitation
        if u[0] > v_max: u[0] = v_max
        if u[0] < v_max: u[0] = v_max
        if u[1] > w_max: u[1] = w_max
        if u[1] < w_max: u[1] = w_max

        # actuator commands
        q = self.ddrive.inv_kinematics(u)

        wheel_speed_msg = Vector3Stamped()
        wheel_speed_msg.vector.x = q[0,0] # Right wheelspeed rad/s
        wheel_speed_msg.vector.y = q[1,0] # Left wheelspeed rad/s


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--time",
        type=float,
        default=-1,
        help="Time to run an experiment for. If negative, run forever.",
    )
    parser.add_argument(
        "--simulation",
        action="store_true",
        help="Run in simulation mode. Defaults to False",
    )

    args = parser.parse_args()

    laptop_pilot = LaptopPilot(args.simulation)
    laptop_pilot.run(args.time)
