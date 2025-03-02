"""
Copyright (c) 2023 The uos_sess6072_build Authors.
Authors: Miquel Massot, Blair Thornton, Sam Fenton
All rights reserved.
Licensed under the BSD 3-Clause License.
See LICENSE.md file in the project root for full license information.
"""

import numpy as np
import argparse
from datetime import datetime
import time
from drivers.aruco_udp_driver import ArUcoUDPDriver
from zeroros import Subscriber, Publisher
from zeroros.messages import (
    LaserScan,
    Vector3Stamped,
    Pose,
    PoseStamped,
    Header,
    Quaternion,
)
from zeroros.datalogger import DataLogger
from zeroros.rate import Rate

# add more libraries here
from model_feeg6043 import (
    ActuatorConfiguration,
    rigid_body_kinematics,
    RangeAngleKinematics,
    TrajectoryGenerate,
    feedback_control,
)
from math_feeg6043 import Vector, l2m, Inverse, HomogeneousTransformation


class LaptopPilot:
    def __init__(self, simulation):
        # network for sensed pose
        aruco_params = {
            "port": 50000,  # Port to listen to (DO NOT CHANGE)
            "marker_id": 25,  # Marker ID to listen to (CHANGE THIS to your marker ID)
        }
        self.robot_ip = "192.168.90.1"

        # handles different time reference, network amd aruco parameters for simulator
        self.sim_time_offset = 0  # used to deal with webots timestamps
        self.sim_init = False  # used to deal with webots timestamps
        self.simulation = simulation
        if self.simulation:
            self.robot_ip = "127.0.0.1"
            aruco_params["marker_id"] = (
                0  # Ovewrites Aruco marker ID to 0 (needed for simulation)
            )
            self.sim_init = True  # used to deal with webots timestamps

        print("Connecting to robot with IP", self.robot_ip)
        self.aruco_driver = ArUcoUDPDriver(aruco_params, parent=self)

        ############# INITIALISE ATTRIBUTES ##########
        # path
        self.path = None
        self.northings_path = [
            0.375,
            1.625,
            1.625,
            0.375,
            0.375,
            1.625,
            1.625,
            0.375,
            0.375,
        ]  # create a list of waypoints
        self.eastings_path = [
            0.375,
            0.375,
            1.625,
            1.625,
            0.375,
            0.375,
            1.625,
            1.625,
            0.375,
        ]  # create a list of waypoints
        self.relative_path = False  # False if you want it to be absolute

        # model pose
        self.est_pose_northings_m = 0
        self.est_pose_eastings_m = 0
        self.est_pose_yaw_rad = 0
        self.initialise_pose = True

        # modelling parameters
        wheel_distance = 0.0815  # measure this
        wheel_diameter = 0.065  # measure this
        self.ddrive = ActuatorConfiguration(
            wheel_distance, wheel_diameter
        )  # look at your tutorial and see how to use this

        # measured pose
        self.measured_pose_timestamp_s = None
        self.measured_pose_northings_m = None
        self.measured_pose_eastings_m = None
        self.measured_pose_yaw_rad = None

        # wheel speed commands
        self.cmd_wheelrate_right = None
        self.cmd_wheelrate_left = None

        # encoder/actual wheel speeds
        self.measured_wheelrate_right = None
        self.measured_wheelrate_left = None

        # lidar
        self.lidar_timestamp_s = None
        self.lidar_data = None
        lidar_xb = 0.1  # location of lidar centre in b-frame primary axis
        lidar_yb = 0  # location of lidar centre in b-frame secondary axis
        self.lidar = RangeAngleKinematics(lidar_xb, lidar_yb)

        # trajectory planning parameters
        self.velocity = 0.1  # m/s
        self.acceleration = 0.1 / 3  # takes 3s to get to 0.1m/s
        self.turning_radius = 0.25  # m
        self.acceptance_radius = 0.1  # m

        # control gains
        self.tau_s = 0.2  # s to remove along track error
        self.L = 0.075  # m distance to remove normal and angular error
        self.v_max = 0.4  # fastest the robot can go
        self.w_max = np.deg2rad(60)  # fastest the robot can turn

        self.k_s = 1 / self.tau_s
        self.k_n = 0.1
        self.k_g = 0.1

        self.initialise_control = True  # False once control gains is initialised
        ###############################################################

        self.datalog = DataLogger(log_dir="logs")

        # Wheels speeds in rad/s are encoded as a Vector3 with timestamp,
        # with x for the right wheel and y for the left wheel.
        self.wheel_speed_pub = Publisher(
            "/wheel_speeds_cmd", Vector3Stamped, ip=self.robot_ip
        )

        self.true_wheel_speed_sub = Subscriber(
            "/true_wheel_speeds",
            Vector3Stamped,
            self.true_wheel_speeds_callback,
            ip=self.robot_ip,
        )
        self.lidar_sub = Subscriber(
            "/lidar", LaserScan, self.lidar_callback, ip=self.robot_ip
        )
        self.groundtruth_sub = Subscriber(
            "/groundtruth", Pose, self.groundtruth_callback, ip=self.robot_ip
        )

    def true_wheel_speeds_callback(self, msg):
        # print("Received sensed wheel speeds: R=", msg.vector.x, ", L=", msg.vector.y)

        # update wheel rates
        self.measured_wheelrate_right = msg.vector.x
        self.measured_wheelrate_left = msg.vector.y
        self.datalog.log(msg, topic_name="/true_wheel_speeds")

    def lidar_callback(self, msg):
        # This is a callback function that is called whenever a message is received
        # print("Received lidar message", msg.header.seq)
        if self.sim_init == True:
            self.sim_time_offset = datetime.utcnow().timestamp() - msg.header.stamp
            self.sim_init = False

        msg.header.stamp += self.sim_time_offset

        self.lidar_timestamp_s = (
            msg.header.stamp
        )  # we want the lidar measurement timestamp here

        self.lidar_data = np.zeros(
            (len(msg.ranges), 2)
        )  # specify length of the lidar data
        self.lidar_data[:, 0] = (
            msg.ranges
        )  # use ranges as a placeholder, workout northings in Task 4
        self.lidar_data[:, 0] = (
            msg.angles
        )  # use angles as a placeholder, workout eastings in Task 4
        self.datalog.log(msg, topic_name="/lidar")

        # b to e frame
        p_eb = Vector(3)
        p_eb[0] = self.measured_pose_northings_m
        p_eb[1] = self.measured_pose_eastings_m
        p_eb[2] = self.measured_pose_yaw_rad

        # m to e frame
        self.lidar_data = np.zeros((len(msg.ranges), 2))

        z_lm = Vector(2)
        # for each map measurement
        for i in range(len(msg.ranges)):
            z_lm[0] = msg.ranges[i]
            z_lm[1] = msg.angles[i]

            t_em = self.lidar.rangeangle_to_loc(p_eb, z_lm)

            self.lidar_data[i, 0] = t_em[0]
            self.lidar_data[i, 1] = t_em[1]

        # this filters out any NaN
        self.lidar_data = self.lidar_data[~np.isnan(self.lidar_data).any(axis=1)]

    def groundtruth_callback(self, msg):
        """This callback receives the odometry ground truth from the simulator."""
        self.datalog.log(msg, topic_name="/groundtruth")

    def pose_parse(self, msg, aruco=False):
        # parser converts pose data to a standard format for logging
        time_stamp = msg[0]

        if aruco == True:
            if self.sim_init == True:
                self.sim_time_offset = datetime.utcnow().timestamp() - msg[0]
                self.sim_init = False

            # self.sim_time_offset is 0 if not a simulation. Deals with webots dealing in elapse timeself.sim_time_offset
            # print("Received position update from", datetime.utcnow().timestamp() - msg[0] - self.sim_time_offset, "seconds ago")
            time_stamp = msg[0] + self.sim_time_offset

        pose_msg = PoseStamped()
        pose_msg.header = Header()
        pose_msg.header.stamp = time_stamp
        pose_msg.pose.position.x = msg[1]
        pose_msg.pose.position.y = msg[2]
        pose_msg.pose.position.z = 0

        quat = Quaternion()
        if self.simulation == False and aruco == True:
            quat.from_euler(0, 0, np.deg2rad(msg[6]))
        else:
            quat.from_euler(0, 0, msg[6])
        pose_msg.pose.orientation = quat
        return pose_msg

    def generate_trajectory(self):
        # pick waypoints as current pose relative or absolute northings and eastings
        if self.relative_path == True:
            for i in range(len(self.northings_path)):
                self.northings_path[
                    i
                ] += self.est_pose_northings_m  # offset by current northings
                self.eastings_path[
                    i
                ] += self.est_pose_eastings_m  # offset by current eastings

        # convert path to matrix and create a trajectory class instance
        C = l2m([self.northings_path, self.eastings_path])
        self.path = TrajectoryGenerate(C[:, 0], C[:, 1])

        # set trajectory variables (velocity, acceleration and turning arc radius)
        self.path.path_to_trajectory(
            self.velocity, self.acceleration
        )  # velocity and acceleration
        self.path.turning_arcs(self.turning_radius)  # turning radius
        self.path.wp_id = 0  # initialises the next waypoint

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
        """Main control loop

        Your code should go here.
        """
        # > Sense < #
        # get the latest position measurements
        aruco_pose = self.aruco_driver.read()

        if aruco_pose is not None:
            # converts aruco date to zeroros PoseStamped format
            msg = self.pose_parse(aruco_pose, aruco=True)

            # reads sensed pose for local use
            self.measured_pose_timestamp_s = msg.header.stamp
            self.measured_pose_northings_m = msg.pose.position.x
            self.measured_pose_eastings_m = msg.pose.position.y
            _, _, self.measured_pose_yaw_rad = msg.pose.orientation.to_euler()
            self.measured_pose_yaw_rad = self.measured_pose_yaw_rad % (
                np.pi * 2
            )  # manage angle wrapping

            # logs the data
            self.datalog.log(msg, topic_name="/aruco")

            ###### wait for the first sensor info to initialize the pose ######
            if self.initialise_pose == True:
                self.est_pose_northings_m = self.measured_pose_northings_m
                self.est_pose_eastings_m = self.measured_pose_eastings_m
                self.est_pose_yaw_rad = self.measured_pose_yaw_rad

                # get current time and determine timestep
                self.t_prev = datetime.utcnow().timestamp()  # initialise the time
                self.t = 0  # elapsed time
                time.sleep(0.1)  # wait for approx a timestep before proceeding

                # path and tragectory are initialised
                self.initialise_pose = False
                self.generate_trajectory()

        # > Think < #
        ################################################################################
        #  TODO: Implement your state estimation
        if self.initialise_pose != True:
            # convert true wheel speeds into twist (velocity and angular rate)
            q = Vector(2)
            if (
                self.measured_wheelrate_right != None
                and self.measured_wheelrate_left != None
            ):
                q[0] = self.measured_wheelrate_right  # wheel rate rad/s (measured)
                q[1] = self.measured_wheelrate_left  # wheel rate rad/s (measured)
            u = self.ddrive.fwd_kinematics(q)
            # print("Estimated velocity: ", u[0], "m/s; Estimated angular rate: ", u[1], "rad/s;")

            # determine the time step
            t_now = datetime.utcnow().timestamp()

            dt = t_now - self.t_prev  # timestep from last estimate
            self.t += dt  # add to the elapsed time
            self.t_prev = t_now  # update the previous timestep for the next loop

            # take current pose estimate and update by twist
            p_robot = Vector(3)
            p_robot[0, 0] = self.est_pose_northings_m
            p_robot[1, 0] = self.est_pose_eastings_m
            p_robot[2, 0] = self.est_pose_yaw_rad

            p_robot = rigid_body_kinematics(p_robot, u, dt)
            p_robot[2] = p_robot[2] % (2 * np.pi)  # deal with angle wrapping
            # print("Estimated northings: ", p_robot[0,0], "m; Estimated eastings: ", p_robot[1, 0], "m; Estimated yaw:", p_robot[2,0], "rad;")

            # update for show_laptop.py
            self.est_pose_northings_m = p_robot[0, 0]
            self.est_pose_eastings_m = p_robot[1, 0]
            self.est_pose_yaw_rad = p_robot[2, 0]

            msg = self.pose_parse(
                [
                    datetime.utcnow().timestamp(),
                    self.est_pose_northings_m,
                    self.est_pose_eastings_m,
                    0,
                    0,
                    0,
                    self.est_pose_yaw_rad,
                ]
            )
            self.datalog.log(msg, topic_name="/est_pose")

            ################################################################################
            #  TODO: Implement your controller here

            ##################### Trajectory sample ##########################

            # feedforward control: check wp progress and sample reference trajectory
            self.path.wp_progress(
                self.t, p_robot, self.acceptance_radius
            )  # fill turning radius
            p_ref, u_ref = self.path.p_u_sample(
                self.t
            )  # sample the path at the current elapsetime (i.e. seconds from start of motion modelling)
            # print("Reference northings: ", p_ref[0, 0], "m; Reference eastings: ", p_ref[1, 0], "m; Reference yaw:", p_ref[2, 0], "rad;")
            # print("Reference velocity: ", u_ref[0], "m/s; Reference angular rate: ", u_ref[1], "rad/s;")

            self.est_pose_northings_m = p_ref[0, 0]
            self.est_pose_eastings_m = p_ref[1, 0]
            self.est_pose_yaw_rad = p_ref[2, 0]

            # feedback control: get pose change to desired trajectory from body
            dp = (
                p_ref - p_robot
            )  # compute difference between reference and estimated pose in the e-frame
            dp[2] = (dp[2] + np.pi) % (
                2 * np.pi
            ) - np.pi  # handle angle wrapping for yaw

            H_eb = HomogeneousTransformation(p_robot[0:2], p_robot[2])
            ds = Inverse(H_eb.H_R) @ dp
            # print(
            #     "Northings diff: ",
            #     dp[0, 0],
            #     "m; Eastings diff: ",
            #     dp[1, 0],
            #     "m; Yaw diff:",
            #     dp[2, 0],
            #     "rad;",
            # )
            # compute control gains for the initial condition (where the robot is stationary)
            if self.initialise_control == True:
                self.k_n = 2 * u_ref[0] / (self.L**2)
                self.k_g = u_ref[0] / self.L
                self.initialise_control = (
                    False  # maths changes a bit after the first iteration
                )

            # update the controls
            du = feedback_control(ds, self.k_s, self.k_n, self.k_g)

            # total control
            u = (
                u_ref + du
            )  # combine the feedforward and feedback control twist components

            # ensure within performance limitation
            if u[0] > self.v_max:
                u[0] = self.v_max
            if u[0] < -self.v_max:
                u[0] = -self.v_max
            if u[1] > self.w_max:
                u[1] = self.w_max
            if u[1] < -self.w_max:
                u[1] = -self.w_max

            # update control gains for the next timestep using current velocity, which is stored in u
            self.k_n = 2 * u[0] / (self.L**2)
            self.k_g = u[0] / self.L
            # print("Ks: ", self.k_s, "; Kn: ", self.k_n, "; Kg: ", self.k_g)

            # actuator commands
            q = self.ddrive.inv_kinematics(u)

            wheel_speed_msg = Vector3Stamped()
            wheel_speed_msg.vector.x = q[0, 0]  # Right wheel speed
            wheel_speed_msg.vector.y = q[1, 0]  # Left wheel speed
            # print("New right wheel speed: ", q[0, 0], "rad/s; New left wheel speed: ", q[1, 0], "rad/s")

            self.cmd_wheelrate_right = wheel_speed_msg.vector.x
            self.cmd_wheelrate_left = wheel_speed_msg.vector.y
            ################################################################################

            # > Act < #
            # Send commands to the robot
            self.wheel_speed_pub.publish(wheel_speed_msg)
            self.datalog.log(wheel_speed_msg, topic_name="/wheel_speeds_cmd")


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
