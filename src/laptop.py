"""
Copyright (c) 2023 The uos_sess6072_build Authors.
Authors: Miquel Massot, Blair Thornton, Sam Fenton
All rights reserved.
Licensed under the BSD 3-Clause License.
See LICENSE.md file in the project root for full license information.
"""
import numpy as np
import argparse
from datetime import datetime, timezone
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
import numpy as np


class LaptopPilot:
    # def __init__(self, simulation, self.kg, self,kn, self.tau_s):
    def __init__(self, simulation):

        # network for sensed pose
        aruco_params = {
            "port": 50000,  # Port to listen to (DO NOT CHANGE)
            "marker_id": 25,  # Marker ID to listen to (CHANGE THIS to your marker ID)            
        }
        self.robot_ip = "192.168.90.1"
        
        # handles different time reference, network amd aruco parameters for simulator
        self.sim_time_offset = 0 #used to deal with webots timestamps
        self.sim_init = False #used to deal with webots timestamps
        self.simulation = simulation
        if self.simulation:
            self.robot_ip = "127.0.0.1"          
            aruco_params['marker_id'] = 0  #Ovewrites Aruco marker ID to 0 (needed for simulation)
            self.sim_init = True #used to deal with webots timestamps

        print("Connecting to robot with IP", self.robot_ip)
        self.aruco_driver = ArUcoUDPDriver(aruco_params, parent=self)

        ############# INITIALISE ATTRIBUTES ##########       

        self.velocity = 0.075  # velocity in m/s
        self.acceleration = 0.1 # acceleration in m/s^2
        self.turning_radius = 0.2 # turning radius in meters
        self.acceptance_radius = 0.1  # acceptance radius in meters

        self.t_prev = 0  # previous time
        self.t = 0 #elapsed time

        # modelling parameters
        wheel_distance = 0.16  #m wheel seperation to centreline
        wheel_diameter = 0.074 #m wheel diameter
        self.ddrive = ActuatorConfiguration(wheel_distance, wheel_diameter) #look at your tutorial and see how to use this class
        
        self.initialise_pose = True # False once the pose is initialised  

        # waypoint for octagon
        # self.northings_path = [0.5, 1, 1.5, 1.5, 1, 0.5, 0.25, 0.25]
        # self.eastings_path = [0.25, 0.25, 0.75, 1.25, 1.65, 1.65, 1.25, 0.75]

        self.northings_path = [0.25, 1.6, 1.6, 0.25]
        self.eastings_path = [0.25, 0.25, 1.5, 1.5]
        self.relative_path = False # False if you want it to be absolute , True it will offset based on 1st point but the same shape   

        self.path = TrajectoryGenerate(self.northings_path, self.eastings_path ) #initialise the path

        # control parameters     
        # self.tau_s = 0.5 # s to remove along track error
   
        self.tau_s = 0.3 # s to remove along track error
        self.L = 0.1  # m distance to remove normal and angular error
        self.v_max = 0.2  # fastest the robot can go
        self.w_max = np.deg2rad(30)  # fastest the robot can turn

        self.k_s = 1 / self.tau_s  # ks
        self.k_n = 0.1  # kn
        self.k_g = 0.1 # kg

        self.initialise_control = True  # False once control gains are initialized, dont change

        # model pose
        self.est_pose_northings_m = 0
        self.est_pose_eastings_m = 0
        self.est_pose_yaw_rad = 0

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

        lidar_xb = 0.03 # location of lidar centre in b-frame primary axis, m
        lidar_yb = 0.03 # location of lidar centre in b-frame secondary axis, m 
        self.lidar = RangeAngleKinematics(lidar_xb,lidar_yb) 
        ###############################################################        

        self.datalog = DataLogger(log_dir="logs")

        # Wheels speeds in rad/s are encoded as a Vector3 with timestamp, 
        # with x for the right wheel and y for the left wheel.        
        self.wheel_speed_pub = Publisher(
            "/wheel_speeds_cmd", Vector3Stamped, ip=self.robot_ip
        )

        self.true_wheel_speed_sub = Subscriber(
            "/true_wheel_speeds",Vector3Stamped, self.true_wheel_speeds_callback,ip=self.robot_ip,
        )
        self.lidar_sub = Subscriber(
            "/lidar", LaserScan, self.lidar_callback, ip=self.robot_ip
        )
        self.groundtruth_sub = Subscriber(
            "/groundtruth", Pose, self.groundtruth_callback, ip=self.robot_ip
        )
                    
    def true_wheel_speeds_callback(self, msg):
        print("Received sensed wheel speeds: R=", msg.vector.x,", L=", msg.vector.y)
        # update wheel rates
        self.measured_wheelrate_right = msg.vector.x
        self.measured_wheelrate_left = msg.vector.y
        self.datalog.log(msg, topic_name="/true_wheel_speeds")

    def lidar_callback(self, msg):
        # This is a callback function that is called whenever a message is received        
        print("Received lidar message", msg.header.seq)
        
        if self.sim_init == True:
            self.sim_time_offset = datetime.now(timezone.utc).timestamp() - msg.header.stamp
            self.sim_init = False     

        msg.header.stamp += self.sim_time_offset

        self.lidar_timestamp_s = msg.header.stamp  # we want the lidar measurement timestamp here

        # b to e frame
        p_eb = Vector(3)
        p_eb[0] = self.measured_pose_northings_m  # robot pose northings (see Task 3)
        p_eb[1] = self.measured_pose_eastings_m  # robot pose eastings (see Task 3)
        p_eb[2] = self.measured_pose_yaw_rad  # robot pose yaw (see Task 3)

        # m to e frame
        self.lidar_data = np.zeros((len(msg.ranges), 2))  # specify length of the lidar data

        z_lm = Vector(2)
        # for each map measurement
        for i in range(len(msg.ranges)):
            z_lm[0] = msg.ranges[i]
            z_lm[1] = msg.angles[i]

            t_em = self.lidar.rangeangle_to_loc(p_eb, z_lm)  # see tutorial

            self.lidar_data[i, 0] = t_em[0]
            self.lidar_data[i, 1] = t_em[1]

        # this filters out any NaN values
        self.lidar_data = self.lidar_data[~np.isnan(self.lidar_data).any(axis=1)]
        self.datalog.log(msg, topic_name="/lidar")

    def true_wheel_speeds_callback(self, msg):
        # print("Received sensed wheel speeds: R=", msg.vector.x, ", L=", msg.vector.y)

        # update wheel rates
        self.measured_wheelrate_right = msg.vector.x
        self.measured_wheelrate_left = msg.vector.y
        self.datalog.log(msg, topic_name="/true_wheel_speeds")


    def groundtruth_callback(self, msg):
        """This callback receives the odometry ground truth from the simulator."""
        self.datalog.log(msg, topic_name="/groundtruth")
    
    def pose_parse(self, msg, aruco = False):
        # parser converts pose data to a standard format for logging
        time_stamp = msg[0]
        if aruco == True:
            if self.sim_init == True:
                self.sim_time_offset = datetime.now(timezone.utc).timestamp()-msg[0]
                self.sim_init = False                                         
                
            # self.sim_time_offset is 0 if not a simulation. Deals with webots dealing in elapse timeself.sim_time_offset
            # print(
            #     "Received position update from",
            #     datetime.now(timezone.utc).timestamp() - msg[0] - self.sim_time_offset,
            #     "seconds ago",
            # )
            time_stamp = msg[0] + self.sim_time_offset                

        pose_msg = PoseStamped() 
        pose_msg.header = Header()
        pose_msg.header.stamp = time_stamp
        pose_msg.pose.position.x = msg[1]
        pose_msg.pose.position.y = msg[2]
        pose_msg.pose.position.z = 0

        quat = Quaternion()        
        if self.simulation == False and aruco == True: quat.from_euler(0, 0, np.deg2rad(msg[6]))
        else: quat.from_euler(0, 0, msg[6])
        pose_msg.pose.orientation = quat        
        return pose_msg

    def generate_trajectory(self):
        # pick waypoints as current pose relative or absolute northings and eastings
        if self.relative_path == True:

            for i in range(len(self.northings_path)):
                self.northings_path[i] += self.measured_pose_northings_m #offset by current northings
                self.eastings_path[i] += self.measured_pose_eastings_m  #offset by current eastings

            # convert path to matrix and create a trajectory class instance
            # C = l2m([self.northings_path, self.eastings_path])       
            self.path = TrajectoryGenerate(self.northings_path,self.eastings_path)     
            
            # set trajectory variables (velocity, acceleration and turning arc radius)
            self.path.path_to_trajectory(self.velocity, self.acceleration) #velocity and acceleration
            self.path.turning_arcs(self.turning_radius) #turning radius
            # self.path.wp_id = len(self.path.Tp_arc) #initialises the next waypoint
            self.path.wp_id = 0 #initialises the next waypoint

            # self.path.t_complete = np.nan # will log when the trajectory was complete 
            print('Trajectory wp timestamps\n',self.path.Tp_arc,'s')

    # def extended_kalman_filter(self):

    def run(self, time_to_run=-1):
        self.start_time = datetime.now(timezone.utc).timestamp()
        
        try:
            r = Rate(10.0)
            while True:
                current_time = datetime.now(timezone.utc).timestamp()
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
                            
        aruco_pose = self.aruco_driver.read()    

        if aruco_pose is not None:
            
            # <code that parses aruco and logs the topic>
            msg = self.pose_parse(aruco_pose, aruco = True)

            # reads sensed pose for local use
            self.measured_pose_timestamp_s = msg.header.stamp
            self.measured_pose_northings_m = msg.pose.position.x
            self.measured_pose_eastings_m = msg.pose.position.y
            _, _, self.measured_pose_yaw_rad = msg.pose.orientation.to_euler()        
            self.measured_pose_yaw_rad = self.measured_pose_yaw_rad % (np.pi*2) # manage angle wrapping
            print('measured_pose_yaw_rad: ', self.measured_pose_yaw_rad)

            self.datalog.log(msg, topic_name="/aruco")

            ###### wait for the first sensor info to initialize the pose ######
            if self.initialise_pose == True:

                self.est_pose_northings_m = self.measured_pose_northings_m
                self.est_pose_eastings_m = self.measured_pose_eastings_m
                self.est_pose_yaw_rad = self.measured_pose_yaw_rad

                # get current time and determine timestep
                self.t_prev = datetime.now(timezone.utc).timestamp() #initialise the time
                self.t = 0 #elapsed time
                time.sleep(0.1) #wait for approx a timestep before proceeding
                
                self.generate_trajectory()
                self.initialise_pose = False 

        if self.initialise_pose != True:  

            #determine the time step
            t_now = datetime.now(timezone.utc).timestamp()   
            dt = t_now - self.t_prev #timestep from last estimate
            self.t += dt #add to the elapsed time
            self.t_prev = t_now #update the previous timestep for the next loop

            # convert true wheel speeds into twist
            q = Vector(2) #twist, m/s, rad/s
            q[0] = self.measured_wheelrate_right  # wheel rate rad/s (measured)
            q[1] = self.measured_wheelrate_left  # wheel rate rad/s (measured)
            
            u = self.ddrive.fwd_kinematics(q)

            # take current pose estimate and update by twist            
            p_robot = Vector(3)
            p_robot[0,0] = self.est_pose_northings_m
            p_robot[1,0] = self.est_pose_eastings_m
            p_robot[2,0] = self.est_pose_yaw_rad

            # might be conflicting here as we meausres the wheelrate convert to twist and take it for the next robot pose 
            p_robot = rigid_body_kinematics(p_robot, u, dt) #est robot position aftter twist, t+1
            p_robot[2] = p_robot[2] % (2 * np.pi)            
            # these are new robot pose estimates

            # # update for show_laptop.py    , this one should show after commenting below       
            self.est_pose_northings_m = p_robot[0,0]
            self.est_pose_eastings_m = p_robot[1,0]
            self.est_pose_yaw_rad = p_robot[2,0]
            
            # logs the data             
            msg = self.pose_parse([datetime.now(timezone.utc).timestamp(),self.est_pose_northings_m,self.est_pose_eastings_m,0,0,0,self.est_pose_yaw_rad])
            self.datalog.log(msg, topic_name="/aruco")
            self.datalog.log(msg, topic_name="/est_pose")

            #################### Trajectory sample #################################    
            # feedforward control: check wp progress and sample reference trajectory
            self.path.path_to_trajectory(self.velocity, self.acceleration)
            self.path.turning_arcs(self.turning_radius)

            # p_robot[0,0] = self.measured_pose_northings_m 
            # p_robot[1,0] = self.measured_pose_eastings_m
            # p_robot[2,0] = self.measured_pose_yaw_rad

            self.path.wp_progress(self.t, p_robot, self.acceptance_radius)

            # print(self.measured_pose_northings_m, self.measured_pose_eastings_m, self.measured_pose_yaw_rad)
            
            p_ref, u_ref = self.path.p_u_sample(self.t)  # sample the path at the current elapsed time
            # print('pref = ' ,  p_ref, 'uref = ' ,u_ref)
            
            # For visualization purposes, set the estimated pose to the reference pose
            self.est_pose_northings_m = p_ref[0,0]
            self.est_pose_eastings_m = p_ref[1,0]
            self.est_pose_yaw_rad = p_ref[2,0]

            # feedback control: get pose change to desired trajectory from body
            dp = p_ref - p_robot  # compute difference between reference and estimated pose in the e-frame
            dp[2] = (dp[2] + np.pi) % (2 * np.pi) - np.pi # handle angle wrapping for yaw
            H_eb = HomogeneousTransformation(p_robot[0:2], p_robot[2])
            ds = Inverse(H_eb.H_R) @ dp    
            # print('probot = ', p_robot, 'dp = ' ,dp)
            # compute control gains for the initial condition (where the robot is stationary)
            
            self.k_s = 1 / self.tau_s  # ks

            if self.initialise_control == True:
                self.k_n = 2 * u_ref[0] / self.L**2  # kn
                self.k_g = u_ref[0] / self.L  # kg
                self.initialise_control = False  # maths changes a bit after the first iteration
            
            # update the controls
            du = feedback_control(ds, self.k_s, self.k_n, self.k_g)

            # total control
            u = u_ref + du  # combine feedback and feedforward control twist components

            # update control gains for the next timestep
            self.k_n = 2 * u[0] / self.L**2  # kn
            self.k_g = u[0] / self.L  # kg
            print('kn kg = ',self.k_n, self.k_g)
            # ensure within performance limitation
            if u[0] > self.w_max: u[0] = self.w_max
            if u[0] < -self.w_max: u[0] = -self.w_max
            if u[1] > self.v_max: u[1] = self.v_max
            if u[1] < -self.v_max: u[1] = -self.v_max
            # print(u)
            # p_robot = rigid_body_kinematics(p_robot,u,dt)
            # p_robot[2] = p_robot[2] % (2*np.pi)
            # actuator commands
            q = self.ddrive.inv_kinematics(u)

            # # update laptop.py
            # p_robot = rigid_body_kinematics(p_robot, u, dt) #est robot position aftter twist
            # p_robot[2] = p_robot[2] % (2 * np.pi)            
            # self.est_pose_northings_m = p_robot[0,0]
            # self.est_pose_eastings_m = p_robot[1,0]
            # self.est_pose_yaw_rad = p_robot[2,0]
            
            # print('q = ', q)
            
            wheel_speed_msg = Vector3Stamped()
            wheel_speed_msg.vector.x = q[0, 0]  # Right wheelspeed rad/s
            wheel_speed_msg.vector.y = q[1, 0]  # Left wheelspeed rad/s
        
            self.cmd_wheelrate_right = wheel_speed_msg.vector.x
            self.cmd_wheelrate_left = wheel_speed_msg.vector.y
     
            self.wheel_speed_pub.publish(wheel_speed_msg)
            self.datalog.log(wheel_speed_msg, topic_name="/wheel_speeds_cmd")
            

    # else :

        # wheel_speed_msg = Vector3Stamped()
        # wheel_speed_msg.vector.x = 0 * np.pi  # Right wheel 1 rev/s = 1*pi rad/s
        # wheel_speed_msg.vector.y = 2 * np.pi  # Left wheel 1 rev/s = 2*pi rad/s

        # self.cmd_wheelrate_right = wheel_speed_msg.vector.x
        # self.cmd_wheelrate_left = wheel_speed_msg.vector.y

        # self.wheel_speed_pub.publish(wheel_speed_msg)
        # self.datalog.log(wheel_speed_msg, topic_name="/wheel_speeds_cmd")

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
