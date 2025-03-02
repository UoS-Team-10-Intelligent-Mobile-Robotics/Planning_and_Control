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
from model_feeg6043 import extended_kalman_filter_predict, extended_kalman_filter_update

class LaptopPilot:
    # def __init__(self, simulation, self.kg, self,kn, self.tau_s):
    def __init__(self, simulation):

        # network for sensed pose
        aruco_params = {
            "port": 50000,  # Port to listen to (DO NOT CHANGE)
            "marker_id": 20,  # Marker ID to listen to (CHANGE THIS to your marker ID)            
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

        self.velocity = 0.1  # velocity in m/s
        self.acceleration = 0.1/3 # acceleration in m/s^2
        self.turning_radius = 0.45  # turning radius in meters
        # self.turning_radius = 0.495  # turning radius in meters

        self.t_prev = 0  # previous time
        self.t = 0 #elapsed time

        # modelling parameters
        wheel_distance = 0.16  #m wheel seperation to centreline
        wheel_diameter = 0.074 #m wheel diameter
        self.ddrive = ActuatorConfiguration(wheel_distance, wheel_diameter) #look at your tutorial and see how to use this class
        
        self.initialise_pose = True # False once the pose is initialised  

        #waypoint
        self.northings_path= [0.25, 1.25 , 1.25  , 0.25, 0.25]
        self.eastings_path = [0.25 ,0.25 ,1.25 , 1.25  , 0.25]
        self.relative_path = False # False if you want it to be absolute       

        self.path = TrajectoryGenerate(self.northings_path, self.eastings_path ) #initialise the path

        # control parameters     
        # self.tau_s = 0.5 # s to remove along track error
   
        self.tau_s = 0.25 # s to remove along track error
        self.L = 0.02  # m distance to remove normal and angular error
        self.v_max = 0.2  # fastest the robot can go
        self.w_max = np.deg2rad(30)  # fastest the robot can turn

        self.k_s = 1 / self.tau_s  # ks
        self.k_n = 0.1  # kn
        self.k_g = 0.1 # kg

        self.initialise_control = True  # False once control gains are initialized

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
        print("Received sensed wheel speeds: R=", msg.vector.x, ", L=", msg.vector.y)

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
            print(
                "Received position update from",
                datetime.now(timezone.utc).timestamp() - msg[0] - self.sim_time_offset,
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
            C = l2m([self.northings_path, self.eastings_path])       
            self.path = TrajectoryGenerate(C[:,0],C[:,1])     
            
            # set trajectory variables (velocity, acceleration and turning arc radius)

            self.path.path_to_trajectory(self.velocity, self.acceleration) #velocity and acceleration
            self.path.turning_arcs(self.turning_radius) #turning radius
            self.path.wp_id = 0 #initialises the next waypoint
            # self.path.t_complete = np.nan # will log when the trajectory was complete 
            print('Trajectory wp timestamps\n',self.path.Tp_arc,'s')


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

            # convert true wheel speeds into twist
            q = Vector(2)
            q[0] = self.measured_wheelrate_right  # wheel rate rad/s (measured)
            q[1] = self.measured_wheelrate_left  # wheel rate rad/s (measured)
            
            u = self.ddrive.fwd_kinematics(q)
            #determine the time step
            t_now = datetime.now(timezone.utc).timestamp()   
            dt = t_now - self.t_prev #timestep from last estimate
            self.t += dt #add to the elapsed time
            self.t_prev = t_now #update the previous timestep for the next loop

            # take current pose estimate and update by twist            
            p_robot = Vector(3)
            p_robot[0,0] = self.est_pose_northings_m
            p_robot[1,0] = self.est_pose_eastings_m
            p_robot[2,0] = self.est_pose_yaw_rad

                                
            p_robot = rigid_body_kinematics(p_robot, u, dt)
            p_robot[2] = p_robot[2] % (2 * np.pi)  # deal with angle wrapping          

            # update for show_laptop.py            
            self.est_pose_northings_m = p_robot[0,0]
            self.est_pose_eastings_m = p_robot[1,0]
            self.est_pose_yaw_rad = p_robot[2,0]
            
            # print("Estimated Position:", self.est_pose_northings_m, self.est_pose_eastings_m, self.est_pose_yaw_rad)
            # print("Measured Position:", self.measured_pose_northings_m, self.measured_pose_eastings_m, self.measured_pose_yaw_rad)

            # logs the data             
            msg = self.pose_parse([datetime.now(timezone.utc).timestamp(),self.est_pose_northings_m,self.est_pose_eastings_m,0,0,0,self.est_pose_yaw_rad])
            self.datalog.log(msg, topic_name="/aruco")
            self.datalog.log(msg, topic_name="/est_pose")

            #################### Trajectory sample #################################    
            # feedforward control: check wp progress and sample reference trajectory
            self.path.path_to_trajectory(self.velocity, self.acceleration)
            self.path.turning_arcs(self.turning_radius)
            self.path.wp_progress(self.t, p_robot, self.turning_radius)
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

            # ensure within performance limitation
            if u[0] > self.w_max: u[0] = self.w_max
            if u[0] < -self.w_max: u[0] = -self.w_max
            if u[1] > self.v_max: u[1] = self.v_max
            if u[1] < -self.v_max: u[1] = -self.v_max

            # p_robot = rigid_body_kinematics(p_robot,u,dt)
            # p_robot[2] = p_robot[2] % (2*np.pi)
            # actuator commands
            q = self.ddrive.inv_kinematics(u)

            wheel_speed_msg = Vector3Stamped()
            wheel_speed_msg.vector.x = q[0, 0]  # Right wheelspeed rad/s
            wheel_speed_msg.vector.y = q[1, 0]  # Left wheelspeed rad/s
        
            self.cmd_wheelrate_right = wheel_speed_msg.vector.x
            self.cmd_wheelrate_left = wheel_speed_msg.vector.y
     
            self.wheel_speed_pub.publish(wheel_speed_msg)
            self.datalog.log(wheel_speed_msg, topic_name="/wheel_speeds_cmd")
            

    # else :

        wheel_speed_msg = Vector3Stamped()
        wheel_speed_msg.vector.x = 0 * np.pi  # Right wheel 1 rev/s = 1*pi rad/s
        wheel_speed_msg.vector.y = 2 * np.pi  # Left wheel 1 rev/s = 2*pi rad/s

        self.cmd_wheelrate_right = wheel_speed_msg.vector.x
        self.cmd_wheelrate_left = wheel_speed_msg.vector.y

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
    
#Defining the extended kalman filter
def extended_kalman_filter_predict(mu, Sigma, u, f, R, dt):
    # (1) Project the state forward
    pred_mu, F = f(mu, u, dt)
      
    # (2) Project the error forward: 
    pred_Sigma = (F @ Sigma @ F.T) + R
    
    # Return the predicted state and the covariance
    return pred_mu, pred_Sigma

def extended_kalman_filter_update(mu, Sigma, z, h, Q, wrap_index = None):
    
    # Prepare the estimated measurement
    pred_z, H = h(mu)
 
    # (3) Compute the Kalman gain
    K = Sigma @ H.T @ np.linalg.inv(H @ Sigma @ H.T + Q)
    
    # (4) Compute the updated state estimate
    delta_z = z- pred_z        
    if wrap_index != None: delta_z[wrap_index] = (delta_z[wrap_index] + np.pi) % (2 * np.pi) - np.pi    
    cor_mu = mu + K @ (delta_z)

    # (5) Compute the updated state covariance
    cor_Sigma = (np.eye(mu.shape[0], dtype=float) - K @ H) @ Sigma
    
    # Return the state and the covariance
    return cor_mu, cor_Sigma

#Ekf motion model implementation
N = 0
E = 1
G = 2
DOTX = 3
DOTG = 4

def motion_model(state, u, dt):
        
    N_k_1 = state[N]
    E_k_1 = state[E]
    G_k_1 = state[G]
    DOTX_k_1 = state[DOTX]
    DOTG_k_1 = state[DOTG]

    p = Vector(3)
    p[0] = N_k_1
    p[1] = E_k_1
    p[2] = G_k_1
    
    # note rigid_body_kinematics already handles the exception dynamics of w=0
    p = rigid_body_kinematics(p,u,dt)    

    # vertically joins two vectors together
    state = np.vstack((p, u))
    
    N_k = state[N]
    E_k = state[E]
    G_k = state[G]
    DOTX_k = state[DOTX]
    DOTG_k = state[DOTG]
    
    # Compute its jacobian
    F = Identity(5)    
    
    if abs(DOTG_k) <1E-2: # caters for zero angular rate, but uses a threshold to avoid numerical instability
        F[N, G] = -DOTX_k * dt * np.sin(G_k_1)
        F[N, DOTX] = dt * np.cos(G_k_1)
        F[E, G] = DOTX_k * dt * np.cos(G_k_1)
        F[E, DOTX] = dt * np.sin(G_k_1)
        F[G, DOTG] = dt       
        
    else:
        F[N, G] = (DOTX_k/DOTG_k)*(np.cos(G_k)-np.cos(G_k_1))
        F[N, DOTX] = (1/DOTG_k)*(np.sin(G_k)-np.sin(G_k_1))
        F[N, DOTG] = (DOTX_k/(DOTG_k**2))*(np.sin(G_k_1)-np.sin(G_k))+(DOTX_k*dt/DOTG_k)*np.cos(G_k)
        F[E, G] = (DOTX_k/DOTG_k)*(np.sin(G_k)-np.sin(G_k_1))
        F[E, DOTX] = (1/DOTG_k)*(np.cos(G_k_1)-np.cos(G_k))
        F[E, DOTG] = (DOTX_k/(DOTG_k**2))*(np.cos(G_k)-np.cos(G_k_1))+(DOTX_k*dt/DOTG_k)*np.sin(G_k)
        F[G, DOTG] = dt

    return state, F

#v define a zero matrix and a one matrix
zm = Matrix(1,1)
om = Matrix(1,1); om[0,0] = 1

# Previous belief and timestep
state = zm; covariance = om 
dt = 1

# Set control and process noise
u = 2*om
R = 2*om

# Set measurement and measurement noise 
z = -2*om

# Task 2
Q = np.multiply(om, 10E-10)
view_flag = True

pred_state, pred_covariance = extended_kalman_filter_predict(state, covariance, u, f_nonlin, R, dt,view_flag=view_flag)

# Task 2
cor_state, cor_covariance = extended_kalman_filter_update(pred_state, pred_covariance, z, h, Q,view_flag=view_flag)

z_list = [0.5, 1.0]
Z = l2m(z_list)
tz_list = [2, 4]
Q = 2*om
idx = 0 #initial pointer to check for sensor measurements
###################################################################
# Previous belief and timestep
###################################################################
state = zm; covariance = om 
t_prev=0


X = Vector(0)
COV = Vector(0)

#################################################################
# Main simulation loop
#################################################################
for i in range(len(T)):

    # get next timestamp to model to and compute time interval ()
    t=T[i]
    dt = t-t_prev
    
    # state, covariance = ?? # Task 1
    state, covariance = extended_kalman_filter_predict(state, covariance, u, f_nonlin, R, dt, view_flag = True)
    print('Time is', t, 's', 'control is', u, 'state is', state, 'covariance is', covariance)
    
    if t == tz_list[idx]:
        z = Z[idx:idx+1]        
        # state, covariance = ?? # Task 2
        state, covariance = extended_kalman_filter_update(state, covariance, z, h, Q, view_flag = True)
        if idx < len(tz_list)-1: idx+=1        
   
    # store state and covariance
    X = np.vstack((X, state))
    COV = np.vstack((COV, covariance))
    
    # store timestamp to calculate dt
    t_prev = t

    # Easy names for indexing
N = 0
E = 1
G = 2
DOTX = 3
DOTG = 4

def motion_model(state, u, dt):
        
    N_k_1 = state[N]
    E_k_1 = state[E]
    G_k_1 = state[G]
    DOTX_k_1 = state[DOTX]
    DOTG_k_1 = state[DOTG]

    p = Vector(3)
    p[0] = N_k_1
    p[1] = E_k_1
    p[2] = G_k_1
    
    # note rigid_body_kinematics already handles the exception dynamics of w=0
    p = rigid_body_kinematics(p,u,dt)    

    # vertically joins two vectors together
    state = np.vstack((p, u))
    
    N_k = state[N]
    E_k = state[E]
    G_k = state[G]
    DOTX_k = state[DOTX]
    DOTG_k = state[DOTG]
    
    # Compute its jacobian
    F = Identity(5)    
    
    if abs(DOTG_k) <1E-2: # caters for zero angular rate, but uses a threshold to avoid numerical instability
        F[N, G] = -DOTX_k * dt * np.sin(G_k_1)
        F[N, DOTX] = dt * np.cos(G_k_1)
        F[E, G] = DOTX_k * dt * np.cos(G_k_1)
        F[E, DOTX] = dt * np.sin(G_k_1)
        F[G, DOTG] = dt       
        
    else:
        F[N, G] = (DOTX_k/DOTG_k)*(np.cos(G_k)-np.cos(G_k_1))
        F[N, DOTX] = (1/DOTG_k)*(np.sin(G_k)-np.sin(G_k_1))
        F[N, DOTG] = (DOTX_k/(DOTG_k**2))*(np.sin(G_k_1)-np.sin(G_k))+(DOTX_k*dt/DOTG_k)*np.cos(G_k)
        F[E, G] = (DOTX_k/DOTG_k)*(np.sin(G_k)-np.sin(G_k_1))
        F[E, DOTX] = (1/DOTG_k)*(np.cos(G_k_1)-np.cos(G_k))
        F[E, DOTG] = (DOTX_k/(DOTG_k**2))*(np.cos(G_k)-np.cos(G_k_1))+(DOTX_k*dt/DOTG_k)*np.sin(G_k)
        F[G, DOTG] = dt

    return state, F

start = 0
end = 52
timestep = 1 #s
num_points = int( end / timestep )+1
t_list = np.linspace(start, end, num_points)
v_list = []
w_list = []
for i in range(len(t_list)):
    if t_list[i] <= 10:
        v_list.append(2)
        w_list.append(np.deg2rad(0.0))
    elif t_list[i] <= 14:
        v_list.append(0)
        w_list.append(np.deg2rad(90/4))
    elif t_list[i] <= 24:
        v_list.append(2)
        w_list.append(np.deg2rad(0.0))
    elif t_list[i] <= 28:
        v_list.append(0)
        w_list.append(np.deg2rad(90/4))
    elif t_list[i] <= 38:
        v_list.append(2)
        w_list.append(np.deg2rad(0.0))
    elif t_list[i] <= 42:
        v_list.append(0)
        w_list.append(np.deg2rad(90/4))    
    else:
        v_list.append(2)
        w_list.append(np.deg2rad(0.0))   
    
T_u = l2m(t_list)
U = l2m([v_list,w_list])

# initialise states
state = Vector(5)
X = state.T
t=copy.copy(T_u[0])

# counter to point to the next control timestep, flag to show an event just happend
idx = 0


# simulation in a while loop
while t <T_u[-1]:
    
    # calculate dt
    dt= T_u[idx] - t       
        
    # apply the model with to progress to dt with old control values
    state, F = motion_model(state, U[idx:idx+1,:].T, dt)    
        
    # progress time and store state
    X = np.vstack((X, state.T))        
        
    t = T_u[idx]
    if idx != len(T_u): idx += 1
 
    
# plot the path
P = X[:,0:3]

# initial state and covariance
state = init_state
covariance = init_covariance

# containors to store the data
X = state.T

COV = init_covariance.T

# containors for control and measyrenebts

# define pointers for our control and measurement data
u_idx = 0
g_idx = 0
ne_idx = 0

# get initial and timestamp and create containor to store times
t_prev=np.min([T_u[0],T_g[0],T_ne[0]])
t_end=np.max([T_u[-1],T_g[-1],T_ne[-1]]) 
T = Vector(1)
T[0,0] = t_prev

#################################################################
# Main simulation loop
#################################################################
while t_prev < t_end:    
    
    # Check which timestamp to process first
    next_stamp = None
    u_stamp = None
    g_stamp = None
    ne_stamp = None 
          
    if u_idx < len(T_u):
        u_stamp = T_u[u_idx]

    if g_idx < len(T_g):
        g_stamp = T_g[g_idx]

    if ne_idx < len(T_ne):
        ne_stamp = T_ne[ne_idx]     
    
    potential_stamps = [
        u_stamp,
        g_stamp,
        ne_stamp        
    ]
        
    valid_stamps = [i for i in potential_stamps if i is not None]
    
    if valid_stamps:
        t = min(valid_stamps) 
    else:
        break  # If there is no timestep left to predict to, simulation finishes
    
    dt = t - t_prev
    
    if t == u_stamp:
            u = U[u_idx:u_idx+1,:].T
            u_idx += 1
            
    if dt != 0:                              
        state, covariance = extended_kalman_filter_predict(state, covariance, u, motion_model, R, dt)
        
    if t == g_stamp: 
        z = Vector(5)
        Q = Identity(5)        
        
        h = h_g_update
        z[G] = Z_g[g_idx:g_idx+1,:]
        Q[G, G] = G_std[g_idx:g_idx+1]**2
        if g_idx < len(T_g): g_idx+=1 
        
        #Task 3
        state, covariance = extended_kalman_filter_update(state, covariance, z, h, Q, wrap_index = G)      

        
    # position updatee
    
    if t == ne_stamp: 
        z = Vector(5)
        Q = Identity(5)        
        
        h = h_ne_update
        z[N] = Z_ne[ne_idx:ne_idx+1,:][0,0]
        z[E] = Z_ne[ne_idx:ne_idx+1,:][0,1]
        Q[N, N] = NE_std[ne_idx:ne_idx+1][0,0]**2
        Q[E, E] = NE_std[ne_idx:ne_idx+1][0,1]**2
        
        if ne_idx < len(T_ne): ne_idx+=1 

        state, covariance = extended_kalman_filter_update(state, covariance, z, h, Q)  
        
        
    X = np.vstack((X,state.T))
        
    COV = np.vstack((COV,covariance.T))
    T = np.vstack((T,t))
        
    t_prev = t
    
    
plot_EKF_trajectory(X, COV, flip=True, measurements = [Z_ne, NE_std], keyframe = 1)


P = X[:,0:3]
plot_path(P, legend_flag = False, verbose = False)

# plot the heading as a function of time
heading = []

for i in range(len(T)): 
    heading.append(X[i,G])

   
# plot the uncertainties as a function of time
n_std = []
e_std = []
g_std = []
j = len(state)

for i in range(len(T)):    
    n_std.append(np.sqrt(COV[i*j+N,N]))
    e_std.append(np.sqrt(COV[i*j+E,E]))
    g_std.append(np.rad2deg(np.sqrt(COV[i*j+G,G])))

