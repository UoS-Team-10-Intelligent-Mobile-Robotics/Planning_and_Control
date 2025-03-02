import numpy as np
import copy
from matplotlib import pyplot as plt
from math_feeg6043 import Vector,Matrix,Identity,Transpose,Inverse,v2t,t2v,HomogeneousTransformation, interpolate, short_angle, inward_unit_norm, line_intersection, cartesian2polar, polar2cartesian
from plot_feeg6043 import plot_kalman
from numpy.random import randn, random, uniform, multivariate_normal, seed
from scipy.stats import norm
from sklearn.neighbors import KernelDensity


def rigid_body_kinematics(mu,u,dt=0.1):
    #################################################################################
    # This function models the forwards kinematics of rigid body, b
    # It implements a model of form
    #  mu_k = f(mu_k-1,u_k)
    # and progresses the pose of the robot 'mu' (in the fixed frame e)) 
    # to the next timestep 'mu_' by applying combined linear and angular velocity (twist) 
    # as its control action 'u' for the period 'dt'. 
    #
    # The notation of 'mu' is used as it represents the mean estimate of the robot pose p
    #
    # It implements the homogeneous transformation below to achieve this.
    # 
    # Xk = Xk-1 Trans(tbc)Rot(wdt)Trans(tcb')
    #
    # as 
    #
    # H_eb' = H_eb @ H_bc.H_T @ H_cb'.H_R @ H_cb'.H_T
    #
    # Inputs
    #.  mu: The previous pose p_{k-1}, which is a [3x1] matrix of form
    #          [[x]     Northings in m (fixed frame, e )
    #           [y]     Eastings in m (fixed frame, e )
    #           [g]]    Heading in rads (fixed frame, e )
    #.  u:  The control is a [2 x 1] matrix of the form, 
    #         [[v]      Linear forwards velocity in m/s (body frame, b)
    #.         [w]]     Angular velocity in rads/s (body frame, b)
    # Output
    #.  mu: The new pose p_k
    #
    # Dependancies 
    #.  standard python libraries (numpy)
    #.  class HomogeneousTransformation from course library `math_feeg6043.py`
    #################################################################################    
 
    # create an empty containor for the output
    mu_=Vector(3)

    # create an empty containor for the output homogeneous transformation matrix     
    H_eb_ = HomogeneousTransformation()
    
    # convert input pose p_k-1 into a homogeneous transformation matrix 
    t_eb=Vector(2)
    t_eb[0] = mu[0] # northings (p_k-1)
    t_eb[1] = mu[1] #eastings (p_k-1)
    g_eb = mu[2] #heading (p_k-1)
    H_eb = HomogeneousTransformation(t_eb,g_eb)

    tol = 1E-3            

    if abs(u[0])<tol and abs(u[1])<tol:
        # handles the stationary case where 
        H_eb_ = H_eb

    elif abs(u[1])<tol:
        # implement a simpler vesion of the model that doesn't need to compute twist 
        # H_eb_ = H_eb@H_bb' 

        v = u[0] #surge rate        
            
        # compute motion in the body frame due to the pure linear velocity
        t_bb_=Vector(2) # [2x1] matrix of 0
        t_bb_[0] = v*dt
        
        # create the homogeneous transformation from b to b', 
        H_bb_ = HomogeneousTransformation(t_bb_,0) # heading doesn't change as w=0 (u[1]=0]
        
        # left multiply be the homogeneous transformation from the fixed frame e to the body frame b
        H_eb_.H = H_eb.H@H_bb_.H            
            
    else: 
        # implements the model derived for twist
        
        v = u[0] #surge rate
        w = u[1] #yaw rate

        # calculate centre of rotation from the initial body position
        t_bc = Vector(2) # [2x1] matrix of 0                       
        t_bc[1]=v/w      # centre of rotation is v/w in the +ve y direction of the body Eq(A1.2.12)
        
        # the centre of rotation 'c' keeps the heading of the body, so is 0 as seen from the body
        H_bc = HomogeneousTransformation(t_bc,0)
            
        # to rotate the body about 'c' so need 'b' as seen from the centre of rotation
        H_cb = HomogeneousTransformation() 
        H_cb.H = Inverse(H_bc.H)  

        # to rotate the body b around the centre of rotation w*dt while maintain the same radius
        H_cb_ = HomogeneousTransformation(H_cb.t,w*dt)
        
        # we rotate first, and then translate
        H_eb_.H = H_eb.H@H_bc.H@H_cb_.H_R@H_cb_.H_T                      
        
    # create the pose, handling angle wrap at 2pi (in rads)
    mu_[0] = H_eb_.t[0]
    mu_[1] = H_eb_.t[1]
    mu_[2] = H_eb_.gamma % (2 * np.pi ) #(H_eb_.gamma + np.pi) % (2 * np.pi ) - np.pi 
    
    return mu_

class ActuatorConfiguration:

    """
    Class to model kinematics of a 2 wheel differential drive
    Assumes symmetry about the vehicle centre line and the 
    body frame origin coincides with the centre of rotation
    
    Parameters:
    -----------
    W: float
        Perpendicular distance of each wheel to robot's x-axis centre line in metres
    d: float
        Diametre of each wheel in metres

    Attributes:
    -----------
    G: Matrix(2,2)
        Actuator configuration matrix [d/4 d/4; -d/4W d/4W] 
    invG: Matrix(2,2)
        Inverse of actuator configuration matrix [2/d -2W/d; 2/d 2W/d]     
        
    Functions:
    -----------
    inv_kinematics(self, u):
        Models the actuator inverse kinematics and computes wheel 
        commands to achieve desired rigid body twist 

        Input:
            u:  Body twist is a [2 x 1] matrix of the form
                  [[v]      Linear forwards velocity in m/s (body frame, b)
                   [w]]     Angular velocity in rads/s (body frame, b)
        Output:
            q: The wheel rates of each wheel in rad/s, of the form
               [[w_r]    wheel aligned with +ve y-axis of robot
                [w_l]]   wheel aligned with -ve y-axis of robot   

    fwd_kinematics(self, u): 
        Models the actuator kinematics and computes rigid body twist corresponding 
        to provided wheel commands      
        
        Input:
            q: The wheel rates of each wheel in rad/s, of the form
               [[w_r]    wheel aligned with +ve y-axis of robot
                [w_l]]   wheel aligned with -ve y-axis of robot   

        Output:
            u:  Body twist is a [2 x 1] matrix of the form 
                  [[v]      Linear forwards velocity in m/s (body frame, b)
                   [w]]     Angular velocity in rads/s (body frame, b)

    """    
    
    def __init__(self, W, d):
        
        self.W = W
        self.d = d
        self.G = Matrix(2,2)
        self.invG = Matrix(2,2)
        
        self.G[0,0] = d/4
        self.G[0,1] = d/4
        self.G[1,0] = -d/(4*W)
        self.G[1,1] = d/(4*W)
        
        self.invG[0,0] = 2/d
        self.invG[0,1] = -2*W/d                
        self.invG[1,0] = 2/d        
        self.invG[1,1] = 2*W/d                        
    
    def inv_kinematics(self, u): q = self.invG @ u; return q
    def fwd_kinematics(self, q): u = self.G @ q ; return u

    
class RangeAngleKinematics():
    
    """ Class to model the geomtry of range and angle sensor measurements
    
    Parameters:
    -----------
    x_bl: Float
        Location of the sensor in the body frame x-direction in m
    y_bl: Float
        Location of the sensor in the body frame y-direction in m        
    gamma_bs: Float
        Zero angle bearing of the sensor relative to the body frame x-direction in radians   
        Default is 0 (aligned with robot x-axis)
    distance_range: [2 x 1] Float list
        Minimum and aximum distance the sensor can make a measurement in m
        Default is min 0.1 m to max 1 m
    scan_fov: Float
        Maximum range of bearing angles where objects can be detected in rad. Assumed to be centred about gamma_bs with 0.5 scan_fov either side
        Default is np.deg2rad(120), which gives 60degrees either side of the robot

    Attributes:
    -----------
    t_bl: Matrix 2x1
        Location of the sensor $s$ relative to the moving body, defined as
            t_bl = [[x_bl]
                    [y_bl]]
    H_bl: Class HomogenousTranformation()
        Homogenous transformation class instance to map the 
        2D location and orientation of the range bearing sensor $l$ 
        relative to the moving body $b$, using x_bl, y_bl and gamma_bl
        
    Functions:
    -----------
    loc_to_rangeangle(self, p_eb, t_em):
        Models the sensor inverse kinematics to computes range bearing measurements 
        that correspond to the robot observing some map landmark

        Input:
            p_eb:  Pose of the robot in the fixed frame $e$ as a [3 x 1] matrix of the form
                  [[x_eb]      Northings distance in m (fixed frame, e)
                   [y_eb]      Eastings distance in m (fixed frame, e)
                   [gamma_eb]] Heading angle in rads (body frame, b)
                   
            t_em:  Coordinate of the map landmark in the fixed frame $e$ as a [2 x 1] matrix of the form
                  [[x_em]      Northings distance in m (fixed frame, e)
                   [y_em]      Eastings distance in m (fixed frame, e)
        Output:
            z_lm: Sensor Range and bearing observation
               [[r_lm]    range to the map landmark as seen from the sensor, l
                [theta_lm]]   bearing to the map landmark as seen from the sensor, l

    rangeangle_to_loc(self, p_eb, z_lm): 
        Models the sensor kinematics to compute the location of a map landmark (fixed frame, e)
        accoridng to the a sensor observation range and bearing.
        
        Input:
            z_lm: Sensor range and bearing observation
               [[r_lm]    range to the map landmark as seen from the sensor, l
                [theta_lm]]   bearing to the map landmark as seen from the sensor, l
            p_eb:  Pose of the robot in the fixed frame $e$ as a [3 x 1] matrix of the form
                  [[x_eb]      Northings distance in m (fixed frame, e)
                   [y_eb]      Eastings distance in m (fixed frame, e)
                   [gamma_eb]] Heading angle in rads (body frame, b)                
                
        Output:
            t_em:  Coordinate of the map landmark in the fixed frame $e$ as a [2 x 1] matrix of the form
                  [[x_em]      Northings distance in m (fixed frame, e)
                   [y_em]      Eastings distance in m (fixed frame, e)
                   
    check_range(self,z_lm):
        Checks that both the distance_range and the scan_fov are feasible. If not it returns np.nan values
        
        Inputs:
            z_lm: Sensor range and bearing observations as a [2x1] matrix (previously define)
        Output:
            z_lm: Sensor range and bearing observations as a [2x1] matrix (previously define)
                after checking it is within the sensor specification. If not returns np/nan values        
    """
    
    def __init__(self, x_bl, y_bl, gamma_bl = 0, distance_range = [0.1, 1], scan_fov = np.deg2rad(120)):
        
        self.t_bl = Vector(2)
        self.t_bl[0] = x_bl
        self.t_bl[1] = y_bl                
        self.H_bl = HomogeneousTransformation(self.t_bl,gamma_bl)
        self.distance_range = distance_range
        self.scan_fov = scan_fov   
        
        print('*******************************************************************************************')
        print('Sensor located at:',self.t_bl[0],self.t_bl[1],'m offset, at angle:',np.rad2deg(gamma_bl),'deg to the body')
        print('Measurement range is between:',self.distance_range,'m')
        print('Scanning field of view is:',np.rad2deg(self.scan_fov),'deg') 
        print('*******************************************************************************************')
        
    def loc_to_rangeangle(self, p_eb, t_em):

        # generate a homogeneous transformation of the robot pose
        H_eb = HomogeneousTransformation(p_eb[0:2],p_eb[2])

        # get the location of the map feature in the body frame using Inverse(H_eb)=H_be, and then to from the body to the sensor frame using Inverse(H_bl)=H_lb
        t_lm = t2v(Inverse(self.H_bl.H)@Inverse(H_eb.H)@v2t(t_em))

        # Convert the cartesian vector t_lm in the sensor frame to the equivalent polar coordinates
        r,theta = cartesian2polar(t_lm[0],t_lm[1])

        z_lm = Vector(2)
        z_lm[0] = r
        z_lm[1] = theta   
        
        # check that the sensor readings provided are possible
        z_lm = self.check_range(z_lm)        

        return z_lm

    def rangeangle_to_loc(self, p_eb, z_lm):
        
        # generate a homogeneous transformation of the robot pose
        H_eb = HomogeneousTransformation(p_eb[0:2],p_eb[2])
        
        # check that the sensor readings provided are possible
        z_lm = self.check_range(z_lm)
        
        # convert the sensor observation in from polar to cartesian coordinates        
        r = z_lm[0]
        theta = z_lm[1]
        
        
        t_lm = polar2cartesian(r,theta)
        
        # Apply the homogeneous transformation to get from the sensor frame to the body with H_bl, and then to the fixed frame H_eb
        t_em = t2v(H_eb.H@self.H_bl.H@v2t(t_lm))
        
        return t_em
    
    def check_range(self,z_lm):
 
        # out of range
        if z_lm[0]>self.distance_range[1] or z_lm[0]<self.distance_range[0]: 
            z_lm[0] = np.nan
            z_lm[1] = np.nan           
        
        # out of scan fov
        if z_lm[1]>0.5*self.scan_fov or z_lm[1]<-0.5*self.scan_fov: 
            z_lm[1] = np.nan
            z_lm[0] = np.nan       
            
        return z_lm

            

class TrajectoryGenerate():
    """Class to generate trajectories from a path of x,y, coordinates defined in the earth fixed frame
    This path can be used as input for mobile robot control. 
    
    It key functions are to 
    - Assign timestamps, headings and linear and angular velocities to waypoints by implementing v-a couple trapezoidal trajectories between points
    - Generate turning arcs of a defined radius 
    - Sample the trajectory and associated model commands for any given timestamp
    - Manage waypoint progress considering waypoint acceptance radius and timeouts between consecutive waypoints

    Parameters:
    -----------
    x_path: List of Floats[]
        A sequence of x-axis (Northings) coordinates in the earth fixed frame in metres
    y_path: List of Floats[]
        A sequence of y-axis (Eastings) coordinates in the earth fixed frame in metres

    Attributes:
    -----------
    turning_radius: Float
        Radius in metres of turning arcs
    wp_id: Integer
        Pointer to log the next waypoint the robot trajectory should head towards
    t_complete: Float
        Log the time when the mission was completed
    
    P: Matrix(nx3) 
        Trajectory of poses connected by straightlines, with corresponding timestamps Tp. 
        Each row consists of x,y,gamma defined in the fixed frame
    Tp: Vector(n)    
        Timestamps in seconds, measured as elapsed mission time associated with the poses in the trajectory P
    V: Vector(n)        
        Linear velocities in the body-frame x-direction corresponding to each timestamp Tp
    W: Vector(n)            
        Angular velocities about the body-frame z-direction corresponding to each timestamp Tp    
    D: Vector(n)                
        The relative distance travelled between consecutive poses in P

    P_arc: Matrix(nx3) 
        Trajectory of poses connected by straightlines and turning arcs, with corresponding timestamps Tp. 
        Each row consists of x,y,gamma defined in the fixed frame
    Tp_arc: Vector(n)    
        Timestamps in seconds, measured as elapsed mission time associated with the poses in the trajectory P_arc
    V_arc: Vector(n)        
        Linear velocities in the body-frame x-direction corresponding to each timestamp Tp_arc
    W_arc: Vector(n)            
        Angular velocities about the body-frame z-direction corresponding to each timestamp Tp_arc    
    D_arc: Vector(n)                
        The relative distance travelled between consecutive poses in P_arc
    Arc: Matrix(nx2) 
        A path of turning arc centre points associated with the trajectory P_arc
        
    Functions:
    -----------
    path_to_trajectory(v, a):
        Takes the x_path and y_path parameters and generates trajectories by implementing v-a coupled trapezoidal 
        trajectories between point pairs.
        
        It assumes the first path entry starts from stationary, and that the last path entry ends at stationary. 
        It populates the class attributes:
        
        P, Tp, V, W, D 
        
        and uses the class internal functions _point_to_point() and _stack_trajectory() to do this.        

        Input:
            v:  Float
                Linear forwards velocity in m/s (body frame, b) to reach during coasting phases once accelerated and before decceleration
            a:  Float                
                Linear forwards acceleration in m/s2 (body frame, b) to apply during acceleration and decceleration phases
                
    turning_arcs(radius):
        A function that generates turning arcs of specified radius between staightline sections with different heading angles. 
        It populates the class attibutes:
        
        P_arc, Tp_arc, V_arc, W_arc, D_arc, Arc, turning_radius
        
        and uses geometric operations implemented in math_feeg6043 inward_unit_norm() and line_intersection() to do this.
        
        Input:
            radius:  Float
                Radius in metres of turning arcs
                
    p_u_sample(t):
        A function that samples P_arc, Tp_arc, V_arc, W_arc, D_arc, Arc, turning_radius at time t, where the outputs can 
        be used as inputs for control. 
        It uses geometric operations implemented in math_feeg6043 short_angle() amd interpolate() to do this
        
        Input:
            t: Float
                Timestamp in seconds to sample the trajectory at
        
        Output:
            p: Vector(3)
                Pose is a [3 x 1] matrix of the form
                  [[x]      Northings in m in the earth fixed frame, e
                   [y]      Eastings in m in the earth fixed frame, e
                   [gamma]] Heading in radians about the downwards axis in the earth fixed frame, e
                   
            u: Vector(2)            
                Body twist is a [2 x 1] matrix of the form
                  [[v]      Linear forwards velocity in m/s (body frame, b)
                   [w]]     Angular velocity in rads/s (body frame, b)
    
    wp_progress(t_robot, p_robot, accept_radius, timeout_factor = 2, initial_timeout = 30):
        A function checks whether the trajectory sampled by p_u_sample can progress beyond the next waypoint in P_arc. 
        It implements two conditions to allow this, the first is proximity, which progresses to the next waypoint 
        if the distance between the robot and the waypoint is less than the acceptance radius
        The second is a segment timeout, where waypoint will be progressed if the time takes to reach it is double 
        the expected time as default. This can be changed. The function logs any delay to expected arrival at each waypoint 
        to update future trajectory timestamps
        For the initial waypoint as timeout of 10s is set in the code. The function modifies the class attributes
        
        Tp_arc, wp_id
        
        Input:
            t_robot: Float 
                Timestamp in seconds corresponding to the robot pose
            p_robot: Vector(3)
                Pose of the robot is a [3 x 1] matrix of the form
                  [[x]      Northings in m in the earth fixed frame, e
                   [y]      Eastings in m in the earth fixed frame, e
                   [gamma]] Heading in radians about the downwards axis in the earth fixed frame, e            
            accept_radius: Float
                Acceptance radius in m for the next waypoint to be judged as passed
            timeout_factor: Float
                Default of 2 factor set and applied to the expected time to complete each segment of the trajectory. 
                If exceeded the waypoint will be bypassed and move on to the next
            initial_timeout: Float
                Default of 30, number of seconds to set the first wapyoint timeout to. 
                   
    _point_to_point(self, x_points,y_points, params, start_stationary = True, end_stationary = True):
        An internal function that is used by path_to_trajectory to compute the v-a trapezoidal constraints between a pair of consecutive waypoints
        
    _stack_trajectory(self,P,x_prev,y_prev,gamma,Tp,time,V, velocity,W, angular_velocity, D, distance):
        An internal function that concatenates the outputs of _point_to_point to compute the v-a trapezoidal constraints between a pair of consecutive waypoints

    """


    def __init__(self, x_path, y_path):

        self.x_path = x_path
        self.y_path = y_path        
        self.turning_radius = np.nan
        self.t_complete = np.nan
        
        self.wp_id = 0        
        
        # straight line trajectories
        self.P = np.nan*Matrix(1,3)    
        self.Tp = np.nan*Vector(1)
        self.V = np.nan*Vector(1)        
        self.W = np.nan*Vector(1)                
        self.D = np.nan*Vector(1)
        
        # straight line trajectories
        self.P_arc = np.nan*Matrix(1,3)    
        self.Tp_arc = np.nan*Vector(1)
        self.V_arc = np.nan*Vector(1)        
        self.W_arc = np.nan*Vector(1)                
        self.D_arc = np.nan*Vector(1)                
        self.Arc = np.nan*Matrix(1,2)

    def path_to_trajectory(self, v, a):
    
        for i in range(len(self.x_path)-1):

            # if conditions so only the first waypoint starts as a stationary one, and only the last way point ends stationary
            if i==0: start_stationary = True
            else: start_stationary = False

            if i==len(self.x_path)-2: end_stationary = True
            else: end_stationary = False

            # calculate trajectory between points and store in temporary containors. Remember the python range convention means [i-1:i+1] gives i-1 and i
            P_p2p, Tp_p2p, V_p2p, W_p2p, D_p2p = self._point_to_point(self.x_path[i:i+2],self.y_path[i:i+2], [v, a], start_stationary, end_stationary)

            # stack the trajectories
            if i == 0:
                # For the first trajectory, replace the containors as they are initialised with 0 values            
                self.P = P_p2p
                self.Tp = Tp_p2p
                self.V = V_p2p
                self.W = W_p2p                
                self.D = D_p2p
            else:
                # Note that the first new trajectory pose information should overwrite the last of the previous so that at the intermediate waypoint, the heading is correct
                self.P = np.vstack((self.P[:-1],P_p2p))
                self.V = np.vstack((self.V[:-1],V_p2p))        
                self.W = np.vstack((self.W[:-1],W_p2p))                        

                # Conversly the last distance should be kept and underwrite the first of the new point to point trajectory, since the first entry of this is alway 0
                self.D = np.vstack((self.D,D_p2p[1:]))      

                # Timestamps need to be cumulative. Since point to point timestamps are all start at t=0, we need to add the time so far (last element of T). We follow the convention for poses when we stack time (should not matter either way)
                Tp_p2p = Tp_p2p + self.Tp[-1]
                self.Tp = np.vstack((self.Tp[:-1],Tp_p2p))
                
    def turning_arcs(self,radius):
        # This function takes a trajectory defined by the poses P, corresponding timestamps Tp, and associated velocities V and segment distances D, and replaces corners with turning arcs of defined radius.
        # The function assumes that the provided trajectory has straight line sections between the poses in P
        # It outputs a new trajectory with turning arcs implemented, where segment velocities, distances and waypoint timestamps are updated to be consisent with the new trajectory    

        self.turning_radius = radius
        # store accumulated time to modify waypoints
        t_accumulate = 0
        delta_distance = 0

        # set a threshold angle below which to judge line segment angles to be the same either side of a waypoint
        min_angle = np.deg2rad(1)

        # temporary containor to store waypoint positions
        wp = Vector(3)
        arc_centre = Vector(2)


        for i in range(len(self.Tp)):

            # Calculate the angle between consecutive line segments to decide if a turning arc is needed using arctan2(dy,dx)                
            if i == len(self.Tp)-1 or i==0: delta_angle = 0
            else:
                angle = np.arctan2(self.P[i+1][1] - self.P[i][1], self.P[i+1][0] - self.P[i][0])    
                angle_prev = np.arctan2(self.P[i][1] - self.P[i-1][1], self.P[i][0] - self.P[i-1][0])    

                # handle negative values
                if(angle<0): angle += 2 * np.pi
                if(angle_prev<0): angle_prev += 2 * np.pi           

                delta_angle = abs(angle - angle_prev)

            # store corrected timestamp accounting for cumulative increase due to lengthening/shortening of trajectory
            t = self.Tp[i] + t_accumulate  

            # if the line angle is straight these are used in the new trajectory    
            if delta_angle < min_angle:
                wp[0] = copy.copy(self.P[i,0])
                wp[1] = copy.copy(self.P[i,1])            
                wp[2] = copy.copy(self.P[i,2])                        

                arc_centre[0] = np.nan
                arc_centre[1] = np.nan            

                # angular velocity for straight line sections
                w = 0

                # account for increased distance up to the point
                d = self.D[i]+delta_distance
                # reset delta_distance
                delta_distance = 0


                # initial matrices are 0 values, need to be overwritten
                if i == 0:
                    self.P_arc = copy.copy(wp.T)
                    self.Arc = copy.copy(arc_centre.T)
                    self.Tp_arc = copy.copy(t)
                    self.V_arc = copy.copy(self.V[i])
                    self.D_arc = copy.copy(d)
                    self.W_arc = np.array([w])

                else:
                    self.P_arc = np.vstack((self.P_arc, wp.T))                                
                    self.Arc = np.vstack((self.Arc, arc_centre.T))                
                    self.Tp_arc = np.vstack((self.Tp_arc, t))                                
                    self.V_arc = np.vstack((self.V_arc, self.V[i]))                                
                    self.D_arc = np.vstack((self.D_arc, d))
                    self.W_arc = np.vstack((self.W_arc, np.array([w])))  

            # calculate the coordinates of the rounded corners
            else:
                # reset the distance change
                delta_distance = 0

                #determine direction to offset lines by to get the internal intersect (i.e., centre of the turning arc)            
                x_list = [self.P[i-1][0], self.P[i][0], self.P[i+1][0]]
                y_list = [self.P[i-1][1], self.P[i][1], self.P[i+1][1]]

                # vector0 for incoming line0, and vector1 for outgoing line1
                vector0 = np.array([x_list[1]-x_list[0], y_list[1]-y_list[0]])
                vector1 = np.array([x_list[1]-x_list[2], y_list[1]-y_list[2]])


                # calculate normal vectors in the small angle direction between the lines
                u_norm0, u_norm1 = inward_unit_norm(vector0, vector1)        

                #offset line0 endpoints inwards by the turning radius
                line0_offset_wp0 = np.array([x_list[0], y_list[0]]) + radius * u_norm0
                line0_offset_wp1 = np.array([x_list[1], y_list[1]]) + radius * u_norm0


                #offset line1 endpoints inwards by the turning radius
                line1_offset_wp1 = np.array([x_list[1], y_list[1]]) + radius * u_norm1
                line1_offset_wp2 = np.array([x_list[2], y_list[2]]) + radius * u_norm1

                #define lines between the offset points
                line0_offset = np.array([line0_offset_wp0, line0_offset_wp1])
                line1_offset = np.array([line1_offset_wp1, line1_offset_wp2])

                # calculate intersect of the offset lines as the centre of the turning arc                        
                intersect = line_intersection(line0_offset,line1_offset) 
                arc_centre[0] = intersect[0]
                arc_centre[1] = intersect[1]

                # calculate where the arc's tangent intersects as the projection of the intersect by a radius offset outwards along the normals from earlier
                line0_arc_intersect = arc_centre.T[0] - radius * u_norm0
                line1_arc_intersect = arc_centre.T[0] - radius * u_norm1

                # calculate the distance of the arc from the vector dot product
                cos_theta = np.dot(u_norm0,u_norm1) 

                # Calculate the angle in radians using cos(theta) = 2 vector dot product and determine distance along the arc
    #             angle_radians = np.arccos(np.clip(cos_theta, -1.0, 1.0))
                arc_angle = np.arccos(np.dot(u_norm0,u_norm1))                
                arc_distance = arc_angle*radius

                # calculate the straight line distances from the arc intersect
                straight_distance0 = np.sqrt((line0_arc_intersect[0]-x_list[1])**2+(line0_arc_intersect[1]-y_list[1])**2)
                straight_distance1 = np.sqrt((line1_arc_intersect[0]-x_list[1])**2+(line1_arc_intersect[1]-y_list[1])**2)    

                max_allowable0 = np.sqrt((x_list[0]-x_list[1])**2+(y_list[0]-y_list[1])**2)
                max_allowable1 = np.sqrt((x_list[2]-x_list[1])**2+(y_list[2]-y_list[1])**2)                

                if straight_distance0 < max_allowable0 and straight_distance1 < max_allowable1:
                    # add first turning arc waypoint
                    wp[0] = line0_arc_intersect[0]
                    wp[1] = line0_arc_intersect[1] 
                    wp[2] = self.P[i-1][2] # on line0 so should be same angle as incoming line
                    self.P_arc = np.vstack((self.P_arc, wp.T))
                    self.V_arc = np.vstack((self.V_arc, self.V[i])) #velocity also needs to be maintained

                    # store arc centre that we need to go around            
                    self.Arc = np.vstack((self.Arc, arc_centre.T))   

                    # need to determine if it is a right turn or a left turn
                    H_eb = HomogeneousTransformation(wp[0:2],wp[2])
                    t_ec = v2t(arc_centre)
                    t_bc = Inverse(H_eb.H)@t_ec
                    if t_bc[1]>0: w = self.V[i]/radius #if centre is +ve y, turn clockwise at w=v/r                
                    else: w = -self.V[i]/radius # if centre is -ve y, turn anticlockwise                  
                    self.W_arc = np.vstack((self.W_arc, np.array([w])))            

                    # timestamp is the corner waypoint minus time to travel the distance from turning arc intersect    
                    t = self.Tp[i] - straight_distance0/self.V[i] + t_accumulate            
                    self.Tp_arc = np.vstack((self.Tp_arc, t))

                    # distance is cornver distance minus distance from turning arc intersect
                    d = self.D[i]-straight_distance0
                    self.D_arc = np.vstack((self.D_arc, d))                                                                   

                    # add second turning arc waypoint
                    wp[0] = line1_arc_intersect[0]
                    wp[1] = line1_arc_intersect[1] 
                    wp[2] = self.P[i][2] # on line1 so should be same angle as outgoing line
                    self.P_arc = np.vstack((self.P_arc, wp.T))
                    self.V_arc = np.vstack((self.V_arc, self.V[i])) #velocity also needs to be maintained

                    # add second turning arc waypoint
                    arc_centre[0] = np.nan
                    arc_centre[1] = np.nan            
                    self.Arc = np.vstack((self.Arc, arc_centre.T))  
                    w = 0
                    self.W_arc = np.vstack((self.W_arc, np.array([w])))            

                    # timestamp added to first arc + time to traverse the arc
                    t += arc_distance/self.V[i]
                    self.Tp_arc = np.vstack((self.Tp_arc, t))

                    # distance is cornver distance minus distance from turning arc intersect
                    d = arc_distance
                    self.D_arc = np.vstack((self.D_arc, d))

                    t_accumulate += (arc_distance-(straight_distance0+straight_distance1))/self.V[i]
                    delta_distance = -straight_distance1
                else:
                    wp[0] = copy.copy(self.P[i,0])
                    wp[1] = copy.copy(self.P[i,1])            
                    wp[2] = copy.copy(self.P[i,2])                        

                    arc_centre[0] = np.nan
                    arc_centre[1] = np.nan            

                    # angular velocity for straight line sections
                    w = 0

                    # account for increased distance up to the point
                    d = self.D[i]+delta_distance
                    # reset delta_distance
                    delta_distance = 0

                    self.P_arc = np.vstack((self.P_arc, wp.T))                                
                    self.Arc = np.vstack((self.Arc, arc_centre.T))                
                    self.Tp_arc = np.vstack((self.Tp_arc, t))                                
                    self.V_arc = np.vstack((self.V_arc, self.V[i]))                                
                    self.D_arc = np.vstack((self.D_arc, d))
                    self.W_arc = np.vstack((self.W_arc, np.array([w])))  
                    
                    print('Turning arcs cannot fit along line section. Increase angle between line section or reduce turning arc radius.')
    
    def p_u_sample(self, t):
        # function that uses interpotation and extrapolation to sample the desired trajectory 
        # defined by poses P, timestamps Tp, velocities V, angular velocities W,
        # turning arc centre's Arc with radius 'radius' at time t

        if len(self.Tp_arc) == 1:
            # means turning_arcs has not run so sample straight line trajectories
            P = self.P
            Tp = self.Tp
            V = self.V
            W = self.W
            D = self.D
            Arc = Matrix(len(self.Tp),2)*np.nan  
            turning_arc_flag = False
        else:
            P = self.P_arc
            Tp = self.Tp_arc
            V = self.V_arc
            W = self.W_arc
            D = self.D_arc
            Arc = self.Arc
            turning_arc_flag = True

        # initialise the pose to return
        p = Vector(3)
        u = Vector(2)        
        init_wp_timeout=30        

        # find the closest timestamp for gamma
        idx = (np.argmin(abs(Tp-t)))

        # determine nearest timestamps to sample
        # if Tp[idx][0] it is smaller than t, then the next timestamp must be after t
        if t>=Tp[idx][0]:        
            id_T0 = idx #id_Tg0 is always the lower id to use
            id_T1 = idx+1 #id_Tg1 is always the upper id to use            

        else:
            # if it is larger, then the previous timestamp must be before t   
            id_T0 = idx-1
            id_T1 = idx


        # check if trajectory can pass the next waypoint
        if id_T1 > self.wp_id:

            #cap pose and control commands to waypoint        
            p[0] = P[self.wp_id][0]
            p[1] = P[self.wp_id][1]
            p[2] = P[self.wp_id][2]

            u[0] = V[self.wp_id][0]
            u[1] = W[self.wp_id][0]              

        # if we are at the end of the data, in which case we sample the last
        else: 
            if t > Tp[-1]:
                #means final waypoint
                p[0] = P[-1][0]
                p[1] = P[-1][1]
                p[2] = P[-1][2]

                u[0] = V[-1][0]
                u[1] = 0 #straightline                    
            # else if we are at the end of the data, in which case we sample the last
            elif t < Tp[0]:
                #means final waypoint
                p[0] = P[0][0]
                p[1] = P[0][1]
                p[2] = P[0][2]

                u[0] = V[0][0]
                u[1] = 0 #straightline                    
            # anything intermediate use extrapolation
            else:        
                # samples times and angles as scalar values expected by interpolate
                t0 = Tp[id_T0][0] 
                t1 = Tp[id_T1][0]         

                # if there is no corresponding turning arc, it is a straight line
                if np.isnan(Arc[id_T0]).any():
                    p[0] = interpolate(t, t0, t1, P[id_T0][0], P[id_T1][0])
                    p[1] = interpolate(t, t0, t1, P[id_T0][1] , P[id_T1][1]) 
                    if turning_arc_flag == False: p[2] = P[id_T0][2] #needs to maintain heading until it arrives at the next waypoint for linear 
                    else: p[2] = P[id_T1][2] #needs to adopt the next headingp[2] = interpolate(t, t0, t1, P[id_T0][2] , P[id_T1][2] , wrap = np.deg2rad(360)) 


                    u[0] = interpolate(t, t0, t1, V[id_T0][0] , V[id_T1][0])
                    u[1] = 0 #straightline

                else:
                    # calculate arc angles respect to arc centre
                    dx = P[id_T0][1]-Arc[id_T0][1]
                    dy = P[id_T0][0]-Arc[id_T0][0]
                    gamma_arc0 = np.arctan2(dx,dy)# measure from dx (north)

                    dx = P[id_T1][1]-Arc[id_T0][1]
                    dy = P[id_T1][0]-Arc[id_T0][0]
                    gamma_arc1 = np.arctan2(dx,dy)# measure from dx (north)            

                    # interpolate the current arc angle
                    arc_angle = interpolate(t, t0, t1, gamma_arc0, gamma_arc1, wrap = np.deg2rad(360))

                    # calculate position for the current arc angle
                    p[0] = Arc[id_T0][0] + self.turning_radius * np.cos(arc_angle)
                    p[1] = Arc[id_T0][1] + self.turning_radius * np.sin(arc_angle)                              
                    p[2] = interpolate(t, t0, t1, P[id_T0][2] , P[id_T1][2], wrap = np.deg2rad(360))            

                    # calculate linear and angular velocity command
                    u[0] = interpolate(t, t0, t1, V[id_T0][0] , V[id_T1][0])
                    u[1] = W[id_T0][0] # angular velocity is constant along the turning arc

        return p, u
    
    def wp_progress(self, t_robot, p_robot, accept_radius, timeout_factor = 2, initial_timeout = 30):

        # Containor for updated timestamps 
        if len(self.Tp_arc) == 1:
            # means turning_arcs has not run so sample straight line trajectories
            P = self.P
            Tp = self.Tp
            V = self.V
            W = self.W
            D = self.D
            Arc = Matrix(len(self.Tp),2)*np.nan        
        else:
            P = self.P_arc
            Tp = self.Tp_arc
            V = self.V_arc
            W = self.W_arc
            D = self.D_arc
            Arc = self.Arc

        #set an initial schedule time to get to the first waypoint
        scheduled_time = 0 #s
        
        wp_progress_flag = False # flag to progress to next waypoint

        # check distance to the next waypoint
        distance_to_wp = np.sqrt((P[self.wp_id,0]-p_robot[0])**2+(P[self.wp_id,1]-p_robot[1])**2)

        # calculate the time spent at on the current waypoint segment
        if self.wp_id ==0: segment_time = t_robot
        else: segment_time = t_robot - Tp[self.wp_id-1]

        # set a segment delay timeout as double the scheduled time
        scheduled_time = Tp[self.wp_id]
        if self.wp_id == 0: 
            timeout = initial_timeout
            latest_allowable = timeout
        else: 
            timeout = timeout_factor*(scheduled_time-Tp[self.wp_id-1])
            latest_allowable = timeout+Tp[self.wp_id-1]

        # compute delay
        if t_robot > scheduled_time: delay = t_robot - scheduled_time
        else: delay = 0  



        if self.wp_id == len(Tp)-1: 
            if np.isnan(self.t_complete): 
                self.t_complete = t_robot
                
                print('************************************************************')
                print('Trajectory completed at:',self.t_complete,'s')    
                print('************************************************************')                
        else:
            # if within acceptance radius, follow trajectory to next waypoint
            if distance_to_wp <= accept_radius:                
           
                print('************************************************************')
                print('Reached waypoint ',self.wp_id,' at t=',t_robot,'s') 
                print('Delay',delay,'s, Scheduled arrival', scheduled_time,'s, Latest allowable arrival', latest_allowable,'s')        
                print('************************************************************')
                wp_progress_flag = True

            elif t_robot > latest_allowable:            
                print('************************************************************')
                print('Failed to reach waypoint ',self.wp_id) 
                print('Delay',delay,'s, Scheduled arrival', scheduled_time,'s')
                print('Time ',t_robot,' s exceeds Latest allowable arrival', latest_allowable,'s')
                print('************************************************************')
                wp_progress_flag = True

            if wp_progress_flag == True:

                if self.wp_id <= len(Tp)-2: 
                    self.wp_id += 1             
                    print('Go to next waypoint ',self.wp_id,' at ',P[self.wp_id])

                    # add any delay to all future waypoints
                    if delay > 0:
                        for i in range(self.wp_id-1,len(Tp)): Tp[i]=Tp[i]+delay
                        if len(self.Tp_arc) == 1: self.Tp = copy.copy(Tp)
                        else: self.Tp_arc = copy.copy(Tp)                 
         
                
    def _point_to_point(self, x_points,y_points, params, start_stationary = True, end_stationary = True):

        # computes the trapezoidal trajectory between two points (x_points, y_points) subject to 
        # velocity and acceleration constraints (params = [v, a])
        # returns the trajectory of poses P with start-relative timestamps Tp, velocities V and distances of each segment

        #initialise
        t=0 #time 
        d=0 #distance

        # Trajectory matrices: pose, timestamps, velocities and distances to store
        P = Matrix(1,3)    
        Tp = Vector(1)
        V = Vector(1)
        W = Vector(1)
        D = Vector(1)
        

        # read parameters
        v = params[0]
        a = params[1]

        # check they are OK
        if v==0: print('Must have v > 0') 
        if a==0: print('Must have a > 0')                

        # determine straight line angle between points
        gamma=np.arctan2(y_points[1]-y_points[0],x_points[1]-x_points[0]) 
        if gamma<0: gamma += 2*np.pi # wraps angles to be between 0 and 2pi 

        # determine straight line distance between points
        total_distance = np.sqrt((y_points[1]-y_points[0])**2+(x_points[1]-x_points[0])**2)

        if start_stationary == True:
            # start of accelerate phase corresponds to first element of Trajectory matrices
            # store starting pose (note velocity, time and distance are initially zero)
            P[0]=np.array([x_points[0],y_points[0],gamma]).T

            # end of accelerate phase corresponds to next element of Trajectory matrices
            t += v/a
            d = 0.5*a*(v/a)**2 

            # store next trajectory row

            P, Tp, V, W, D = self._stack_trajectory(P,x_points[0],y_points[0],gamma,Tp,t,V,v,W,0,D,d)

            if end_stationary == True: 
                # end of coast phase corresponds to next element of Trajectory matrices                      
                t += (total_distance - 2*(0.5*a*(v/a)**2))/v #total time - time to accelerate and decelerate
                d += total_distance - 2*(0.5*a*(v/a)**2)

                # store next trajectory row            
                P, Tp, V, W, D = self._stack_trajectory(P,x_points[0],y_points[0],gamma,Tp,t,V,v,W,0,D,d)

                # end of deccelerate phase corresponds to next element of Trajectory matrices                      
                t += v/a
                d += 0.5*a*(v/a)**2

                # store next trajectory row, note velocity needs to be 0            
                P, Tp, V, W, D = self._stack_trajectory(P,x_points[0],y_points[0],gamma,Tp,t,V,0,W,0,D,d)

            else:
                # coast to the end
                dd = total_distance - (0.5*a*(v/a)**2)
                d += dd
                t += dd/v            

                # store next trajectory row            
                P, Tp, V, W, D = self._stack_trajectory(P,x_points[0],y_points[0],gamma,Tp,t,V,v,W,0,D,d)

        else: #start already coasting

            # store starting pose and velocity (time, distance are initially zero)
            P[0] = np.array([x_points[0],y_points[0],gamma]).T
            V[0] = v  
            W[0] = 0

            if end_stationary == True:
                # end of coast phase corresponds to next element of Trajectory matrices
                dd = total_distance - (0.5*a*(v/a)**2)
                d += dd
                t += dd/v

                # store next trajectory row            
                P, Tp, V, W, D = self._stack_trajectory(P,x_points[0],y_points[0],gamma,Tp,t,V,v,W,0,D,d)

                # end of deccelerate phase corresponds to next element of Trajectory matrices                                                              
                t += v/a
                d += 0.5*a*(v/a)**2

                # store next trajectory row, note velocity needs to be 0            
                P, Tp, V, W, D = self._stack_trajectory(P,x_points[0],y_points[0],gamma,Tp,t,V,0,W,0,D,d)

            else:
                # coast to end
                d += total_distance
                t += d/v

                # store next trajectory row            
                P, Tp, V, W, D = self._stack_trajectory(P,x_points[0],y_points[0],gamma,Tp,t,V,v,W,0,D,d)

        return P, Tp, V, W, D
    
    def _stack_trajectory(self,P,x_prev,y_prev,gamma,Tp,time,V, velocity,W, angular_velocity, D, distance):

        # calculate new position after traveling 'distance' along 'gamma'
        x = x_prev + distance*np.cos(gamma)
        y = y_prev +distance*np.sin(gamma)

        # calculate the distance increment
        distance_increase = distance - sum(D[:,0])
        Tp = np.vstack((Tp,time))         
        D = np.vstack((D,distance_increase)) #we store the distance increment, so subtract the previous distance 
        V = np.vstack((V,velocity)) 
        W = np.vstack((W,angular_velocity))         
        P=np.vstack((P,np.array([x,y,gamma]).T))        
        
        return P, Tp, V, W, D     

def feedback_control(ds, ks = None, kn = None, kg = None):
    
    if ks == None: ks = 0.1
    if kn == None: kn = 0.1
    if kg == None: kg = 0.1        
    
    dv = ks*ds[0]                
    dw = (kn*ds[1]+kg*ds[2])
    
    du = Vector(2)
    du[0] = dv
    du[1] = dw
    
    return du

def kalman_filter_predict(mu, Sigma, u, A, B, R, view_flag=None, x=None, ylim=None):
    """
    Keyword arguments:    
    view_flag -- A boolean to show intermediate plots of the prediction and measurement update
    optional argument -- a range of states to plot probabilities over, using numpy arrays e.g. np.arange(-8,8,0.05)    
    """
    # (1) Project the state forward: x = Ax + Bu
    pred_mu = A @ mu+ B @ u

    # (2) Project the error forward: 
    pred_Sigma = A @ Sigma @ A.T + R
    
    # Return the predicted state and the covariance
    if view_flag is True:
        if x is None: x = np.arange(-8,8,0.05) 
        if ylim is None: ylim=[0,0.55] #default, otherwise use what is provided
    
        plot_kalman(mu, Sigma, pred_mu, pred_Sigma, x, ylim)

    return pred_mu, pred_Sigma

def kalman_filter_update(mu, Sigma, z, C, Q, view_flag=None, x=None, ylim=None):
    """
    Keyword arguments:    
    view_flag -- A boolean to show intermediate plots of the prediction and measurement update
    optional argument -- a range of states to plot probabilities over, using numpy arrays e.g. np.arange(-8,8,0.05)    
    """

    # (3) Compute the Kalman gain
    K = Sigma @ C.T @ np.linalg.inv(C @ Sigma @ C.T + Q)        

    # (4) Compute the updated state estimate
    update_mu = mu + K @ (z - C @ mu)

    # (5) Compute the updated state covariance
    update_Sigma = (np.eye(mu.shape[0], dtype=float) - K @ C) @ Sigma
    
    if view_flag is True:
        if x is None: x = np.arange(-8,8,0.05) 
        if ylim is None: ylim=[0,0.55] #default, otherwise use what is provided

        print(C[0,0])
        plot_kalman(mu, Sigma, update_mu, update_Sigma, x, ylim,z, C[0,0], Q)

    # Return the correct state and the covariance        
    return update_mu, update_Sigma

def extended_kalman_filter_predict(mu, Sigma, u, f, R, dt, view_flag=False, x=None, xlim=None, ylim=None):
    """
    Keyword arguments:    
    view_flag -- A boolean to show intermediate plots of the prediction and measurement update
    optional argument -- a range of states to plot probabilities over, using numpy arrays e.g. np.arange(-8,8,0.05)    
    """    
    # (1) Project the state forward
    pred_mu, F = f(mu, u, dt)
      
    # (2) Project the error forward: 

    pred_Sigma = (F @ copy.copy(Sigma) @ F.T) + R
    

    
    # Return the predicted state and the covariance

    if view_flag is True:
        if x is None: x = np.arange(mu-10,mu+10,0.05) 
        if ylim is None: ylim=[0,0.6] #default, otherwise use what is provided    
        if xlim is None: xlim=[-5,5] #default, otherwise use what is provided            
        plot_kalman(mu, Sigma, pred_mu, pred_Sigma, x, xlim, ylim)

    # Return the state and the covariance
    return pred_mu, pred_Sigma

def extended_kalman_filter_update(mu, Sigma, z, h, Q, view_flag=False, x=None, xlim=None, ylim =None, wrap_index = None):
    """Keyword arguments:    
    view_flag -- A boolean to show intermediate plots of the prediction and measurement update
    optional argument -- a range of states to plot probabilities over, using numpy arrays e.g. np.arange(-8,8,0.05)    
    """
    
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
    
    if view_flag is True:
        if x is None: x = np.arange(mu-10,mu+10,0.05) 
        if xlim is None: xlim=[-5,5] #default, otherwise use what is provided
        if ylim is None: ylim=[0,0.6] #default, otherwise use what is provided
            
        plot_kalman(mu, Sigma, cor_mu, cor_Sigma, x, xlim, ylim, z, H[0,0], Q)    

    # Return the state and the covariance
    return cor_mu, cor_Sigma


class Particles:
    def __init__(self, N):
        self.N = N
        self.northings = np.array([None] * N)  # State - to be estimated
        self.eastings = np.array([None] * N)   # State - to be estimated
        self.gamma = np.array([None] * N)      # State provided with noise
        self.x_dot = np.array([None] * N)       # State provided with noise
        self.gamma_dot = np.array([None] * N)      # State provided with noise
        self.weight = np.ones(N) / float(N)    # Particle weights     
        
    def __str__(self):
        return "northings: " + str(self.northings) + "\neastings: " + str(self.eastings) + "\nweights: " + str(self.weight)


def systematic_resample(weights, demo=False):
    N = len(weights)
    random_number = np.random.uniform()
    positions = (np.arange(N) + random_number) / N
    indexes = np.zeros(N, "i")

    # Normalize the weights, so that the sum of the weights is 1
    ws = np.sum(weights)
    for i in range(N):
        weights[i] /= ws
    cumulative_sum = np.cumsum(weights)
    cumulative_sum[-1] = 1.0  # avoid round-off error

    i, j = 0, 0
    while i < N:
        if positions[i] < cumulative_sum[j]:
            indexes[i] = j
            i += 1
        else:
            j += 1
    if not demo:
        return indexes
    if demo:
        return indexes, random_number
    

def kde_probability(particles, sigma_resolution=1.0, sampling_resolution=None):
    
    # fit the particles to the KDE model
    locations = np.vstack([particles.northings, particles.eastings])
    
    if sampling_resolution == None:
        # returns the probability of each particle location
        kde = KernelDensity(kernel='gaussian', bandwidth=sigma_resolution).fit(locations.T)
        log_density = kde.score_samples(locations.T)
        probability = np.exp(log_density)
        
        return probability   
    
    else:
        # returns the most likely location at a given sample resolution
        kde = KernelDensity(kernel='gaussian', bandwidth=sigma_resolution).fit(locations.T, sample_weight=particles.weight.T)

        state_range_northings = np.linspace(min(particles.northings),max(particles.northings), num=sampling_resolution)
        state_range_eastings = np.linspace(min(particles.eastings),max(particles.eastings), num=sampling_resolution)

        state_range = np.vstack([state_range_northings, state_range_eastings])
        density = kde.score_samples(state_range.T)       
        est_northings, est_eastings = state_range.T[density.argmax()]
        
        return est_northings, est_eastings   
    
def initialise_particle_distribution(particles, centre=[0, 0], radius = 1, heading = 0):

    # sample angles and ranges
    theta = np.random.uniform(0, 360, particles.N)
    r = np.sqrt(np.random.uniform(0, 1, particles.N)) * radius    

    # convert to cartesian    
    northings, eastings = polar2cartesian(r,np.deg2rad(theta))
    
    # add centre offset
    particles.northings  = northings + centre[0]
    particles.eastings = eastings + centre[1]
    
    # particles could be pointing anywhere
    particles.gamma = np.random.uniform(heading - np.deg2rad(10) , heading + np.deg2rad(10), particles.N)
    
def pf_measurement_probability(particles, measurement):    
    # calculate likelihood of each particle given the observation 
    delta = np.sqrt(measurement.northings_std*measurement.eastings_std)
    probability = []
    for i in range(particles.N):
        particle_to_measurement=(particles.northings[i]-measurement.northings)**2+(particles.eastings[i]-measurement.eastings)**2
        num = np.exp(-(particle_to_measurement)/(2*delta**2))
        den = np.sqrt(2*np.pi*delta**2)
        probability.append(num/den)   
        
    return probability

class Measurement:
    def __init__(self):
        self.timestamp = np.array([None])
        self.northings = np.array([None])
        self.eastings = np.array([None])
        self.northings_std = np.array([None])
        self.eastings_std = np.array([None])   


def kde_probability(particles, sigma_resolution=1.0, sampling_resolution=None):
    
    # fit the particles to the KDE model
    locations = np.vstack([particles.northings, particles.eastings])
    
    if sampling_resolution == None:
        # returns the probability of each particle location
        kde = KernelDensity(kernel='gaussian', bandwidth=sigma_resolution).fit(locations.T)
        log_density = kde.score_samples(locations.T)
        probability = np.exp(log_density)
        
        return probability   
    
    else:
        # returns the most likely location at a given sample resolution
        kde = KernelDensity(kernel='gaussian', bandwidth=sigma_resolution).fit(locations.T, sample_weight=particles.weight.T)

        state_range_northings = np.linspace(min(particles.northings),max(particles.northings), num=sampling_resolution)
        state_range_eastings = np.linspace(min(particles.eastings),max(particles.eastings), num=sampling_resolution)

        state_range = np.vstack([state_range_northings, state_range_eastings])
        density = kde.score_samples(state_range.T)       
        est_northings, est_eastings = state_range.T[density.argmax()]
        
        return est_northings, est_eastings    

def pf_normalise_weights(weights):
    ws = sum(weights)
    for i in range(len(weights)):
        weights[i] /= ws    
    return weights

def pf_update(particles, measurement):
    tau = pf_measurement_probability(particles, measurement) 
    
    sigma_resolution = np.sqrt(measurement.northings_std*measurement.eastings_std)
    prior_tau = kde_probability(particles, sigma_resolution)

    particles.weight *= tau /prior_tau
    pf_normalise_weights(particles.weight)

    return particles

def neff(particles): 
    particles.weight /= np.sum(particles.weight)
    return 1.0 / (particles.N * np.sum(np.square(particles.weight)))

def pf_resample(particles, jitter, verbose = False):
    """Resample particles using systematic resample"""
    
    if verbose == True: print('effective particles:', neff(particles))
    if neff(particles) < 0.5:
        # Get the indexes of the particles to resample
        indexes = systematic_resample(particles.weight)
        if verbose == True: print('Resampling needed, sampled particles:',indexes)
              
        # Copy all the particles, to overwrite "particles"
        particles_copy = copy.deepcopy(particles)
        
        angle = np.random.uniform(0,2*np.pi, particles.N)
        radius = np.random.normal(0,jitter, particles.N) 
        
        # Overwrite "particles" with the copied ones and correct indices
        for i in range(particles.N):
            particles.northings[i] = copy.deepcopy(particles_copy.northings[indexes[i]])+radius[i] * np.cos(angle[i])
            particles.eastings[i] = copy.deepcopy(particles_copy.eastings[indexes[i]])+radius[i] * np.sin(angle[i])
        
        # Reset the weights to equally probable
        particles.weight = np.ones(particles.N) / float(particles.N)
    else: 
        if verbose == True: print('No resampling needed')


def discrete_motion_model(particles, gamma, u, dt, process_noise):
    
        g_std = np.sqrt(process_noise[0])
        x_dot_std = np.sqrt(process_noise[1])
        g_dot_std = np.sqrt(process_noise[2])
        
        for i in range(particles.N):

            p = Vector(3)
            
            # noise is not added to the location of the particles as the distribution already represents the noise and this is what our PF will update based on sensor observations
            p[0] = particles.northings[i]
            p[1] = particles.eastings[i]

            # We treat the other states as auxilliary (Rao-Blackwellisation), where noise is random sampled from a distribution and added   
            if gamma != None: p[2] = gamma + np.random.normal(scale=g_std) % (2*np.pi)               
            else: p[2] = particles.gamma[i] + np.random.normal(scale=g_std) % (2*np.pi)  
            
            u_noise = Vector(2)
            u_noise[0] = u[0] + np.random.normal(scale=x_dot_std)
            u_noise[1] = u[1] + np.random.normal(scale=g_dot_std)                        
            
            # note rigid_body_kinematics already handles the exception dynamics of w=0
            p = rigid_body_kinematics(p,u_noise,dt)    
    
            # update the particles and store the information
            particles.northings[i] = p[0]
            particles.eastings[i] = p[1]
            particles.gamma[i] = p[2]            
            particles.x_dot[i] = u_noise[0]
            particles.gamma_dot[i] = u_noise[1]                        