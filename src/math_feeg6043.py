import numpy as np

# Functions to streamline numpy calls to define:

# Zero filled Vectors and Matrices of different dimensions and a square identity matrix
def Vector(n_rows): return np.zeros((n_rows,1), dtype=float)
def Matrix(n_rows, n_cols): return np.zeros((n_rows, n_cols), dtype=float)
def Identity(dim): return np.eye(dim, dtype=float)

# Transpose, Inverse matrix operations
def Transpose(m): return m.T
def Inverse(m): return np.linalg.inv(m)

# v2t adds a dimension of 1 on the bottom of a vector and t2v removes the last element of a vector
def v2t(x): return np.insert(x, len(x), 1, axis=0)
def t2v(x): return x[0:len(x)-1]

class HomogeneousTransformation:
    """Class to handle homogeneous transformations
    H = [R t; 0 1] where R is a 2x2 rotation matrix and t is a 2x1 translation vector
    H = (t, \gamma) where t is a 2x1 translation vector and \gamma is the rotation angle in radians
    Parameters:
    -----------
    t: Vector(2)
        translation vector
    gamma: float
        Rotation angle in radians
    Attributes:
    -----------
    R: Matrix(2,2)
        Rotation matrix
    H: Matrix(3,3)
        Homogeneous transformation matrix
    H_R: Matrix(3,3)
        Homogeneous transformation matrix with zero translation
    H_T: Matrix(3,3)
        Homogeneous transformation matrix with zero rotation
    t: Vector(2)
        Translation vector
    gamma: float
        Rotation angle in radians
    """

    def __init__(self, t=Vector(2), gamma=0):
        # Note that the class implementation ONLY stores the translation vector and
        # rotation angle. The rotation matrix and homogeneous transformation matrix are
        # computed on the fly.
        self._t = t
        self._gamma = gamma

    def _check_homogeneous(self, H):
        if H.shape != (3, 3):
            raise ValueError("H must be a 3x3 matrix")
        if H[2, 2] != 1.0:
            raise ValueError("H must be a homogeneous matrix")

    def _check_rotation(self, R):
        if R.shape != (2, 2):
            raise ValueError("R must be a 2x2 matrix")
        if not np.allclose(np.linalg.det(R), 1.0, rtol=1e-2):
            raise ValueError(
                "R must be a rotation matrix. Determinant is not 1.0, but",
                np.linalg.det(R),
                "difference is",
                np.linalg.det(R) - 1.0,
            )

    @property  # automatically populates elements if the class is called
    def gamma(self):
        return self._gamma

    @gamma.setter
    def gamma(self, gamma):
        self._gamma = gamma

    @property  # automatically populates elements if the class is called
    def R(self):
        R = Matrix(2, 2)
        c = np.cos(self.gamma)
        s = np.sin(self.gamma)
        R[0, 0] = c
        R[0, 1] = -s
        R[1, 0] = s
        R[1, 1] = c
        return R

    @R.setter
    def R(self, R):
        self._check_rotation(R)
        gamma_atan1 = np.arctan2(R[1, 0], R[0, 0])
        gamma_atan2 = np.arctan2(-R[0, 1], R[1, 1])
        if np.allclose(gamma_atan1, gamma_atan2):
            self.gamma = gamma_atan1
        else:
            raise ValueError(
                "R must be a rotation matrix. gamma is not the same for both arctan2"
            )

    @property  # automatically populates elements if the class is called
    def H(self):
        H = Identity(3)
        H[:2, :2] = self.R
        H[:2, 2:3] = self.t
        return H

    @H.setter
    def H(self, H):
        self._check_homogeneous(H)
        self.t = H[:2, 2:3]
        self.R = H[:2, :2]

    @property  # automatically populates elements if the class is called
    def H_R(self):
        H_R = Identity(3)
        H_R[:2, :2] = self.R
        return H_R

    @H_R.setter
    def H_R(self, H_R):
        self._check_homogeneous(H_R)
        self.R = H_R[:2, :2]

    @property  # automatically populates elements if the class is called
    def H_T(self):
        H_T = Identity(3)
        H_T[:2, 2:3] = self.t
        return H_T

    @H_T.setter
    def H_T(self, H_T):
        self._check_homogeneous(H_T)
        self.t = H_T[:2, 2:3]

    @property  # automatically populates elements if the class is called
    def t(self):
        return self._t

    @t.setter
    def t(self, t):
        self._t = t      
        

def cartesian2polar(x, y):
    'Converts 2D cartesian (x, y offsets) to return equivalent polar coordinates (range, angle)   '    
    
    # Calculate radial distance
    r = np.sqrt(x**2 + y**2)    
    # Calculate angular coordinate
    theta = np.arctan2(y, x)  % (2*np.pi)
    
    return r, theta

def polar2cartesian(r, theta):
    'Converts 2D polar coordinates (range, angle) to return equivalent catesian (x, y offsets) '
    # Calculate Cartesian coordinates
    x = r * np.cos(theta)
    y = r * np.sin(theta)
        
    return x, y

def short_angle(a0, a1, dt,wrap):    
    # this function handles angle wrapping between 0 and the specified value
    # calculate the difference between angle a1 and a0, wrapping the result to give the small angle
    
    da = (a1 - a0) % wrap
    
    # scale the difference the query timestamp via dt 
    # where dt is the ratio (t_query-t0)/(t1-t0)
    return a0 + (2*da % wrap - da)*dt


def interpolate(t_query, t_lower, t_upper, x_lower, x_upper, wrap = None):
    # This function linearly interpolates or extrapolates a parameter (x) to some query timestamp
    # for periodic values like angles, the wrap value can be specified to address numerical discontinuity 
    # and keep output between [0:wrap]
    
    
    # handles the case where the input times to interpolate or extrapolate are the same (zero order)
    if t_upper == t_lower:
        x_query = x_lower
        
    else:

        if wrap == None:
        # id no periodic wrap value is specified, implements normal linear interpolation             
            x_query = (x_upper-x_lower)/(t_upper-t_lower)*(t_query-t_lower)+x_lower
        else: 
        # if a wrap value is specified, implements normal linear interpolation
            dt = (t_query-t_lower)/(t_upper-t_lower)
            x_query = short_angle(x_lower, x_upper, dt ,wrap) % wrap

    return x_query

def inward_unit_norm(vector0,vector1):
    # This function determines the unit normal vectors of two lines in the direction that points towards the other line.
    # It takes two lines as input, where each line has the form:
    # line = np.array([dx, dy])
    # The unit norms of each line are returned in the same form as the input    
    
    #determine unit vectors by dividing by the magnitude
    distance0 = np.sqrt((vector0[0])**2+(vector0[1])**2)    
    distance1 = np.sqrt((vector1[0])**2+(vector1[1])**2)        

    unit_vector0 = vector0/distance0
    unit_vector1 = vector1/distance1       

    #determine the internal direction
    sign = (np.cross(vector0,vector1)) 

    # determine the offsets
    if sign >= 0:
            # if +ve line0 needs to be rotated counterclockwise to line1 for the small angle
            # so normal points up and left with the x-shift and y-shift swapped
        unit_norm0 = np.array([unit_vector0[1], -unit_vector0[0]])

            # if +ve line1 needs to be rotated clockwise to line0 for the small angle
            # so normal points down and right with the x-shift and y-shift swapped
        unit_norm1 = np.array([-unit_vector1[1], unit_vector1[0]])            

    if sign < 0:            
            # if -ve line0 needs to be rotated clockwise to line1 for the small angle
            # so normal points down and right with the x-shift and y-shift swapped
        unit_norm0 = np.array([-unit_vector0[1], unit_vector0[0]])   

            # if +ve line1 needs to be rotated counterclockwise to line0 for the small angle
            # so normal points up and left with the x-shift and y-shift swapped                  
        unit_norm1 = np.array([unit_vector1[1], - unit_vector1[0]])   

    return unit_norm0, unit_norm1

def line_intersection(line0, line1):
    # returns the intersection of two lines, where the lines are defined as
    # line = np.array([x0,y0],[x1,y1])
    # returns the x,y coordinate of the intersecting point

    xdiff = (line0[0][0] - line0[1][0], line1[0][0] - line1[1][0])
    ydiff = (line0[0][1] - line0[1][1], line1[0][1] - line1[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
        raise Exception('lines do not intersect')

    d = (det(*line0), det(*line1))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return x, y   

def gaussian(mu, sigma, x):  return 1/(np.sqrt(2*np.pi)*sigma)*np.exp(-(x-mu)**2/(2*(sigma**2)))

def eigsorted(cov):
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    return vals[order], vecs[:, order]

def l2m(nlist):
    # converts a list into a vector, or a list or lists into a matrix
    # this if checks if we have a list (it returns False) or a list of lists (returns True)
    if all(isinstance(item, list) for item in nlist) == False:  
        N = Vector(len(nlist[:]))
        for i in range(len(nlist[:])):
            N[i] = nlist[i]
    else: 
        N = Matrix(len(nlist[0]),len(nlist[:]))
        
        for i in range(len(nlist[:])):
            for j in range(len(nlist[i])):
                N[j,i] = nlist[i][j]                            
    return N

def fill_timegaps(dt_max,T):
    add_timestamp = []
    t_prev=T[0]
    
    #everytime the timegap is big, halve it
    for i in range(len(T)):
        if T[i]-t_prev >= dt_max:
            add_timestamp.append(0.5*(T[i]-t_prev)+t_prev)
        t_prev = T[i]
        
    if add_timestamp != []:    
        #add those new timestamps and put in chronological order
        T = np.concatenate((T, add_timestamp))
        sorted_indices = np.argsort(T[:,0])
        T=T[sorted_indices]    
    return T

def id_interpolate_pair(t,T):
    # finds the location of a pair of timestamps in T that should be used for interpolation to t
    # find the closest timestamp for gamma
    idx = (np.argmin(abs(T-t)))

    # if the closest timestamp is <t, the next timestamp must be the closest after t
    if T[idx]<=t:
        id0 = idx 
        if id0 == len(T)-1: id0 = idx -1 # unless we are at the end of the data, in which case we use our previous pair and forward extrapolate        
            
    # if the closest timestamp is >t, the previous timestamp must be the closest betfore t
    else:
        id0 = idx-1            
            
    return id0, id0+1    

def wrapped_mean(angles): 
    return np.arctan2(np.sum(np.sin(angles)), np.sum(np.cos(angles)))

# calculates the std of a list of angles accounting for angle wrapping
def wrapped_std(angles):     
    a = []
    for x in angles:
        a.append(np.arctan2(np.sin(x), np.cos(x)))
    return np.std(a) 

# Generate a complex observation function to simulate a sensor probability distribution
def complex_observation_function(mu,sigma,k,x):
    pd_observe=np.zeros(len(x))
    for i in range(len(mu)):  pd_observe+=k[i]*gaussian(mu[i], sigma[i], x)
    return pd_observe/sum(pd_observe)

# Define the kernel function
def rbf_kernel(X1, X2, l=1.0, sigma_f=1.0):
    sqdist = np.sum(X1**2, 1).reshape(-1, 1) + np.sum(X2**2, 1) - 2 * np.dot(X1, X2.T)
    return sigma_f**2 * np.exp(-0.5 / l**2 * sqdist)
