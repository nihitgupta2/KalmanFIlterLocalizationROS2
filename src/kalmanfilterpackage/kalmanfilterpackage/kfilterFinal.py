import rclpy
from rclpy.qos import ReliabilityPolicy, QoSProfile 
from rclpy.action import ActionServer
from rclpy.node import Node
from geometry_msgs.msg import Twist
import math
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu,JointState
from tf2_msgs.msg import TFMessage
import numpy as np
import matplotlib.pyplot as plt


class KalmanFilterNode(Node):

    def __init__(self):
        super().__init__('kalman_filter_node')
        self.dt = 0.2  # seconds
        print("Started")
        self.A = np.matrix([[1, 0, self.dt, 0, 0], [0, 1, 0,self.dt,0], [0 , 0, 1, 0, 0],[0,0,0,1,0],[0,0,0,0,1]]) # A matrix from equation of motion model
        self.B = np.matrix([[1 / 2 * self.dt ** 2, 0,0],[0,1 / 2 * self.dt ** 2,0],[self.dt, 0, 0],[0,self.dt,0],[0 , 0,self.dt]]) # this is the "input" multiplied matrix
        self.Q = np.identity(5)*0.05 # matrix Q from equations
        self.R = np.identity(5)*0.05 # matrix R from equations
        self.u = np.matrix([[0], [0], [0]]) # input control matrix u from equations
        self.P = np.identity(5) # uncertainty of an estimate-covariance matrix 
        self.C = np.identity(5) # observation matrix from equations
        self.x_hat = np.matrix([[0],[0],[0],[0],[0]])
        self.xhatList = []
        self.yList = []
        self.y = np.matrix([[0],[0],[0],[0],[0]])
        self.uList = []
        self.timespanList = []
        self.linA = np.matrix([[0], [0]])
        self.x_hatgroundTruth = np.matrix([[0],[0],[0],[0],[0]]) #ground truth for the states
        self.groundTruthList = [] #noting all the ground truth values for states in alist later used for plotting

        self.linAx = 0.0
        self.angVz = 0.0
        self.globalxposition = [0.00005791279699608568]
        self.globalyposition = [-9.939039374154975e-7]
        self.xDistance = []
        self.vx = []
        self.theta = [8.076071639054034e-07]


        self.flagOdom = False #flag used in odom callback for initialization of values
        self.flagQ = False  #flag used in imu callback for initialization of values
        self.flagk = True   #flag used in kalman filter timer triggered function to stop running after 75 iterations

        
        # The subscribers needed for the FIlter
        self.OdomSubscriber = self.create_subscription(Odometry,'/odom',self.update_OdomData,QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT))    #odom subscriber
        self.IMUSubscriber = self.create_subscription(Imu,'/imu',self.update_ImuData,QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT))   #IMU subscriber
        #self.EncoderSubscriber = self.create_subscription(JointState,'/joint_states',self.update_EncoderData,10)    #joint_states subscriber (encoders)   
        #self.TfSubscriber = self.create_subscription(JointState,'/tf',self.update_EncoderData,10)   #tf subscriber
        
        self.counter = 0    # iteration counter so that after defined number of iterations we can stop kalman filter
        
        #timer triggered kalman filter function
        self.timer = self.create_timer(self.dt, self.run_kalman_filter) #kalman filter running at a rate
        
    
    def update_ImuData(self,msg):
		#callback function for IMU msg and Data
        #getting acceleration in x and y, and angular velocity for input
        linAx = msg._linear_acceleration.x
        linAy = msg._linear_acceleration.y
        angVz = msg._angular_velocity.z
        self.linA = np.matrix([[linAx], [linAy]])
        self.u = np.matrix([[linAx], [linAy], [angVz]])
        self.uList.append(self.u)
        #update Q matrix first time only using the Covariance Matrix Multiplication
        if not self.flagQ:
            #this is only for the first time to set the initial values and runs only once
            self.xhatList.append(self.x_hat)
            self.timespanList.append(0.0)
            self.flagQ = True


    def update_OdomData(self,msg):
        #callback function Odom data
        # getting pose x, pose y, velocity in x, velocity in y and theta from odom
        posex = msg.pose.pose.position.x
        posey = msg.pose.pose.position.y
        tempQuaternion = [msg.pose.pose.orientation.x,msg.pose.pose.orientation.y,msg.pose.pose.orientation.z,msg.pose.pose.orientation.w]
        qx = tempQuaternion[0]
        qy = tempQuaternion[1]
        qz = tempQuaternion[2]
        qw = tempQuaternion[3]
        #convert quaternion to euler get roll pitch theta
        #theta is angle - rotation in z axis (yaw)
        # roll and pitch are rotations in x/y axis
        posetheta = self.euler_from_quaternion(qx,qy,qz,qw)
        velX = msg.twist.twist.linear.x
        velY = msg.twist.twist.linear.y
        self.y = np.matrix([[posex],[posey],[velX],[velY],[posetheta]])
        self.yList.append(self.y)
        self.x_hatgroundTruth = np.matrix([[posex],[posey],[velX],[velY],[posetheta]])
        if not self.flagOdom:
            #to make sure the data has been retrevied before the kalman filter starts
            self.x_hatgroundTruth = np.matrix([[posex],[posey],[velX],[velY],[posetheta]])
            self.groundTruthList.append(self.x_hatgroundTruth)
            self.flagOdom = True
        
    

    def euler_from_quaternion(self,x, y, z, w):
        """
        Convert a quaternion into euler angles (roll, pitch, theta)
        theta is rotation around z in radians (counterclockwise)
        """
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        theta_z = math.atan2(t3, t4)
        return theta_z # in radians
    

    def run_kalman_filter(self):
        if self.flagOdom ==True and self.flagQ==True and self.flagk == True:   #if the all data has been received from subscriptions and everything has been initialized
            print(self.timespanList)
            self.timespanList.append(self.timespanList[-1]+self.dt)   # for timespan in order to make a plot
            #Also recording the ground truth from odom to compare against the estimates
            self.groundTruthList.append(self.x_hatgroundTruth)
            
            #converting acceleration received from imu in global frame using rotation matrix
            ang = self.x_hat[4,0]
            rotationMatrix = np.matrix([[math.cos(ang),-1*math.sin(ang)],[math.sin(ang),math.cos(ang)]])
            vectorAcceleration = rotationMatrix*self.linA        
            
            # input matrix u with proper required values
            ui = np.matrix([[vectorAcceleration[0,0]],[vectorAcceleration[1,0]],[self.u[2,0]]])
            
            #########################################
            ###### Kalman Filter Estimation #########
            #########################################
            
            # Prediction update
            xhat_k = self.A * self.x_hat + self.B * ui # we do not put noise on our prediction
            P_predict = self.A*self.P*self.A.transpose() + self.Q
            # this co-variance is the prediction of essentially how the measurement and sensor model move together
            # in relation to each state and helps scale our kalman gain by giving
            # the ratio. By Definition, P is the variance of the state space, and
            # by applying it to the motion model we're getting a motion uncertainty
            # which can be propogated and applied to the measurement model and
            # expand its uncertainty as well

            # Measurement Update and Kalman Gain
            K = P_predict * self.C.transpose()*np.linalg.inv(self.C*P_predict*self.C.transpose() + self.R)
            # the pseudo inverse of the measurement model, as it relates to the model covariance
            # if we don't have a measurement for velocity, the P-matrix tells the
            # measurement model how the two should move together (and is normalised
            # in the process with added noise), which is how the kalman gain is
            # created --> detailing "how" the error should be scaled based on the
            # covariance. If you expand P_predict out, it's clearly the
            # relationship and cross-projected relationships, of the states from a
            # measurement and motion model perspective, with a moving scalar to
            # help drive that relationship towards zero (P should stabilise).

            self.x_hat = xhat_k + K * (self.y - self.C * xhat_k) # y is measurement
            self.P = (np.identity(5) - K * self.C) * P_predict # the full derivation for this is kind of complex relying on
                                                # some pretty cool probability knowledge

            self.xhatList.append(self.x_hat)
            self.counter+=1
            if self.counter>120:
                self.plot_results()
                self.flagk = False #stopping the Kalman Filter
        else:
            pass
        

    def plot_results(self):
        #function to plot the data
        lengthXhat = len(self.xhatList)
        listposeEstimateX = []
        listposeEstimateY = []
        listEstimateVx = []
        listEstimateVy = []
        listposeEstimateTheta = []
        listposeGTruthX = []
        listposeGTruthY = []
        listGTruthVx = []
        listGTruthVy = []
        listposeGTruthTheta = []

        #initializing variables for rms error
        rmsx = 0
        rmsy = 0
        rmsvx = 0
        rmsvy = 0
        rmstheta = 0
        counterRMS = 0

        # making individual lists of each state keeping in mind if it is estimate or ground truth for plotting
        for i in range(lengthXhat):
            listposeEstimateX.append(self.xhatList[i][0,0])
            listposeEstimateY.append(self.xhatList[i][1,0])
            listEstimateVx.append(self.xhatList[i][2,0])
            listEstimateVy.append(self.xhatList[i][3,0])
            listposeEstimateTheta.append(self.xhatList[i][4,0])
            listposeGTruthX.append(self.groundTruthList[i][0,0])
            listposeGTruthY.append(self.groundTruthList[i][1,0])
            listGTruthVx.append(self.groundTruthList[i][2,0])
            listGTruthVy.append(self.groundTruthList[i][3,0])
            listposeGTruthTheta.append(self.groundTruthList[i][4,0])
            #RMS calculations
            rmsx = math.pow((self.xhatList[i][0,0] - self.groundTruthList[i][0,0]),2)
            rmsy = math.pow((self.xhatList[i][1,0] - self.groundTruthList[i][1,0]),2)
            rmsvx = math.pow((self.xhatList[i][2,0] - self.groundTruthList[i][2,0]),2)
            rmsvy = math.pow((self.xhatList[i][3,0] - self.groundTruthList[i][3,0]),2)
            rmstheta = math.pow((self.xhatList[i][4,0] - self.groundTruthList[i][4,0]),2)
            counterRMS+=1

        # Remaining rms calculations
        rmsx = math.pow((rmsx/counterRMS),1/2)
        rmsy = math.pow((rmsy/counterRMS),1/2)
        rmsvx = math.pow((rmsvx/counterRMS),1/2)
        rmsvy = math.pow((rmsvy/counterRMS),1/2)
        rmstheta = math.pow((rmstheta/counterRMS),1/2)

        #printing RMS
        print(rmsy)
        print(rmsx)
        print(rmsvx)
        print(rmsvy)
        print(rmstheta)

        plt.figure(1)
        plt.plot(self.timespanList, listposeEstimateX)
        plt.plot(self.timespanList, listposeEstimateY)
        plt.plot(self.timespanList, listEstimateVx)
        plt.plot(self.timespanList, listEstimateVy)
        plt.plot(self.timespanList, listposeEstimateTheta)
        plt.legend(['position x est.','position y est.', 'velocity x est.', 'velocity y est.', 'theta est.'])

        plt.figure(2)
        plt.plot(self.timespanList, listposeGTruthX)
        plt.plot(self.timespanList, listposeGTruthY)
        plt.plot(self.timespanList, listGTruthVx)
        plt.plot(self.timespanList, listGTruthVy)
        plt.plot(self.timespanList, listposeGTruthTheta)
        plt.legend(['pos x g. Truth','pos y g. Truth', 'vel x g. Truth', 'vel y g. Truth', 'theta g. Truth'])

        plt.figure(3)
        plt.plot(self.timespanList, listposeEstimateX)
        plt.plot(self.timespanList, listposeGTruthX)
        plt.legend(['position x est.','pos x g. Truth'])

        plt.figure(4)
        plt.plot(self.timespanList, listposeEstimateY)
        plt.plot(self.timespanList, listposeGTruthY)
        plt.legend(['position y est.','pos y g. Truth'])

        plt.figure(5)
        plt.plot(self.timespanList, listEstimateVx)
        plt.plot(self.timespanList, listGTruthVx)
        plt.legend(['velocity x est.','vel x g. Truth'])

        plt.figure(6)
        plt.plot(self.timespanList, listEstimateVy)
        plt.plot(self.timespanList, listGTruthVy)
        plt.legend(['velocity y est.','vel y g. Truth'])

        plt.figure(7)
        plt.plot(self.timespanList, listposeEstimateTheta)
        plt.plot(self.timespanList, listposeGTruthTheta)
        plt.legend(['theta est.','theta g. Truth'])

        plt.show()




def main(args=None):
    rclpy.init(args=args)

    kalmanFilter = KalmanFilterNode()

    rclpy.spin(kalmanFilter)


if __name__ == '__main__':
    main()