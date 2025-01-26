#!/usr/bin/env python3

import rospy
import numpy as np
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge,CvBridgeError
from nav_msgs.msg import Odometry
from nav_msgs.msg import Path
from svo.srv import save_path
from geometry_msgs.msg import Pose,PoseStamped,Point
from time import time
from std_msgs.msg import String
import csv
import cv2

BASELINE=0.07
FOCAL_LENGTH=476.703083601

class StereoVisualOdometry:
    def __init__(self):
        rospy.init_node("stereo_visual_odometry_node", anonymous=True)
        self.bridge = CvBridge()
        self.left_img_sub = rospy.Subscriber(
            "/car_1/camera/left/image_raw/compressed", CompressedImage, self.left_img_callback_detect
        )
        self.right_img_sub = rospy.Subscriber(
            "/car_1/camera/right/image_raw/compressed", CompressedImage, self.right_img_callback_detect
        )
        #self.intrinsic=rospy.Subscriber("/car_1/camera/intrinsic",String,self.intrinsic)
        cameraMatrix_0_2 = 400.5
        cameraMatrix_1_2 = 400.5
        cameraMatrix_0_0 = 476.70
        cameraMatrix_1_1 = 476.70
        self.intr=np.array([[cameraMatrix_0_0, 0, cameraMatrix_0_2],
                            [0, cameraMatrix_1_1, cameraMatrix_1_2],
                            [0, 0, 1]])
        self.odom_pub = rospy.Publisher("/vo/odom", Odometry, queue_size=10)
        self.path_pub = rospy.Publisher("/vo/path",Path,queue_size=10)      
        
        self.lk_params = dict(winSize=(15, 15),maxLevel=2,criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

        self.save_trajectory_service = rospy.Service('save_path', save_path, self.save_trajectory_handler)

        
        self.prev_frame=None
        self.path= Path()
        self.right_gray=None
        self.left_gray=None
        self.prev_kp=None
        self.prev_pose=np.eye(4)

    def left_img_callback_detect(self, data):
        try:
            compressed_data = data.data
            np_arr_left = np.frombuffer(compressed_data, np.uint8)
            left_img = cv2.imdecode(np_arr_left, cv2.IMREAD_COLOR)
        except CvBridgeError as e:
            rospy.logerr(e)

        self.left_gray = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
        #cv2.imshow("gray",self.left_gray)
        #cv2.waitKey(0)
        if self.prev_frame is not None:
            if self.prev_kp is not None:
                orb = cv2.ORB_create()
                keypoints = orb.detect(self.left_gray, None)
                # Calculate optical flow using Lucas-Kanade method
                new_keypoints, status, _ = cv2.calcOpticalFlowPyrLK(
                self.prev_frame, self.left_gray, self.prev_kp, None, **self.lk_params) #status indicates whether the flow was found for each keypoint (1 for found, 0 for not found)
                                                                                       #new_keypoints, which contains the updated locations of the keypoints in the current frame. 
                #print(new_keypoints)                        
                self.prev_kp= cv2.KeyPoint_convert(keypoints)
                self.prev_frame = self.left_gray.copy()
                
                prev_3d =self.obj(self.prev_kp)
                self.compute_odo(prev_3d,new_keypoints)
                self.tranf=self.pubodo()
                
            else:
                rospy.logwarn("Previous keypoints are not set.")

        else:
            # First frame, initialize keypoints
            orb = cv2.ORB_create()
            keypoints = orb.detect(self.left_gray, None)
            self.prev_kp = cv2.KeyPoint_convert(keypoints) #The detected keypoints are then converted to a NumPy array 
            self.prev_frame = self.left_gray.copy()
        

    def right_img_callback_detect(self, data):
        try:
            compressed_data = data.data
            np_arr_right = np.frombuffer(compressed_data, np.uint8)
            right_img = cv2.imdecode(np_arr_right, cv2.IMREAD_COLOR)
        except CvBridgeError as e:
            rospy.logerr(e)

        self.right_gray = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)
        orb = cv2.ORB_create()
        keypoints = orb.detect(self.right_gray, None)
        img_with_keypoints = cv2.drawKeypoints(self.right_gray, keypoints, None, color=(0, 255, 0), flags=0)
        #cv2.imshow("right",img_with_keypoints)
        #cv2.waitKey(0)

    def obj(self,kp):
        if self.left_gray is not None and self.right_gray is not None:
            stereo = cv2.StereoBM_create(numDisparities=192, blockSize=7)
            disparity = stereo.compute(self.left_gray, self.right_gray)
            norm_dis = cv2.normalize(disparity, None, alpha=5, beta=25, norm_type=cv2.NORM_MINMAX)
            #disparity = np.where(disparity == 0, 1.0, disparity)
            depth = (FOCAL_LENGTH * BASELINE) / norm_dis
            #depth = 255 - depth #inversion for visuvalization purpose
            #print(depth)
            

            # Assuming you have depthMap as a valid depth map corresponding to your stereo images
            objectPoints = []
            cm = self.intr
            for pt in kp:
                x, y = pt.ravel()
                x, y = int(x), int(y)
                #print(x,y)
                depth_value = depth[y, x]
                X = (x - cm[0, 2]) * depth_value / cm[0, 0]
                Y = (y - cm[1, 2]) * depth_value / cm[1, 1]
                Z = depth_value
                objectPoints.append([X, Y, Z])
            objectPoints = np.array(objectPoints, dtype=np.float32)
            objp=objectPoints.reshape(-1,1,3)
        return objp
    
    def compute_odo(self,d_3,d_2):
        #print(d_3.shape)
        #print(d_2.shape)
        cm=np.array(self.intr)
        #print(cm)
        #print(cm.shape)
        new_2d=np.array(d_2,dtype=np.float32)
        #print(new_2d)
        #print(d_3)
        _,rot,tran,inliers=cv2.solvePnPRansac(d_3,new_2d,cm,None)
        #print(f"rot {rot}")
        #print(f"trans {tran}")
        homo_mat = np.eye(4)
        rot_mat, _ = cv2.Rodrigues(rot)
        homo_mat[:3, :3] = rot_mat
        homo_mat[:3, 3] = tran.flatten()
        #print(homo_mat)
        self.prev_pose=np.dot(self.prev_pose,homo_mat)
        #print(self.prev_pose)

    def pubodo(self):
        position = self.prev_pose[:3, 3]
        #print(f"pubodo{self.prev_pose}")
        header_msg=Odometry()
        #header_msg.header.seq
        header_msg.header.stamp = rospy.Time.from_sec(time())
        header_msg.header.frame_id = "odom"
        header_msg.pose.pose.position = Point(*position)
        #print(header_msg.pose.pose.position.x)
        rotation_matrix = self.prev_pose[:3, :3]

        rotation_quaternion = np.zeros(4)
        rotation_quaternion[3] = 0.5 * np.sqrt(1 + rotation_matrix[0, 0] + rotation_matrix[1, 1] + rotation_matrix[2, 2])
        rotation_quaternion[0] = (rotation_matrix[2, 1] - rotation_matrix[1, 2]) / (4 * rotation_quaternion[3])
        rotation_quaternion[1] = (rotation_matrix[0, 2] - rotation_matrix[2, 0]) / (4 * rotation_quaternion[3])
        rotation_quaternion[2] = (rotation_matrix[1, 0] - rotation_matrix[0, 1]) / (4 * rotation_quaternion[3])
        
        header_msg.pose.pose.orientation.x = rotation_quaternion[0]
        header_msg.pose.pose.orientation.y = rotation_quaternion[1]
        header_msg.pose.pose.orientation.z = rotation_quaternion[2]
        header_msg.pose.pose.orientation.w = rotation_quaternion[3]

        self.odom_pub.publish(header_msg)
        #print(header_msg)


        path = Path()
        path.header.frame_id="path"
        path_msg=PoseStamped()
        path_msg.header.frame_id="pose"
        path_msg.pose.position=Point(*position)
        path_msg.pose.orientation.x = rotation_quaternion[0]
        path_msg.pose.orientation.y = rotation_quaternion[1]
        path_msg.pose.orientation.z = rotation_quaternion[2]
        path_msg.pose.orientation.w = rotation_quaternion[3]
        
        self.path.poses.append(path_msg)
        #print(path.poses)
        self.path_pub.publish(path)
        print(path_msg)

    def save_trajectory_handler(self, req):
        try:
            with open(req.filename, 'w', newline='') as csvfile:
                fieldnames = ['x', 'y', 'z', 'orientation_x', 'orientation_y', 'orientation_z', 'orientation_w']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()

                for pose_stamped in self.path.poses:
                    position = pose_stamped.pose.position
                    orientation = pose_stamped.pose.orientation
                    timestamp = pose_stamped.header.stamp.to_sec()
                    writer.writerow({
                        'timestamp': timestamp,
                        'x': position.x,
                        'y': position.y,
                        'z': position.z,
                        'orientation_x': orientation.x,
                        'orientation_y': orientation.y,
                        'orientation_z': orientation.z,
                        'orientation_w': orientation.w
                    })

            rospy.loginfo(f"Trajectory saved to {req.filename}")
            return True
        except Exception as e:
            rospy.logerr(f"Failed to save trajectory: {str(e)}")
            return False
           
    def run(self):
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            rate.sleep()


if __name__ == "__main__":
    svo = StereoVisualOdometry()
    try:
        svo.run()
    except rospy.ROSInterruptException:
        pass
