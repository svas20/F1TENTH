#!/usr/bin/env python3

import rospy
import numpy as np
import csv
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
from ackermann_msgs.msg import AckermannDrive
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from nav_msgs.msg import OccupancyGrid
from visualization_msgs.msg import Marker, MarkerArray
import tf.transformations as tf
from task_1 import rrt_navigator

class RRT():
    def __init__(self):
        self.csv =rospy.get_param('centerline_csv_path','/home/cse4568/catkin_ws/src/cse4568_pa4/maps/task2_centerline.csv')
        self.look_ahead=(0,0)
        self.pose=(0,0)
        self.look=4
        self.occ_grid_data=[]
        self.navigate=[]
        self.look_no_obs=(0,0)
        self. mark_pub = rospy.Publisher('csv_data_markers', Marker, queue_size=10)
        self.comm_pub=rospy.Publisher('/car_1/command',AckermannDrive,queue_size=10)
        self.og_pub=rospy.Publisher('occupancy_grid',OccupancyGrid,queue_size=10)
        self.laser_pub = rospy.Publisher('/laser_scan_markers', MarkerArray, queue_size=10)

        

        self.og_sub=rospy.Subscriber('/car_1/scan',LaserScan,self.occ)
        self.odom_sub=rospy.Subscriber('/car_1/odom',Odometry,self.odo)
        self.laser_sub=rospy.Subscriber('/car_1/scan',LaserScan,self.laser)
        
        with open(self.csv, 'r') as file:
            csv_reader = csv.reader(file)
            next(csv_reader)  
            self.data = [(float(row[0]), float(row[1])) for row in csv_reader]
            #print(self.data)

        self.c_line=self.publish_markers()
        

    def publish_markers(self):
        marker = Marker()
        marker.header.frame_id = "mark"
        marker.header.stamp = rospy.Time.now()
        marker.type = Marker.POINTS
        marker.action = Marker.ADD
        marker.scale.x = 0.005
        marker.scale.y = 0.005
        marker.color.a = 1.0
        marker.color.r = 1.0

        for x, y in self.data:
            point = Point()
            point.x = x
            point.y = y
            marker.points.append(point)

        self.mark_pub.publish(marker)
        #print(marker)
    
    def occ(self,oc_data):
        x=[]
        y=[]
        self.occupancy_grid = OccupancyGrid()
        self.occupancy_grid.header.frame_id = "mark"
        self.occupancy_grid.header.stamp=rospy.Time.now()
        self.occupancy_grid.info.width = 20
        self.occupancy_grid.info.height = 20
        self.occupancy_grid.info.resolution = 0.1
        #occupancy_grid.info.origin=occupancy_grid.info.height//2

        o_g = np.zeros((self.occupancy_grid.info.width, self.occupancy_grid.info.height), dtype=int)
        ang = np.arange(oc_data.angle_min, oc_data.angle_max, oc_data.angle_increment)
        for range in oc_data.ranges:
            x = np.round(range * np.cos(ang) / self.occupancy_grid.info.resolution ) + self.occupancy_grid.info.width
            y = np.round(range * np.sin(ang) / self.occupancy_grid.info.resolution ) + self.occupancy_grid.info.height
        for x, y in zip(x.astype(int), y.astype(int)):
            if 0 <= x < self.occupancy_grid.info.width and 0 <= y < self.occupancy_grid.info.height:
                o_g[y, x] = 1
        o_g = o_g.flatten()
        #print(o_g)

        self.occupancy_grid.data=o_g.tolist()
        self.occ_grid_data=self.occupancy_grid.data
        #print(self.occ_grid_data)
        self.og_pub.publish(self.occupancy_grid)

    def odo(self,od_data):
        distances = []
        x_curr=od_data.pose.pose.position.x
        y_curr=od_data.pose.pose.position.y
        self.pose=(x_curr,y_curr)
        quaternion = (
        od_data.pose.pose.orientation.x,
        od_data.pose.pose.orientation.y,
        od_data.pose.pose.orientation.z,
        od_data.pose.pose.orientation.w
    )
        euler = tf.euler_from_quaternion(quaternion)
        positions = np.array(self.data)
        current_position = np.array([x_curr, y_curr])
        distances = np.linalg.norm(positions - current_position, axis=1)
        min_distance = np.min(distances)
        min_index = np.argmin(distances)
        self.look_ahead_fn(min_index,euler,x_curr,y_curr)

    def look_ahead_fn(self,min_index,euler,x_curr,y_curr):
        self.look=self.look+1
        ld_x, ld_y = self.data[min_index+self.look]
        self.look_ahead=(ld_x,ld_y)
        rotation_matrix = np.array([[np.cos(euler[2]), -np.sin(euler[2])],
                            [np.sin(euler[2]), np.cos(euler[2])]])
        transformation_matrix = np.vstack((np.hstack((rotation_matrix, np.array([[x_curr], [y_curr]]))),
                                   np.array([0, 0, 1])))
        inverse_transformation_matrix = np.linalg.inv(transformation_matrix)
        self.look_ahead = np.dot(inverse_transformation_matrix, np.array(self.look_ahead + (1,)))
        #print("Transformed Look Ahead Vector:", transformed_look_ahead[:2])

        row,col,x,y=self.obstacle()
        if self.occ_grid_data[row][col]==0:
            self.look_no_obs(x,y)
            self.navigate()
        else:
            self.look_ahead_fn(min_index,euler,x_curr,y_curr)
           
    def obstacle(self):
        x_look_ahead = self.look_ahead[0]
        y_look_ahead = self.look_ahead[1]
        row_val = int((20+ y_look_ahead) / 0.1)
        col_val = int((20+ x_look_ahead) / 0.1)
        return row_val,col_val,x_look_ahead,y_look_ahead
    
    def navigator(self):
        Occupancy=np.array(self.occ_grid_data)
        start=self.pose
        end=self.look_ahead
        r=rrt_navigator()
        r.img=Occupancy
        self.navigate=r.path(start,end)
        
        self.steer()

    def steer(self):
        k_dd=5
        ackermann=AckermannDrive()
        ackermann.speed=4
        self.comm_pub.publish(ackermann)
        l_d=k_dd*ackermann.speed
        
         
                
    def laser(self, laser_scan):
        markers = MarkerArray()
        ranges = np.array(laser_scan.ranges)
        angles = np.arange(laser_scan.angle_min, laser_scan.angle_max, laser_scan.angle_increment)

        for i in range(len(ranges)):
            marker = Marker()
            marker.header.frame_id = "mark"
            marker.header.stamp = rospy.Time.now()
            marker.id = i
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position.x = ranges[i] * np.cos(angles[i])
            marker.pose.position.y = ranges[i] * np.sin(angles[i])
            marker.pose.position.z = 0
            marker.pose.orientation.x = 0
            marker.pose.orientation.y = 0
            marker.pose.orientation.z = 0
            marker.pose.orientation.w = 1
            marker.scale.x = 0.1
            marker.scale.y = 0.1
            marker.scale.z = 0.1
            marker.color.a = 1.0
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0

            markers.markers.append(marker)
            self.laser_pub.publish(markers)

    

if __name__=="__main__":
    rospy.init_node('RRT', anonymous=True)
    rate = rospy.Rate(1)
   
    while not rospy.is_shutdown():
        RRT()
        rate.sleep()


