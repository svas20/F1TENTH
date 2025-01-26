#! /usr/bin/env python3

import rospy
import cv2
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import Header
from geometry_msgs.msg import Pose,Point, Quaternion
import numpy as np

class occupy:
    def __init__(self) -> None:
        self.sub=rospy.Subscriber("/car_1/scan",LaserScan,callback=self.grid)
        self.pub=rospy.Publisher("/local_map",OccupancyGrid,queue_size=3)
        self.dist=rospy.get_param('max_distance')
        self.res=rospy.get_param('map_resolution')
        self.dil=rospy.get_param('dilation_kernel')
        self.kernal=np.ones((self.dil,self.dil),dtype=np.uint8)
    def grid(self,scan):
        ranges=scan.ranges
        grid=int(2*self.dist/self.res)
        occupancy=OccupancyGrid()
        occupancy.header=Header()
        occupancy.header.frame_id="car_1/laser"
        occupancy.header.stamp = rospy.Time.now()
        occupancy.info.resolution=self.res
        occupancy.info.width=grid
        occupancy.info.height=grid
        occupancy.info.origin=Pose(Point(-self.dist,-self.dist,0),Quaternion(0,0,0,1))
        occupy=np.zeros((grid,grid),dtype=np.int8)
        for idx,value in enumerate(ranges):
            if value<=self.dist:
                angle=scan.angle_min+idx*scan.angle_increment
                x=value*np.cos(angle)
                y=value*np.sin(angle)
                
                i=int((x+occupancy.info.origin.position.x)/occupancy.info.resolution)
                j=int((y+occupancy.info.origin.position.y)/occupancy.info.resolution)
                occupy[j][i]=100

        occupy_dilate=cv2.dilate(occupy.astype(dtype=np.uint8),self.kernal)
        occupancy.data=occupy_dilate.flatten().tolist()
        print(occupancy.data)
        self.pub.publish(occupancy)
if __name__=="__main__":
    rospy.init_node("occupancy_grid",anonymous=True)
    rate=rospy.Rate(10)
    while not rospy.is_shutdown():
        occupy()
        rate.sleep()
    rospy.spin()
        


            



