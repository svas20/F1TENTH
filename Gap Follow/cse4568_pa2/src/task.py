#! /usr/bin/env python3

import rospy
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDrive
import math 

class gap:
    def __init__(self) -> None:
        self.sub=rospy.Subscriber("/car_1/scan",LaserScan,callback=self.gap_follow)
        self.pub=rospy.Publisher("/car_1/command",AckermannDrive,queue_size=3)
        self.ackermann = AckermannDrive()

    def gap_follow(self,scan):
        
        angle_max=[]
        ang_min= -75*math.pi/180
        ang_max= 75*math.pi/180
        ang_inc=0.0043633
        angle_seen=[]
        idx_ang=[]
        sr=[]
        for idx in range(len(scan.ranges)):
            angle_max.append(scan.angle_min+idx*(scan.angle_increment))
        #print(angle_max)
        for idx,val in enumerate(angle_max):
            if val>ang_min and val<ang_max:
                angle_seen.append(val)
                idx_ang.append(idx)
                sr.append(scan.ranges[idx])
        #print(angle_seen)
        #print(idx_ang)
        #print(sr)
        v=[]
        for range1 in sr:
            if range1>=1.5:
                v.append(0.0)
            else:
                v.append(range1)
        #print(v)

        count=0
        final_count=0
        index=[]
        final=[]
        for idx,range2 in enumerate(v):
            if range2==0:
                count=count+1
                index.append(idx)
            else:
                count=0
                index=[]
            #print(count)
            if count>final_count:
                final_count=count
                final=index.copy()
        #print(final)

        mid= int((final[0] + final[-1]) / 2)
        self.ackermann.steering_angle = -angle_seen[mid]
        self.ackermann.speed=2
        self.pub.publish(self.ackermann) 


                      
if __name__=='__main__':
    rospy.init_node("gap")
    gap()
    rate=rospy.Rate(1)
    while not rospy.is_shutdown():
        rate.sleep()