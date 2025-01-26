#!/usr/bin/env python3

import numpy as np
import cv2 as cv
import random
import matplotlib.pyplot as plt

class rrt_navigator():
    def __init__(self):
        self.img = cv.imread('/home/cse4568/catkin_ws/src/cse4568_pa4/maps/map2.png', cv.IMREAD_GRAYSCALE)
        self.tree = []
        self.max_dist=10
        start_x =2
        start_y = 5
        goal_x =700
        goal_y = 200
        self.path((start_x, start_y), (goal_x, goal_y))
        
    def dist(self, start, end):
        dir = np.array(end) - np.array(start)
        dist = min(self.max_dist, np.linalg.norm(dir))
        if np.any(dir):
            pt = start + dist * (dir / np.linalg.norm(dir))
            if not np.any(np.isnan(pt)) and self.img[int(pt[1]), int(pt[0])]:
                return int(pt[0]), int(pt[1])
        return start

    def path(self, start, goal):
        self.tree.append((start, None)) 
        height, width = self.img.shape
        
        while True:
            x = random.randint(0, width - 1)
            y = random.randint(0, height - 1)
            if self.img[y, x]:
                rand = x,y
            distances = [np.linalg.norm(np.array(rand) - np.array(vertex[0])) for vertex in self.tree]
            idx = np.argmin(distances)
            n_idx=idx
            new = self.dist(self.tree[n_idx][0], rand)

            if new != self.tree[n_idx][0]:
                self.tree.append((new, n_idx)) 
                
            if np.linalg.norm(np.array(new) - np.array(goal)) < 10: 
                return self.tree

if __name__ == "__main__":
    navigator = rrt_navigator()
    

    
    
    
    