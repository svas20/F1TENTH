#! /usr/bin/env python3
import rospy
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import PointStamped
from nav_msgs.msg import Path,OccupancyGrid
from queue import PriorityQueue
from ackermann_msgs.msg import AckermannDrive
import numpy as np

class PathPlanner:
    def __init__(self):
        rospy.init_node('path_planner', anonymous=True)
        rospy.Subscriber('/map', OccupancyGrid, self.grid_callback)
        rospy.Subscriber('/clicked_point', PointStamped, self.goal_callback)
        rospy.Publisher('/car_1/command',AckermannDrive,queue_size=10)
        rospy.spin()
        self.occupancy_grid = None
        self.goal = None

    def grid_callback(self, msg):
        self.occupancy_grid = np.array(msg.data).reshape((msg.info.height, msg.info.width))

    def goal_callback(self, msg):
        self.goal = [int(msg.point.y), int(msg.point.x)]
        self.path()

    def heuristic(self, a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def path(self):
        if self.occupancy_grid is not None and self.goal is not None:
            start = (int(0), int(0)) 
            dir = [(1, 0), (0, 1),(-1, 0),(0, -1), (1, 1), (1, -1),(-1, 1), (-1, -1)]
            costs = [1, 1, 1, 1, np.sqrt(2), np.sqrt(2), np.sqrt(2), np.sqrt(2)]
            open_set = PriorityQueue()
            open_set.put((0, start))
            cost_so_far, came_from = {start: 0}, {start: None}

            while not open_set.empty():
                _, current = open_set.get()
                if current == tuple(self.goal):
                    break
                for i, direction in enumerate(dir):
                    neighbor = (current[0] + direction[0], current[1] + direction[1])
                    if (
                        0 <= neighbor[0] < self.occupancy_grid.shape[0]
                        and 0 <= neighbor[1] < self.occupancy_grid.shape[1]
                        and self.occupancy_grid[neighbor] == 0
                    ):
                        new_cost = cost_so_far[current] + costs[i]
                        if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                            cost_so_far[neighbor] = new_cost
                            priority = new_cost + self.heuristic(neighbor, self.goal)
                            open_set.put((priority, neighbor))
                            came_from[neighbor] = current

            current = self.goal
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path=path[::-1]

        drive_msg = AckermannDrive()
        drive_msg.speed = 1.0
        drive_msg.steering_angle = 0.0

        if len(path) > 1:
            next_point = path[1]
            current_point = path[0]
            delta_x = next_point[0] - current_point[0]
            steering_angle = np.arctan2(delta_x, 1.0)
            drive_msg.steering_angle = steering_angle
        self.drive_publisher.publish(drive_msg)

if __name__ == '__main__':
    try:
        planner = PathPlanner()
    except rospy.ROSInterruptException:
        pass