#! /usr/bin/env python3

import numpy as np


def perform_DR(odom_command_array: str, init_state: list, WB: float, dt: float) -> list:
    """
    This function calculates the dead reckoning pose of the robot
    :param odom_command_array: String path of odometry-commands array (N x 5) [x,y,theta,velocity,steering_angle]
    :param init_state: Initial state of the robot [x,y,theta]
    :param WB: Wheelbase of the robot
    :param dt: Time step
    :return: dead reckoning pose of the robot (N x 3) [x,y,theta]
    """
    # Load the array from the .npy file
    odom_data = np.load(odom_command_array)
    print(odom_data.dtype)
    pos_2d=np.array(init_state, dtype=np.float64)
    pos=[init_state]
    for i in range(len(odom_data)):
        vel, st_ang = odom_data[i,3], odom_data[i,4]
        st_ang %= 2 * np.pi
        pos_2d[0] +=vel*np.cos(pos_2d[2])*dt
        pos_2d[1] +=vel*np.sin(pos_2d[2])*dt
        pos_2d[2] +=vel/WB*np.tan(st_ang)*dt

        pos.append(pos_2d.copy())
    return pos
"""
if __name__=="__main__":

    file_path = '/home/cse4568/catkin_ws/src/particle_filter/task2/odom_commands.npy'

    initial_state = [0.0, 0.0, 0.0]
    wheelbase = 0.324
    time_step = 100*(10**-3)

    res=perform_DR(file_path, initial_state, wheelbase, time_step)
    print(res)
"""
