#! /usr/bin/env python3
"""
Adapted from: https://github.com/BDEvan5/sensor_fusion
"""

import numpy as np
from ScanSimulator import ScanSimulator2D
from matplotlib import pyplot as plt


class AutonomousRacer:
    def __init__(
        self,
        init_state,
        action_noise,
        scan_noise,
        full_odometry,
        full_commands,
        WB=0.324,
        dt=0.025,
    ):
        self.noisy_states = [init_state]
        self.true_states = [init_state]
        self.meaurements = []

        self.Q = np.diag([action_noise])  # action noise
        self.R = np.diag([scan_noise])  # Observation noise for the LiDAR.
        self.T_s = 0

        self.NUM_BEAMS = 10

        self.scan_simulator = ScanSimulator2D("map", self.NUM_BEAMS, np.pi)
        self.measure()  # to correct number of measurements

        self.odoms = full_odometry.reshape(-1, 3)
        self.commands = full_commands.reshape(-1, 2)

        self.true_states = self.odoms

        self.WB = WB
        self.dt = dt

        # Add any other variables you need here

    def move(self):
        # Get the correct command
        # Add action noise to the command
        # Get the previous noisy state
        # Get the next noisy state using the dynamics
        # Add the new noisy state to the list of noisy states
        # Return the noisy control
        cur = self.commands[self.T_s]
        noi = np.random.multivariate_normal(np.zeros(self.Q.shape[0]), self.Q)
        noisy_u = noi+cur
        if self.T_s >0:
            prev= self.noisy_states[self.T_s-1]
        else:
            prev=self.noisy_states[self.T_s]
        noisy_st = self.dynamics([prev],noisy_u)[0]
        self.noisy_states.append(noisy_st)
        self.T_s += 1
        return noisy_u

    def measure(self):
        pose = self.true_states[-1]

        scan = self.scan_simulator.scan(pose)

        # add observation noise to scan to get measurements
        measurements = scan + np.random.normal(0, self.R[0], self.NUM_BEAMS)
        self.meaurements.append(measurements)

        return measurements

    def get_states(self):
        return np.array(self.true_states), np.array(self.noisy_states)

    def get_measurements(self):
        np.array(self.meaurements)

    def f(self, state, u):
        new_state = self.dynamics(state, u)

        return new_state

    def h(self, states):
        scans = np.zeros((states.shape[0], self.NUM_BEAMS))
        for i, state in enumerate(states):
            scans[i] = self.scan_simulator.scan(state)

        return scans

    def dynamics(self, states, control):
        """
        states: (N,3)
        control: (1,2)
        Returns: (N,3) which are the new states for the same control
        """
        x_new=[]
        y_new=[]
        theta_new=[]
        v,st_an=control
        dt = self.dt
        for i in states:
            theta = i[2] + v/self.WB* np.tan(st_an) * dt
            theta %= 2*np.pi
            x= i[0] + v * np.cos(theta) * dt
            y = i[1] + v * np.sin(theta) * dt

            x_new.append(x)
            y_new.append(y)
            theta_new.append(theta)
        return np.stack([x_new, y_new, theta_new], axis=1)
