#! /usr/bin/env python3
"""
Adapted from: https://github.com/BDEvan5/sensor_fusion
"""

import numpy as np
from Gaussian import Gaussian
from AutonomousRacer import AutonomousRacer
from matplotlib import pyplot as plt
from ParticleFilter import ParticleFilter


def simulate_racing(robot, estimator, steps):
    for _ in range(steps - 1):
        control = robot.move()
        estimator.control_update(control)
        measurement = robot.measure()
        if measurement is not None:
            estimator.measurement_update(measurement)


def simulate_car_pf(
    robot_action_noise,
    robot_scan_noise,
    pf_state_noise,
    pf_measurement_noise,
    initial_noise,
    init_state,
    odom_command_str,
    step_range,
    particle_count,
):
    Q = np.diag(pf_state_noise)
    R = np.diag(pf_measurement_noise)
    NP = particle_count

    init_state = np.array(init_state)
    init_belief = Gaussian(init_state, np.diag(initial_noise))

    """
    Load the numpy array here
    Format is [[x,y,theta,v,phi], ...]
    Use the step_range to index the array
    """
    odo=np.load(odom_command_str)
    odom_list=odo[step_range[0]:step_range[1],:][:,:3]
    cmd_list=odo[step_range[0]:step_range[1],:][:,3:]


    robot = AutonomousRacer(
        init_state, robot_action_noise, robot_scan_noise, odom_list, cmd_list
    )
    particle_filter = ParticleFilter(init_belief, robot.f, robot.h, Q, R, NP)

    simulate_racing(robot, particle_filter, len(odom_list))

    true_states, noisy_states = robot.get_states()

    estimates = particle_filter.get_estimated_states()

    # Uncomment to plot the results (not required for submission)
    # plot_racing_localistion(estimates, true_states, noisy_states, robot)

    return estimates, true_states, noisy_states


def plot_racing_localistion(estimates, true_states, noisy_states, robot):
    plt.figure(figsize=(8, 8))
    map_img = robot.scan_simulator.map_img
    map_img[map_img == 0] = 100
    map_img[0, 0] = 0
    plt.imshow(map_img, cmap="gray", origin="lower")
    true_states = robot.scan_simulator.xy_2_rc(true_states)
    noisy_states = robot.scan_simulator.xy_2_rc(noisy_states)
    estimates = robot.scan_simulator.xy_2_rc(estimates)

    plt.plot(true_states[:, 0], true_states[:, 1], color="k", label=f"True trajectory")
    plt.plot(noisy_states[:, 0], noisy_states[:, 1], color="b", label="Dead rekoning")
    plt.plot(estimates[:, 0], estimates[:, 1], "x-", color="r", label="Estimated PF")

    plt.legend()
    plt.show()
