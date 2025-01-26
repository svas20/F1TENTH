#! /usr/bin/env python3
import RacingTests


class PFRunner:
    def __init__(self, init_position, step_range, odm_cmd_loc) -> None:
        self.robot_action_noise = 0.4# Scalar float
        self.robot_measurement_noise = 0.3  # Scalar float

        self.pf_state_noise = [0.06,0.1,0.17] # 1x3 Vector float
        self.pf_measurement_noise = [0.2]  # 1x1 Vector float

        self.initial_noise = [0,0,0]  # 1x3 Vector float

        self.steps_range = step_range  # 1x2 Vector int
        self.particles = 100  # Scalar int

        self.initial_state = init_position  # 1x3 Vector float

        self.odom_commands = odm_cmd_loc  # String location of numpy array

    def run(self):
        estimated, gts, noisy = RacingTests.simulate_car_pf(
            self.robot_action_noise,
            self.robot_measurement_noise,
            self.pf_state_noise,
            self.pf_measurement_noise,
            self.initial_noise,
            self.initial_state,
            self.odom_commands,
            self.steps_range,
            self.particles,
        )
        return estimated, gts, noisy


if __name__ == "__main__":
    init_position = [0, 0, 0]  # We will change these for evaluation
    step_range = [0, 400]  # We will change these for evaluation
    odom_command_location = "odom_commands.npy"  # We will change these for evaluation
    runner = PFRunner(init_position, step_range, odom_command_location)
    runner.run()
