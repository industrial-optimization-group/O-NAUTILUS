import numpy as np
from pygmo import fast_non_dominated_sorting as nds
from typing import List


class ONAUTILUS:
    def __init__(
        self, known_data: np.ndarray, optimistic_data: np.ndarray, num_steps: int = 10
    ):
        self.known_data = known_data
        self.optimistic_data = optimistic_data
        self.num_steps = num_steps

        self.ideal = known_data.min(axis=0)
        self.non_dominated_known = known_data[nds(known_data)[0][0]]
        self.nadir = self.non_dominated_known.max(axis=0)
        self.steps_taken = 0

        self.nadir_to_ideal = self.ideal - self.nadir
        #  self.step = self.nadir_to_ideal / np.linalg.norm(self.nadir_to_ideal)
        self.step = self.nadir_to_ideal / num_steps
        self.preference_point = self.ideal
        self.preference_points_list = []
        self.current_point = self.nadir
        self.previous_points_list = [self.nadir]
        self.improvement_direction = None
        self.currently_achievable: List = None

    def iterate(self, preference: np.ndarray = None):
        self.preference_point = (
            preference if preference is not None else self.preference_point
        )
        self.preference_points_list.append(self.preference_point)
        previous_point = self.previous_points_list[-1]
        improvement_direction = self.preference_point - previous_point
        improvement_direction = improvement_direction / np.linalg.norm(
            improvement_direction
        )
        cos_theta = np.dot(improvement_direction, self.step) / (
            np.linalg.norm(self.step)
        )
        step = improvement_direction * np.linalg.norm(self.step) / cos_theta
        current_point = previous_point + step
        self.previous_points_list.append(current_point)
        achievable_ids = np.nonzero(
            (self.non_dominated_known <= current_point).all(axis=1)
        )[0]
        self.steps_taken += 1
        self.currently_achievable = achievable_ids

    def requests_plot(self):
        if self.steps_taken == 0:
            achievable = self.non_dominated_known
        else:
            achievable = self.non_dominated_known[self.currently_achievable]
        if len(achievable) == 0:
            return (
                [],
                [],
                self.preference_point,
                self.ideal,
                self.nadir,
                self.steps_taken,
            )
        lower_bounds = achievable.min(axis=0)
        upper_bounds = achievable.max(axis=0)
        bounds = np.vstack((lower_bounds, upper_bounds)).T
        return (
            bounds,
            self.non_dominated_known,
            self.currently_achievable,
            self.preference_point,
            self.previous_points_list[-1],
            self.ideal,
            self.nadir,
            self.steps_taken,
            self.num_steps,

        )

