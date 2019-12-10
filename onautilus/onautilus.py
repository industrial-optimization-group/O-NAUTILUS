import numpy as np
import pandas as pd
from pygmo import fast_non_dominated_sorting as nds
from typing import List
from desdeo_tools.interaction.request import ReferencePointPreference, SimplePlotRequest


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
        self.step_size = self.nadir_to_ideal / num_steps
        # self.preference_point = self.ideal
        self.preference_point = np.asarray([0.2, 4])
        self.current_point = self.nadir
        self.current_points_list = [self.nadir]
        self.improvement_direction = self.preference_point - self.current_point
        self.improvement_direction = self.improvement_direction / np.linalg.norm(
            self.improvement_direction
        )
        self.currently_achievable: List = range(len(self.non_dominated_known))
        self.achievable_ranges = np.vstack((self.ideal, self.nadir))
        self.interaction_priority: str = "not_required"

    def iterate(self, preference: ReferencePointPreference = None):
        if preference.response is None:
            # Filler
            pass
        elif preference.response is not None:
            self.preference_point = preference.response
            self.improvement_direction = self.preference_point - self.current_point
            self.improvement_direction = self.improvement_direction / np.linalg.norm(
                self.improvement_direction
            )
            self.interaction_priority: str = "not_required"
        #  Actual step calculation
        cos_theta = np.dot(self.improvement_direction, self.step_size) / (
            np.linalg.norm(self.step_size)
        )
        step = self.improvement_direction * np.linalg.norm(self.step_size) / cos_theta
        #  Taking the step forward
        current_point = self.current_point + step
        #  Finding non-dominated points that are still achievable
        achievable_ids = np.nonzero(
            (self.non_dominated_known <= current_point).all(axis=1)
        )[0]
        if len(achievable_ids) == 0:
            self.interaction_priority = "required"
        #  Finding achievable ranges
        else:
            self.current_point = current_point
            self.currently_achievable = achievable_ids
            achievable = self.non_dominated_known[self.currently_achievable]
            lower_bounds = achievable.min(axis=0)
            upper_bounds = achievable.max(axis=0)
            self.current_points_list.append(current_point)
            self.achievable_ranges = np.vstack((lower_bounds, upper_bounds))
            self.steps_taken += 1
        return self.requests()

    def requests(self):
        return (
            self.request_ranges_plot(),
            self.request_solutions_plot(),
            self.request_preferences(),
        )

    def request_ranges_plot(self):
        objective_names = [f"X{i + 1}" for i in range(self.achievable_ranges.shape[1])]
        data = pd.DataFrame(
            self.achievable_ranges,
            index=["lower_bound", "upper_bound"],
            columns=objective_names,
        )
        dimensions_data = pd.DataFrame(
            np.vstack((np.ones_like(self.ideal), self.ideal, self.nadir)),
            index=["minimize", "ideal", "nadir"],
            columns=objective_names,
        )
        request = SimplePlotRequest(
            data=data, dimensions_data=dimensions_data, message="blah"
        )
        request.content["steps_taken"] = self.steps_taken
        request.content["current_point"] = pd.DataFrame(
            [self.current_point], columns=objective_names
        )
        request.content["total_steps"] = self.num_steps
        request.content["preference"] = pd.DataFrame(
            [self.preference_point], columns=objective_names
        )
        return request

    def request_preferences(self):
        objective_names = [
            f"X{i + 1}" for i in range(self.achievable_ranges.shape[1])
        ]
        data = pd.DataFrame(
            np.vstack((np.ones_like(self.ideal), self.achievable_ranges)),
            index=["minimize", "ideal", "nadir"],
            columns=objective_names,
        )
        message = (
            "Provide a new reference point between the achievable ideal "
            "and current nadir point"
        )
        return ReferencePointPreference(
            dimensions_data=data,
            message=message,
            interaction_priority=self.interaction_priority,
        )

    def request_solutions_plot(self):
        objective_names = [f"X{i + 1}" for i in range(self.achievable_ranges.shape[1])]
        data = pd.DataFrame(
            self.non_dominated_known, columns=objective_names
        )
        dimensions_data = pd.DataFrame(
            np.vstack((np.ones_like(self.ideal), self.ideal, self.nadir)),
            index=["minimize", "ideal", "nadir"],
            columns=objective_names,
        )
        request =  SimplePlotRequest(
            data=data, dimensions_data=dimensions_data, message="blah"
        )
        request.content["achievable_ids"] = self.currently_achievable
        request.content["current_point"] = pd.DataFrame(
            [self.current_point], columns=objective_names
        )
        request.content["preference"] = pd.DataFrame(
            [self.preference_point], columns=objective_names
        )
        return request
