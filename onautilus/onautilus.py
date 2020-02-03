import numpy as np
import pandas as pd
from pygmo import fast_non_dominated_sorting as nds
from typing import List
from desdeo_tools.interaction.request import ReferencePointPreference, SimplePlotRequest
from random import randint

# TODO: Name objectives in the requests
class ONAUTILUS:
    def __init__(
        self, known_data: np.ndarray, optimistic_data: np.ndarray, num_steps: int = 10
    ):
        self.known_data = known_data
        self.optimistic_data = optimistic_data
        self.num_steps = num_steps
        # TODO THIS ASSUMES MINIMIZATION. FIX
        self.ideal_known = known_data.min(axis=0)
        self.non_dominated_known = known_data[nds(known_data)[0][0]]
        self.nadir_known = self.non_dominated_known.max(axis=0)

        self.ideal_optimistic = optimistic_data.min(axis=0)
        self.non_dominated_optimistic = optimistic_data[nds(optimistic_data)[0][0]]
        self.nadir_optimistic = self.non_dominated_optimistic.max(axis=0)

        self.ideal = np.min((self.ideal_known, self.ideal_optimistic), axis=0)
        self.nadir = np.max((self.nadir_known, self.nadir_optimistic), axis=0)
        self.steps_taken = 0

        self.nadir_to_ideal = self.ideal - self.nadir
        self.step_size = self.nadir_to_ideal / num_steps
        # self.preference_point = self.ideal
        self.preference_point = None
        self.current_point = self.nadir
        self.current_points_list = [self.nadir]
        self.improvement_direction = None
        self.currently_achievable_known: List = range(len(self.non_dominated_known))
        self.achievable_ranges_known = np.vstack((self.ideal_known, self.nadir_known))

        self.currently_achievable_optimistic: List = range(
            len(self.non_dominated_optimistic)
        )
        self.achievable_ranges_optimistic = np.vstack(
            (self.ideal_optimistic, self.nadir_optimistic)
        )

        self.interaction_priority: str = "required"
        self.request_id: int = 0

    def iterate(self, preference: ReferencePointPreference = None):
        if self.interaction_priority == "required":
            if preference is None:
                return self.requests()
            elif preference.response is None:
                return self.requests()
            elif preference.request_id != self.request_id:
                return self.requests()

        if preference.response is not None:
            self.preference_point = preference.response.values[0]
            self.improvement_direction = self.preference_point - self.current_point
            self.improvement_direction = self.improvement_direction / np.linalg.norm(
                self.improvement_direction
            )
            self.interaction_priority: str = "not_required"

        #  Actual step calculation
        print(self.preference_point)
        cos_theta = np.dot(self.improvement_direction, self.step_size) / (
            np.linalg.norm(self.step_size)
        )
        step = self.improvement_direction * np.linalg.norm(self.step_size) / cos_theta

        #  Taking the step forward
        current_point = self.current_point + step

        #  Finding non-dominated points that are still achievable
        achievable_ids_known = np.nonzero(
            (self.non_dominated_known <= current_point).all(axis=1)
        )[0]
        achievable_ids_optimistic = np.nonzero(
            (self.non_dominated_optimistic <= current_point).all(axis=1)
        )[0]
        if len(achievable_ids_known) == 0:
            self.interaction_priority = "required"
        #  Finding achievable ranges
        else:
            self.current_point = current_point
            self.current_points_list.append(current_point)

            self.currently_achievable_known = achievable_ids_known
            self.currently_achievable_optimistic = achievable_ids_optimistic
            achievable_known = self.non_dominated_known[self.currently_achievable_known]
            achievable_optimistic = self.non_dominated_optimistic[
                self.currently_achievable_optimistic
            ]

            lower_bounds_known = achievable_known.min(axis=0)
            upper_bounds_known = achievable_known.max(axis=0)
            lower_bounds_optimistic = achievable_optimistic.min(axis=0)
            upper_bounds_optimistic = achievable_optimistic.max(axis=0)

            self.achievable_ranges_known = np.vstack(
                (lower_bounds_known, upper_bounds_known)
            )
            self.achievable_ranges_optimistic = np.vstack(
                (lower_bounds_optimistic, upper_bounds_optimistic)
            )
            self.steps_taken += 1
        return self.requests()

    def requests(self):
        return (
            self.request_ranges_plot(),
            self.request_solutions_plot(),
            self.request_preferences(),
        )

    def request_ranges_plot(self):
        objective_names = [
            f"F{i + 1}" for i in range(self.achievable_ranges_known.shape[1])
        ]
        data = pd.DataFrame(
            np.vstack(
                (self.achievable_ranges_known, self.achievable_ranges_optimistic)
            ),
            index=[
                "lower_bound",
                "upper_bound",
                "optimistic_lower_bound",
                "optimistic_upper_bound",
            ],
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
        if self.preference_point is not None:
            request.content["preference"] = pd.DataFrame(
                [self.preference_point], columns=objective_names
            )
        else:
            request.content["preference"] = pd.DataFrame(
                self.preference_point, columns=objective_names, index=[0]
            )
        return request

    def request_preferences(self):
        objective_names = [f"F{i + 1}" for i in range(self.achievable_ranges_known.shape[1])]
        data = pd.DataFrame(
            np.vstack((np.ones_like(self.ideal), self.achievable_ranges_optimistic)),
            index=["minimize", "ideal", "nadir"],
            columns=objective_names,
        )
        message = (
            "Provide a new reference point between the achievable ideal "
            "and current nadir point"
        )
        self.request_id = randint(1e20, 1e21 - 1)
        return ReferencePointPreference(
            dimensions_data=data,
            message=message,
            interaction_priority=self.interaction_priority,
            request_id=self.request_id,
        )

    def request_solutions_plot(self):
        objective_names = [
            f"F{i + 1}" for i in range(self.achievable_ranges_known.shape[1])
        ]
        data = pd.DataFrame(self.non_dominated_known, columns=objective_names)
        opt_data = pd.DataFrame(self.non_dominated_optimistic, columns=objective_names)
        dimensions_data = pd.DataFrame(
            np.vstack((np.ones_like(self.ideal), self.ideal, self.nadir)),
            index=["minimize", "ideal", "nadir"],
            columns=objective_names,
        )
        request = SimplePlotRequest(
            data=data, dimensions_data=dimensions_data, message="blah"
        )
        request.content["achievable_ids"] = self.currently_achievable_known
        request.content["optimistic_data"] = opt_data
        request.content["achievable_ids_opt"] = self.currently_achievable_optimistic
        request.content["current_point"] = pd.DataFrame(
            [self.current_point], columns=objective_names
        )
        if self.preference_point is not None:
            request.content["preference"] = pd.DataFrame(
                [self.preference_point], columns=objective_names
            )
        else:
            request.content["preference"] = pd.DataFrame(
                self.preference_point, columns=objective_names, index=[0]
            )
        return request

    def continue_optimization(self) -> bool:
        # TODO chech if the following is correct (+- 1)
        return self.steps_taken < self.num_steps
