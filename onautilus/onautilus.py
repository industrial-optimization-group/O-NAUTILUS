import numpy as np
import pandas as pd
from pygmo import fast_non_dominated_sorting as nds
from typing import List
from desdeo_tools.interaction.request import ReferencePointPreference, SimplePlotRequest
from random import randint
from desdeo_tools.scalarization.ASF import PointMethodASF as asf


# TODO: Name objectives in the requests
class ONAUTILUS:
    def __init__(
        self,
        known_data: np.ndarray,
        optimistic_data: np.ndarray,
        objective_names: List[str],
        num_steps: int = 10,
        max_multiplier: [List] = None,
    ):
        self.known_data = known_data * max_multiplier
        self.optimistic_data = optimistic_data * max_multiplier
        self.objective_names = objective_names
        self.num_steps = num_steps
        self.max_multiplier = max_multiplier
        # TODO THIS ASSUMES MINIMIZATION. FIX
        self.ideal_known = self.known_data.min(axis=0)
        self.non_dominated_known = self.known_data[nds(self.known_data)[0][0]]
        self.nadir_known = self.non_dominated_known.max(axis=0)

        self.ideal_optimistic = self.optimistic_data.min(axis=0)
        self.non_dominated_optimistic = self.optimistic_data[
            nds(self.optimistic_data)[0][0]
        ]
        self.nadir_optimistic = self.non_dominated_optimistic.max(axis=0)

        self.ideal = np.min((self.ideal_known, self.ideal_optimistic), axis=0)
        self.nadir = np.max((self.nadir_known, self.nadir_optimistic), axis=0)
        self.steps_taken = 0

        self.nadir_to_ideal = self.ideal - self.nadir
        self.step_size = self.nadir_to_ideal / num_steps
        # self.preference_point = self.ideal
        self.preference_point = None
        self.current_point = self.nadir
        self.current_points_list = self.nadir.reshape(-1, 1).T
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
            self.preference_point = preference.response.values[0] * self.max_multiplier
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
            self.current_points_list = np.vstack(
                (self.current_points_list, current_point)
            )

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
            self.request_long_data(),
            self.request_preferences(),
        )

    def request_ranges_plot(self):
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
            columns=self.objective_names,
        )
        data = data * self.max_multiplier
        dimensions_data = pd.DataFrame(
            index=["minimize", "ideal", "nadir"], columns=self.objective_names
        )
        dimensions_data.loc["minimize"] = self.max_multiplier
        dimensions_data.loc["ideal"] = self.ideal * self.max_multiplier
        dimensions_data.loc["nadir"] = self.nadir * self.max_multiplier
        request = SimplePlotRequest(
            data=data, dimensions_data=dimensions_data, message="blah"
        )
        request.content["steps_taken"] = self.steps_taken
        request.content["current_point"] = pd.DataFrame(
            [self.current_point], columns=self.objective_names
        )
        request.content["total_steps"] = self.num_steps
        if self.preference_point is not None:
            request.content["preference"] = pd.DataFrame(
                [self.preference_point], columns=self.objective_names
            )
        else:
            # TODO *self.current_point????? Look into other requests as well
            request.content["preference"] = pd.DataFrame(
                [self.current_point * self.max_multiplier], columns=self.objective_names
            )
        return request

    def request_preferences(self):
        dimensions_data = pd.DataFrame(
            index=["minimize", "ideal", "nadir"], columns=self.objective_names
        )
        dimensions_data.loc["minimize"] = self.max_multiplier
        dimensions_data.loc["ideal"] = self.ideal * self.max_multiplier
        dimensions_data.loc["nadir"] = self.current_point * self.max_multiplier
        message = (
            "Provide a new reference point between the achievable ideal "
            "and current iteration point"
        )
        self.request_id = randint(1e20, 1e21 - 1)
        return ReferencePointPreference(
            dimensions_data=dimensions_data,
            message=message,
            interaction_priority=self.interaction_priority,
            request_id=self.request_id,
        )

    def request_solutions_plot(self):
        data = (
            pd.DataFrame(self.non_dominated_known, columns=self.objective_names)
            * self.max_multiplier
        )
        opt_data = (
            pd.DataFrame(self.non_dominated_optimistic, columns=self.objective_names)
            * self.max_multiplier
        )

        dimensions_data = pd.DataFrame(
            index=["minimize", "ideal", "nadir"], columns=self.objective_names
        )
        dimensions_data.loc["minimize"] = self.max_multiplier
        dimensions_data.loc["ideal"] = self.ideal * self.max_multiplier
        dimensions_data.loc["nadir"] = self.nadir * self.max_multiplier

        request = SimplePlotRequest(
            data=data, dimensions_data=dimensions_data, message="blah"
        )

        request.content["achievable_ids"] = self.currently_achievable_known
        request.content["optimistic_data"] = opt_data
        request.content["achievable_ids_opt"] = self.currently_achievable_optimistic
        request.content["current_point"] = pd.DataFrame(
            [self.current_point * self.max_multiplier], columns=self.objective_names
        )
        request.content["current_points_list"] = pd.DataFrame(
            self.current_points_list, columns=self.objective_names
        )
        if self.preference_point is not None:
            request.content["preference"] = pd.DataFrame(
                [self.preference_point * self.max_multiplier],
                columns=self.objective_names,
            )
        else:
            request.content["preference"] = pd.DataFrame(
                [self.ideal * self.max_multiplier], columns=self.objective_names
            )
        return request

    def continue_optimization(self) -> bool:
        # TODO chech if the following is correct (+- 1)
        return self.steps_taken < self.num_steps

    def request_long_data(self):
        data = (
            pd.DataFrame(
                self.non_dominated_known[self.currently_achievable_known],
                columns=self.objective_names,
            )
            * self.max_multiplier
        )
        opt_data = (
            pd.DataFrame(
                self.non_dominated_optimistic[self.currently_achievable_optimistic],
                columns=self.objective_names,
            )
            * self.max_multiplier
        )
        ideal = pd.DataFrame(
            [self.ideal * self.max_multiplier], columns=self.objective_names
        )
        nadir = pd.DataFrame(
            [self.nadir * self.max_multiplier], columns=self.objective_names
        )
        if self.preference_point is None:
            preference = self.ideal
        else:
            preference = self.preference_point
        preference = pd.DataFrame(
            [preference * self.max_multiplier], columns=self.objective_names
        )
        current = pd.DataFrame(
            [self.current_point * self.max_multiplier], columns=self.objective_names
        )

        for df, source in (
            (data, "Known front"),
            (opt_data, "Optimistic front"),
            (ideal, "Ideal point"),
            (nadir, "Nadir point"),
            (preference, "Preference point"),
            (current, "Current point"),
        ):
            df["Source"] = source
        return pd.concat([data, opt_data, ideal, nadir, preference, current])


class ONAUTILUS2:
    def __init__(
        self,
        data_known: np.ndarray,
        data_optimistic: np.ndarray,
        objective_names: List[str],
        minimize: List[bool] = None,
    ):
        """The O-NAUTILUS algorithm

        Parameters
        ----------
        data_known : np.ndarray
            All Data HAS to be non-dominated
        data_optimistic : np.ndarray
            All Data HAS to be non-dominated
        objective_names : List[str]
            List of objective names in the order they appear in the previous two arrays.
        minimize : List[bool], optional
            [description], by default None

        Raises
        ------
        NotImplementedError
            [description]
        """
        self._data_known = data_known
        self._data_optimistic = data_optimistic
        self._objective_names = objective_names

        self._fitness_known = None
        self._fitness_optimistic = None
        self._ideal_fitness_known = None
        self._nadir_fitness_known = None
        self._ideal_fitness_optimistic = None
        self._nadir_fitness_optimistic = None
        self._ideal_fitness = None
        self._nadir_fitness = None

        self._known_scalarized_value = None
        self._optimistic_scalarised_value = None

        self.source_points: List = None
        self.destination_points: List = None

        if minimize is None:
            self._minimize = np.ones_like(self._objective_names)
        else:
            raise NotImplementedError()
        self.calculate_fitness()

    def calculate_fitness(self):
        self._fitness_known = self._data_known * self._minimize
        self._fitness_optimistic = self._data_optimistic * self._minimize

        self._ideal_fitness_known = self._fitness_known.min(axis=0)
        self._nadir_fitness_known = self._fitness_known.max(axis=0)

        self._ideal_fitness_optimistic = self._fitness_optimistic.min(axis=0)
        self._nadir_fitness_optimistic = self._fitness_optimistic.max(axis=0)

        self.ideal = np.min(
            (self._ideal_fitness_known, self._ideal_fitness_optimistic), axis=0
        )
        self.nadir = np.max(
            (self._nadir_fitness_known, self._nadir_fitness_optimistic), axis=0
        )
