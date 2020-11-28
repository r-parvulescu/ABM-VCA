"""
This script runs an agent-based model implementation of vacancy chain analysis, i.e. the ABM-VCA.
Author of Python Code:
    Pârvulescu, Radu Andrei (2020)
    rap348@cornell.edu
"""

from mesa import Model, time
from mesa.datacollection import DataCollector
from vc_abm.entities import Actor, Position, Vacancy
import numpy as np
import uuid
import random


class VacancyChainAgentBasedModel(Model):
    """"""

    def __init__(self, positions_per_level, actor_retire_probs, vacancy_trans_prob_matrix, firing_schedule,
                 growth_orders, start_percent_female, percent_female_entry_per_step, vacancy_benefit_deficit_matrix,
                 data_collector, shock_step=0, vacancy_move_period=0):
        """
        :param positions_per_level: list of ints, of positions per level
                                    e.g. [10,20,30] == 10 positions in level 1, 20 in level 2, 30 in level 3

        :param actor_retire_probs: list of ints, the per-actor, per actor-step probability of retirement, for each
                                   level. E.g. with [0.3, 0,1, 0.4], each actor in level 1 has a probability of
                                   0,3 of retiring, each actor-step

        :param vacancy_trans_prob_matrix: list of lists, where each sublist is a row of a transition probability
                                          matrix. The vacancy transition probability matrix is a NxN+1 matrix;
                                          N=number of levels and the last column indicates the probability of a vacancy
                                          retiring, i.e. of calling in an actor from outside the system. If e.g. N=3:

                                                      Level 1     Level 2     Level 3     Retire
                                          Level 1    [[0.3,         0.4,        0.1,        0.2],
                                          Level 2     [0.1,         0.4,        0.4,        0.1],
                                          Level 3     [0.05,        0.05,       0.2,        0.7]]

                                          Some example of interpretations are "a vacancy in level 1 has a 0.4
                                          probability of moving within level 1", or "a vacancy in level 3 has a
                                          probability of 0.1 to retire" or "a vacancy in level 3 has a probability of
                                          0.3 to move to level 2".
                                          NB: rows must sum to 1.
                                          NB: this code assumes that higher rows represent higher levels in the
                                          hierarchy, so level 1 > 2 > 3
                                          Further details on vacancy movements can be found in entity.vacancy.step

        :param firing_schedule: dict, indicating how actors' retirement probabilities should be changed at the specified
                                actor-steps. So, e.g. {"steps": {5, 6, 7}, "ret probs": [0.4, 0.4, 0.6]}
                                means that at each of actor-steps five, six, and seven, the retirement probability of
                                each actor in level 1 will be 0.4, the retirement probability of each actor in level 2
                                will be 0.4, and the retirement probability of each actor in level 3 will be 0.6. Before
                                step five and after step seven we use the actor retirement probabilities given by
                                actor_retire_probs

        :param growth_orders: dict, indicating how many positions should be added to different levels, at the specified
                              actor-steps. So, e.g. {"steps": {7, 8}, "extra positions": [10, 50, 150]} means
                              that at each of actor-steps seven and eight we will add ten new positions in level one,
                              fifty new positions in level two, and one hundred and fifty new positions in level three.
                              Before step seven and after step eight we do not change the size of the system.

        :param start_percent_female: float, gender ratio with which we initialise actors at the beginning of the model,
                                     e.g. 0.6 means that at model initialisation sixty percent of all actors are female

        :param percent_female_entry_per_step: dict, indicating the probability that an actor called from the outside by
                                              a vacancy retirement will be female, at the designated actor steps. So,
                                              e.g. {20: 0.6, 21: 0.7, 22: 0.7} means that at actor step twenty each
                                              newly-called actor has a probability of 0.6 of being female, while at each
                                              of actor steps twenty-one and twenty-two each newly-called actor has a
                                              probability of 0.7 of being female.
                                              NB: this dict must be defined for ALL actor-steps of the model

        :param vacancy_benefit_deficit_matrix: changes in some unit associated with vacancy movements. E.g. this matrix

                                               [[1, 2, 3, 4],
                                                [-1, 3, 4, 5],
                                                [-2, -1, 5, 6]]

                                               refers to changes in a family's utility as it moves up or down the
                                               "niceness" scale of housing: moves up increase utility, moves down
                                               decrease it.

        :param data_collector: dict, indicating the data collector functions that harvest measures from the model.
                               Has form {"title_of_data_collector":name_of_collector_function"}.
                               NB: data collector functions live in collectors.py

        :param shock_step: int, an actor-step at which a shock of particular note occurs, e.g. a major system expansion.
                           By default shock_step == 0.

        :param vacancy_move_period: int, the number of model steps that we give vacancies to work their way out of the
                                    system. The classic, simplest formulation of vacancy chain analysis assumed that
                                    vacancies make their way out of the system within some natural time unit, like a
                                    year, which here is operationalised as the "actor-step". In model terms, the
                                    vacancy_move_period is the number of model steps BETWEEN actor steps; only vacancies
                                    are allowed to move in these "vacancy steps" and only actors can move during "actor
                                    steps". By default this is 0, i.e. vacancies move as often as actors.
                                    NB: adjust this number by consulting collectors.get_count_vacancies_still_in_system

        """
        super().__init__()
        # set parameters
        self.num_levels, self.positions_per_level = len(positions_per_level), positions_per_level
        self.firing_schedule, self.growth_orders = firing_schedule, growth_orders
        self.start_percent_fem, self.percent_fem_entry_per_step = start_percent_female, percent_female_entry_per_step
        self.act_ret_probs, self.vac_trans_prob_mat = actor_retire_probs, vacancy_trans_prob_matrix
        self.vac_trans_mat_num_cols = len(vacancy_trans_prob_matrix[0])
        self.vac_ben_def_mat = vacancy_benefit_deficit_matrix
        self.data_collector = DataCollector(model_reporters=data_collector)
        self.shock_step, self.vac_mov_period = shock_step, vacancy_move_period

        # define the scheduler
        self.schedule = time.RandomActivation(self)

        # make a container for retired moving agents, i.e. those actors and vacancies that have left the system
        self.retirees = {"actor": [], "vacancy": []}

        # initialise system: make positions and occupy them with actors; all starting positions are actor-occupied
        # NB: levels and positions are 1-indexed
        self.positions = {}
        for lvl in range(self.num_levels):
            for spot in range(self.positions_per_level[lvl]):
                self.create_position(lvl + 1, spot + 1, "actor")  # +1 in order to 1-index the positions codes

    def step(self):
        """Make vacancies and actors move."""

        # every fifty-third step, collect data for actors that are STILL in the system, and collect data for vacancies
        # that are NO LONGER in the system, i.e. that worked their way out in the previous fifty-two steps
        if self.schedule.steps % self.vac_mov_period == 0:
            self.data_collector.collect(self)

            # reset the pool of retired vacancies and actors, so that it's fresh for the next round
            self.retirees = {"actor": [], "vacancy": []}

            # reset bools indicating if position witnessed actor movement this step, so its fresh for the next round
            for pos in self.positions.values():
                pos.occupant["actor moved in"] = False

        # if there are growth orders, carry them out
        if self.schedule.steps in self.growth_orders["steps"]:
            self.grow()

        # if there are firing orders, make actors step according to them
        if self.schedule.steps in self.firing_schedule["steps"]:
            baseline_act_ret_prob = self.act_ret_probs  # save baseline actor retirement probabilities
            self.act_ret_probs = self.firing_schedule["actor retirement probs"]  # set new retirement probabilities
            self.schedule.step()  # make actors move
            self.act_ret_probs = baseline_act_ret_prob  # return to baseline actor retirement probabilities
        else:  # make agents (actors and vacancies both) step, for all other cases
            self.schedule.step()

        # update the positions' logs, post-step
        for pos_id, pos in self.positions.items():
            pos.log.append((pos.occupant["id"], pos.occupant["type"]))

    # part of step
    def grow(self):
        """Increase the number of positions. All new positions are vacant."""
        for lvl in range(1, self.num_levels + 1):
            # get the highest ID of current positions in this level so you can add new positions starting from there
            max_position_id_on_this_level = max([int(pos.split("-")[1]) for pos in self.positions
                                                 if int(pos.split("-")[0]) == lvl])
            for new_spot in range(1, self.growth_orders["extra positions"][lvl-1] + 1):  # -1 since python 0-indexes
                new_spot_id = max_position_id_on_this_level + new_spot
                self.create_position(lvl, new_spot_id, "vacancy")

    def create_position(self, lvl, spot, agent_type):
        """
        Given a level, a spot ID, create a position and fill it with the actor type defined
        :param lvl: int, a level in the vacacy systen, e.g. 2
        :param spot: int, a level-unique ID for the particular spot in said level, e.g. 4
        :param agent_type: str, "vacancy" or "actor"
        :return: None
        """
        # generate the position
        position_id = str(lvl) + '-' + str(spot)  # position ID = level-spot
        pos = Position(position_id, self)
        self.positions[position_id] = pos
        # create new moving agent and add it to the scheduler
        new_agent = Vacancy(uuid.uuid4(), self) if agent_type == "vacancy" else Actor(uuid.uuid4(), self)
        if agent_type == "actor":
            new_agent.gender = "f" if bool(np.random.binomial(1, self.start_percent_fem)) else "m"
            # to avoid clumpy cohort effects and therefore the number of burn-in steps, give each actor a random career
            # age of 10-22, since these are the equilibrium career ages per level; this way the "start" of the system
            # is smoother
            rand_age = random.choice(list(range(10, 23)))
            new_agent.log.extend(rand_age * [""])
        self.schedule.add(new_agent)
        # make that vacancy occupy the position
        new_agent.position = pos.unique_id
        pos.occupant["id"], pos.occupant["type"] = new_agent.unique_id, new_agent.type
        # update logs
        new_agent.log.append(pos.unique_id), pos.log.append(new_agent.unique_id)
