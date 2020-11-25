from mesa import Agent
import numpy as np
import random
import uuid


class Position(Agent):
    """
    Initialise positions, which can be occupied by both vacancies and actors. Since positions ultimately inherit from
    MESA's agent class but are not built to move, positions are "inert" agents.

    type = str, always "position"

    occupant = dict, "id" is the unique ID of the position's occupant
                     "type" is the type of the actor, "vacancy" or "actor"
                     "actor moved in" is "True" if an actor moved into this position during this actor-step, False
                                      otherwise. At each new actor-step this value resets to "False"

    log = list, IDs of vacancies/actors that have occupied this position, in the order of appearance/occupancy

    NB: position unique ID's are of the format "level-spot ID", so "1-32" means "level 1, position 32"

    """
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.type = "position"
        self.occupant = {"id": '', "type": '', "actor moved in": False}  # the ID and type of the current occupant
        self.log = []


class Mover(Agent):
    """
    Superclass for mobile agents, i.e. vacancies and actors

    position = the unique ID of the position currently occupied by the mover

    log = list; IDs of positions occupied, in the order of appearance/occupancy

    _next_state = parameter only used for internal cookery, indicates the next state where the mover will go
                  e.g. retirement, level 3

    """
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.type = ''
        self.position = ''
        self.log = []

    def retire(self, diss_outside_agent):
        """
        When an actor retires they call in a vacancy from the outside to fill their spot; when a vacancy retires, it
        calls in an actor from the outside to call its spot.

        Given such a dissimilar, outside agent, this function puts that agent in your spot and puts you its spot
        (which for outside movers is the "empty" position). You are then inserted into the pool of retired, similar
        agents,

        NB: this function assumes, but does not enforce, that diss_outside_agent is a vacancy if the retiring self
            is an actor, and an actor if the retiring self is a vacancy

        :param diss_outside_agent: an Mover-class object
        """
        self.swap(diss_outside_agent)  # put the new moving agent into your spot
        self.model.schedule.add(diss_outside_agent)  # add new mover to scheduler
        self.model.schedule.remove(self)  # take yourself out of schedule
        self.model.retirees[self.type].append(self)  # put yourself in the list of retirees

    def swap(self, diss_agent):
        """
        Swap positions with another mover, i.e. you go in its position and it goes in yours.

        NB: this function assumes, but does not enforce, that diss_agent is a vacancy if self is an actor,
            and a vacancy if self is a vacancy

        :param diss_agent: an Entity-class object
        """
        new_position = diss_agent.position  # mark which position you're going into
        diss_agent.position = self.position  # put swapee in your position
        diss_agent.log.append(diss_agent.position)  # update swapee's log

        # update your old position's occupant information with the swapee's details
        self.model.positions[self.position].occupant = {"id": diss_agent.unique_id, "type": diss_agent.type}
        # if the agent going into the old position is an actor, update the according variable
        if diss_agent.type == "actor":
            self.model.positions[self.position].occupant["actor moved in"] = True

        self.position = new_position  # take your new position
        # if you're not retiring
        if self.position != '':
            # update your log
            self.log.append(self.position)
            # update your new position's occupant information with your details
            self.model.positions[self.position].occupant = {"id": self.unique_id, "type": self.type}

    def unmoving_update_log(self):
        """If you're not moving in this turn, update your log with the positions from the last turn."""
        self.log.append(self.log[-1])


class Actor(Mover):
    """
    Class of actors, i.e. those mover agents that occupy positions and retire.

    gender = str, "m" or "f"

    moved_this_turn = bool, "True" if the actor moves this turn (either they retire or they're moved around by a
                      vacancy, and "False" otherwise
    """
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.type = "actor"
        self.gender = ''

    def step(self):
        """
        Actors only have the power to retire. They retire automatically after a career that is thirty actor-steps long,
        where "actor step" is interpreted in natural terms (e.g. years).

        If an actor's career is NOT at least thirty years long, then the actor retires according to the per-level actor
        retirement probabilities supplied by the model instantiation. These probabilities are given by the model as a
        1xN vector, so we interpret e.g. [0.05, 0.13, 0.01] as "in any given actor-step, there is a 0.05 probability
        that any given actor in level 1 will retire, a 0.13 probability that any given actor in level 2 will retire,
        and a 0.01 probability that any given actor in level 3 will retire.

        If retiring, call an outside vacancy to take your place.

        NB: authors only move every every fifty-third step (this number chosen since it's a prime)
        """
        if self.model.schedule.steps % self.model.vac_mov_period == 0:
            next_step = ''
            # retire if own career is length 30, i.e. impose retirement after thirty year career
            if len(self.log) > 30:  #
                next_step = "retire"
            else:
                current_lvl = int(self.position.split("-")[0])  # recall that position IDs are of the form "level-spot"
                if bool(np.random.binomial(1, self.model.act_ret_probs[current_lvl-1])):  # -1 since Python 0-indexes
                    next_step = "retire"

            if next_step == "retire":
                outside_vacancy = Vacancy(uuid.uuid4(), self.model)
                self.retire(outside_vacancy)
            else:
                self.unmoving_update_log()


class Vacancy(Mover):
    """Class of vacancies, i.e. those mover agents that occupy positions, retire, and swap positions with actors"""

    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.type = "vacancy"

    @staticmethod
    def get_next_level(transition_probability_vector):
        """
        Say you have a vector of state transition probabilities, e.g. [0.2, 0.2, 0.6], which reads "there is a 0.2
        probability of going to State 1, a 0.2 probability of going to State 2, and a 0.6 probability of going to
        State 3." In other words, the index of the 1-indexed vector represents the destination state.

        This function uses that probability vector to pick the next state/level in which you'll be going.

        NB: this function only works if the transition probability vector sums to 1

        :return: the state, an int
        """
        cum_sum = np.cumsum(transition_probability_vector)
        cum_sum = np.insert(cum_sum, 0, 0)
        # throw random dart
        rd = np.random.uniform(0.0, 1.0)
        # see which state the dart hit
        m = np.asarray(cum_sum < rd).nonzero()[0]
        next_state = m[len(m) - 1]  # need -1 since we inserted a zero in the cumulative sum above
        return next_state

    def get_next_position_info(self, next_level):
        """
        Given a level to which we'll be moving, randomly pick a position in said level and return its ID as well as
        the ID of its current occupant.
        :param next_level: int
        """
        # get the position in the level you're going to; recall that position ID is of form "level-spot"
        positions_in_next_level = [pos for pos in self.model.positions.values()
                                   if int(pos.unique_id.split("-")[0]) == next_level]

        # randomly pick a position in said level to move to
        desired_position = random.choice(positions_in_next_level)

        # return the ID of the desired position position ID
        return {"desired position's ID": desired_position.unique_id,
                "desired position's occupant": desired_position.occupant}

    def step(self):
        """
        Vacancies can swap positions with actors already in the system, or retire and put an outside actor in their
        position.

        By convention the vacancy transition probability matrix is a NxN+1 matrix; N=number of levels and the last
        column indicates the probability of a vacancy retiring, i.e. of calling in an actor from outside the system.

        If e.g. N=3 then:

                    Level 1     Level 2     Level 3     Retire
        Level 1       0.3         0.4         0.1        0.2
        Level 2       0.1         0.4         0.4        0.1
        Level 3       0.05        0.05        0.2        0.7

        Some example of interpretation are "a vacancy in level 1 has a 0.4 probability of moving within level 1", or
        "a vacancy in level 3 has a probability of 0.1 to retire" or "a vacancy in level 3 has a probability of 0.3 to
        move to level 2".

        If we see this matrix referring to a 3-level hierarchy where "1" is the highest level, the above transition
        probability matrix depicts a system where vacancies tend to move down (thereby promoting actors up), stay in
        their level (thereby encouraging lateral actor moves) and retire (thereby recruiting outside actors).

        NB: this transition probability matrix ONLY depicts movement, i.e. there is no choice for a vacancy to stay put.

        In this model vacancies do not move in "natural time", i.e. in the actor steps. Rather, they move in artificial
        "vacancy steps." This two-step system is designed so that all vacancy steps will occur in the span of one
        actor step, i.e. so that a vacancy chain exhausts itself in one unit of natural time. The substantive purpose is
        to ensure that no positions are vacant for more than one turn, which is reasonable for instance of employment
        organisations, where a job kept empty for a year will often be abolished. If this 1-actor-step limit is too
        harsh, we can relax it and allow positions to be vacant for 2, 3, etc. years.

        Vacancies must coordinate with each other to ensure orderly movement. The rules of vacancy movement are:
        1) vacancies may only move into spots occupied by actors
        2) vacancies may never move an actor that has joined the system in their current actor step
        3) vacancies may never move an actor that has already been moved by other vacancies in their current actor step
        4) if N>1 vacancies want to move into the same actor-occupied spot, they toss an N-sided die and the winner of
           the toss gets to move into said spot. All the other ones forfeit this turn of this vacancy movement.

        NB: vacancies have 49 internal steps to move around; they omit every fiftieth step
        """
        if self.model.schedule.steps % self.model.vac_mov_period != 0:

            # get the next level where you'll be going;
            # NB: adding and substracting 1's because we 1-index levels but python zero-indexes lists
            current_level = int(self.position.split("-")[0])
            lvl_transition_probs = self.model.vac_trans_prob_mat[current_level-1]
            next_level = self.get_next_level(lvl_transition_probs) + 1
            if next_level == self.model.vac_trans_mat_num_cols:
                next_level = "retire"

            # handle retiring vacancies
            if next_level == "retire":
                outside_actor = Actor(uuid.uuid4(), self.model)
                current_step = self.model.schedule.steps

                # assign actor's gender according to the model-provided probability that new recruits are female
                if random.uniform(0.0, 1.0) <= self.model.percent_fem_entry_per_step[current_step]:
                    outside_actor.gender = "f"
                else:
                    outside_actor.gender = "m"

                self.retire(outside_actor)
                return

            else:  # handle vacancies that want to move an actor

                # select up to five desired positions to which you (the vacancy) would like to move
                # NB: these are NOT uniques, though hopefully they mostly will be since get_next_position_info
                # incorporates a random selector
                desired_positions = [self.get_next_position_info(next_level) for i in range(5)]

                # for each positions, see get information on it and its current occupant
                for des_pos in desired_positions:
                    des_pos_occ = des_pos["desired position's occupant"]

                    # if the desired position is either already occupied by a vacancy or has already
                    # featured actor movement in this actor-step, go to the next one
                    if des_pos_occ["type"] == "vacancy" or des_pos_occ["actor moved in"] is True:
                        continue

                    else:
                        # if not, swap positions with the actor occupying said position
                        for actr in self.model.schedule.agent_buffer():
                            if des_pos_occ["id"] == actr.unique_id:
                                self.swap(actr)
                                return
