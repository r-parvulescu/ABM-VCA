"""
TO GET

CHAIN MEASURES

Transition Matrices

Average Across All Years
- observed vacancy transition matrix
- difference between observed vacancy transition matrix and input transition matrix


ACTOR MEASURES

across all years
- observed actor transition matrix
- difference between observed actor transition matrix and hypothetical input (actor) transition matrix


"""

import statistics
import numpy as np


# POSITION MEASURES

def get_actor_turnover_rate(model):
    """
    Get the actor turnover rate, averaged across all positions in a certain level. The per-position rate is simply the
    number of actors who have occupied that position, divided by the total number of actor steps. A value one means that
    there's a new actor each step, while the more the measure approaches zero the less position there is. We then
    average these measures across all positions in a certain level.
    """
    turnover_per_level = {i: [] for i in range(1, model.num_levels + 1)}
    for pos_id, position in model.positions.items():
        lvl = int(pos_id.split("-")[0])
        number_of_steps, number_of_actors = len(position.log), len(set(position.log))
        turnover_rate = float(number_of_actors) / float(number_of_steps)
        turnover_per_level[lvl].append(turnover_rate)
    # average each list
    for k, v in turnover_per_level.items():
        if v:  # ignore empty lists
            # if we're on the first five steps replace the value with None. That value always starts at 1 and drops very
            # quickly towards zerp, this behaviour is automatic and thus uninteresting and makes for over-tall graphs
            if model.schedule.steps < 6:
                turnover_per_level[k] = None
            else:
                turnover_per_level[k] = statistics.mean(v)
    return turnover_per_level


# VACANCY MEASURES


def get_average_vacancy_chain_length(model):
    """
    Get the average length of vacancy chains of for all vacancies that played themselves out during this actor-step;
    pooled and disaggregated by the level in which the vacancy entered the system.
    """
    return get_average_log_length(model, "vacancy")


def get_count_vacancies_still_in_system(model):
    """
    We assume that all vacancies work themselves out between actors steps. This function checks that assumption, by
    counting the number of vacancies in the system at any given actor-step (by which point they should have all worked
    their way out). Pooled and disaggregated by level.
    """
    vac_per_level = {i: 0 for i in range(1, model.num_levels + 1)}
    for agent in model.schedule.agents:
        if agent.type == "vacancy":
            current_level = int(agent.position.split('-')[0])
            vac_per_level[current_level] += 1
    return vac_per_level


def get_count_vacancies_per_step(model):
    """
    Get the number of vacancies that made their way through the system in this actor step; pooled and disaggregated
    by the level in which they started.
    """
    vac_per_level = {i: 0 for i in range(1, model.num_levels + 1)}
    for vacancy in model.retirees["vacancy"]:
        entry_level = int(vacancy.log[0].split("-")[0])
        vac_per_level[entry_level] += 1
    return vac_per_level


def get_net_vacancy_effects(model):
    """
    Using the vacancy move benefit-deficit matrix, calculate the per-actor-step net impact of all vacancy chains
    initiated in that step. If e.g. we consider the benefits accruing to a change in housing (in some abstract utils)
    then we read a matrix like

    "vacancy_benefit_deficit_matrix": [[1, 2, 3, 4],
                                        [-1, 3, 4, 5],
                                        [-2, -1, 5, 6]],

    to mean that when housing vacancies move down (and therefore people move up) some scale of unit "niceness" then
    they have an increase in utils, and the opposite for moving to less nice housing. This matrix also suggests
    decreasing marginal utility of housing niceness, since people lower down the niceness hierarchy gain more from
    moving up or having a new unit built (presumably pulling people out of homelessness) than people moving within or
    between higher levels.

    Thus function calculates net benefits of vacancy movements, disaggregated by the level in which the vacancy started.

    NB: I prefer to calculate this post-hoc rather than build it into the vacancy-agents as an updating function since
        we don't always want to collect these data and I don't want to encumber the core model code.
    """
    effects_per_level = {i: 0 for i in range(1, model.num_levels + 1)}
    for vacancy in model.retirees["vacancy"]:
        entry_level = int(vacancy.log[0].split("-")[0])
        # calcuate unit changes associated with vacancy mobility, by consulting the logs of vacancies initiated in that
        # actor-step
        for idx, spot in enumerate(vacancy.log):
            if len(vacancy.log) > 1 and idx < len(vacancy.log)-1:
                start_lvl, end_lvl = int(spot.split("-")[0]), int(vacancy.log[idx+1].split("-")[0])
                # need to substrat 1 since the Python vac_ben_def_mat is a list of lists with zero indexing
                effects_per_level[entry_level] += model.vac_ben_def_mat[start_lvl-1][end_lvl-1]
    return effects_per_level


def markov_predicted_chain_length(vacancy_transition_probability_matrix):
    """
    Test if the per-level chain length predicted by a first-order, discrete, embedded Markov-chain with absorbing
    states lines up with your observed average chain length.

    NB: we calculate predicted chain length, disaggregated by the level in which the chain starts, by first creating
    the matrix N = inv(I - Q), where Q is the square submatrix of the vacancy transition matrix excluding absorbing
    states (i.e. the retirement column), I is the identity matrix of similar dimensions, and "inv" signifies inversion.
    To get predicted chain length according to the vacancy's starting level we take Nx1, where 1 a vector of ones.

    :param vacancy_transition_probability_matrix:
    """
    # get the square transition probability submatrix, ignoring the absorbing states, by convention the last column
    trans_prob_submatrix = np.array([row[:-1] for row in vacancy_transition_probability_matrix])
    number_of_levels = len(trans_prob_submatrix)
    identity = np.identity(number_of_levels)
    identity_minus_trans_prob_submatrix = np.subtract(identity, trans_prob_submatrix)
    # make the N matrix
    n_matrix = np.linalg.inv(identity_minus_trans_prob_submatrix)
    vector_of_ones = np.ones(number_of_levels)
    # get the predicted chain lengths, or in White's (1970) jargon, the "multiplier"
    pred_chain_lengths = np.matmul(n_matrix, vector_of_ones)
    return {i+1: pred_chain_lengths[i] for i in range(number_of_levels)}


# ACTOR MEASURES


def get_actor_count(model):
    """Get the number of actors curently in the system, pooled and disaggregated by level."""
    act_per_level = {i: 0 for i in range(1, model.num_levels + 1)}
    for agent in model.schedule.agents:
        if agent.type == "actor":
            current_level = int(agent.position.split('-')[0])
            act_per_level[current_level] += 1
    return act_per_level


def get_average_actor_career_length(model):
    """Get the average length of the career lengths of all actors in the system' pooled and disaggregated by level."""
    return get_average_log_length(model, "actor")


def get_time_to_promotion_from_last_level(model):
    """
    Get the average time to promotion for all those promoted in the last actor-step. Disaggregated by the level from
    which you were promoted. NB: does not account for the promotion destination, just where you left from.
    """
    # NB: can't promote from the last level, so we ignore it
    time_to_prom = {i: [] for i in range(1, model.num_levels)}

    for agent in model.schedule.agents:
        if agent.type == "actor":
            log = list(filter(None, agent.log))  # get rid of empty years at the beginning
            if len(log) > 1:  # can only look at those who had the chance to move
                # check to see if they were promoted last turn)
                if int(log[-2].split("-")[0]) < int(log[-1].split("-")[0]):
                    # check how long that last spell was; reverse the log and starting from the second position look for
                    # the first non-matching value; if you don't find a difference then you've been in the same position
                    # since the beginning of the career except the last step, i.e. career length - 1
                    last_lvl = int(log[-2].split("-")[0])
                    spell_len = next((idx for idx, pos in enumerate(log[::-1][1:])
                                     if int(pos.split("-")[0]) != last_lvl), len(log)-1)

                    # update the promotion dict with the relevant value
                    time_to_prom[last_lvl].append(spell_len)

    # average each list
    if time_to_prom:  # ignore empty dicts
        for k, v in time_to_prom.items():
            if v:  # ignore empty lists
                time_to_prom[k] = statistics.mean(v)

    return time_to_prom


def get_time_to_retirement_from_last_level(model):
    """
    For all actors that retired in the last turn, get the average amount of time that they were in the pre-retirement
    level before they finally lef the system. Disaggregated by level.
    """

    retire_time_per_level = {i: [] for i in range(1, model.num_levels + 1)}
    for actor in model.retirees["actor"]:
        log = list(filter(None, actor.log))  # get rid of empty years at the beginning
        last_lvl = int(log[-1].split("-")[0])
        # check how long that last spell was; reverse the log and look for the first non-matching value; if you don't
        # find a difference then you retired out of your starting position, so last spell = career length
        last_spell_len = next((idx for idx, pos in enumerate(log[::-1]) if int(pos.split("-")[0]) != last_lvl),
                              len(log))
        retire_time_per_level[last_lvl].append(last_spell_len)

    # average each list
    if retire_time_per_level:  # ignore empty dicts
        for k, v in retire_time_per_level.items():
            if v:  # ignore empty lists
                retire_time_per_level[k] = statistics.mean(v)

    return retire_time_per_level


def get_percent_female_actors(model):
    """See what percentage of all actors currently in the system are female; pooled and disaggregated by level."""

    # initialise a dict of dicts, where first-order keys are levels, and sub-dicts contain counts of men and women
    # in the respective level. Also add a sub-dict that pools counts across all levels.
    gend_per_level = {i: {"m": 0., "f": 0.} for i in range(1, model.num_levels + 1)}
    gend_per_level.update({"pooled": {"m": 0., "f": 0.}})

    # fill the dicts with observed counts
    for agent in model.schedule.agents:
        if agent.type == "actor":
            current_level = int(agent.position.split('-')[0])
            if agent.gender == "m":
                gend_per_level[current_level]["m"] += 1.
                gend_per_level["pooled"]["m"] += 1.
            if agent.gender == "f":
                gend_per_level[current_level]["f"] += 1.
                gend_per_level["pooled"]["f"] += 1.

    # turn each count sub-dict into percent female
    for k, v in gend_per_level.items():
        total = gend_per_level[k]["f"] + gend_per_level[k]["m"]
        if total == 0:
            gend_per_level[k] = 0
        else:
            gend_per_level[k] = round(gend_per_level[k]["f"] / total, 2) * 100.

    return gend_per_level


def get_percent_actors_from_before_shock(model):
    """
    See what percentage of actors currently in the system entered the system before the system shock; pooled and
    disaggregated by level.
    """

    # get the current actor-step that we're on, as well as the step at which the shock occurred
    current_step = model.schedule.steps
    steps_since_first_shocks = current_step - model.shock_step  # NB: negative if evaluated before the shock

    pre_shock_per_level = {i: {"total_actors": 0., "actors_from_b4_shock": 0.} for i in range(1, model.num_levels + 1)}
    pre_shock_per_level.update({"pooled": {"total_actors": 0., "actors_from_b4_shock": 0.}})

    # fill the dicts with observed counts
    for agent in model.schedule.agents:
        if agent.type == "actor":
            current_level = int(agent.position.split('-')[0])
            pre_shock_per_level[current_level]["total_actors"] += 1.
            if len(agent.log) > steps_since_first_shocks:
                pre_shock_per_level[current_level]["actors_from_b4_shock"] += 1.

    # turn each count sub-dict into percent from before shock
    for k, v in pre_shock_per_level.items():
        if pre_shock_per_level[k]["total_actors"] == 0:
            pre_shock_per_level[k] = 0
        else:
            pre_shock_per_level[k] = round(pre_shock_per_level[k]["actors_from_b4_shock"] /
                                           pre_shock_per_level[k]["total_actors"], 2) * 100.
    return pre_shock_per_level


# FOR BOTH VACANCIES AND ACTORS

def get_average_log_length(model, agent_type):
    """
    Given an agent type, find the average log length of agents in the relevant agent set, both pooled across all those
    agents and disaggregated by level. For actors, this means average career length of actors currently in the system,
    by their current level of occupancy. For vacancies, this means average chain length of retired vacancies (i.e. out
    of the system), by the level at which they entered the system.
    """

    # initialise a dict where keys are levels and values are lists to be filled with log lengths;
    # also add a list that pools lengths across all levels
    log_lengths_per_level = {i: [] for i in range(1, model.num_levels + 1)}

    # fix the agent set depending on the agent type
    agent_set = model.retirees["vacancy"] if agent_type == "vacancy" else model.schedule.agents

    # fill in the log length lists
    for agent in agent_set:
        if agent.type == agent_type:
            level = agent.log[0].split("-")[0] if agent_type == "vacancy" else agent.position.split('-')[0]
            log_length = len(agent.log)
            log_lengths_per_level[int(level)].append(log_length)

    # average each list
    for k, v in log_lengths_per_level.items():
        if v:  # ignore empty lists, e.g. list of vacancy moves at the first actor step, when no vacancy has moved yet
            log_lengths_per_level[k] = statistics.mean(v)

    return log_lengths_per_level


def get_agent_sets_sizes(model):
    """
    Return the number of positions, number of agents, and number of vacancies associated with one actor-step; this
    is a sanity check.
    """
    return {"actor": len([1 for agent in model.schedule.agents if agent.type == "actor"]),
            "position": len(model.retirees["vacancy"]),
            "vacancy": len(model.positions.keys())}
