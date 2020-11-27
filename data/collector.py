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
