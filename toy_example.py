"""Example of ABM-VCA implementation using toy data."""

from vc_abm import model, batchrun
from data import collector, timeseries, convergence
from local import out_dir

total_actor_steps = 70
burn_in_steps = 40
iterations = 10

out_directory = out_dir + 'toy/'
experiment_name = "Toy Example"
level_names = {1: "Top", 2: "Middle", 3: "Bottom"}

data_collector = {"average_career_length": collector.get_average_actor_career_length,
                  "average_vacancy_chain_length": collector.get_average_vacancy_chain_length,
                  "actor_turnover_rate": collector.get_actor_turnover_rate,
                  "time_to_promotion_from_last_level": collector.get_time_to_promotion_from_last_level,
                  "time_to_retirement_from_last_level": collector.get_time_to_retirement_from_last_level,
                  "net_vacancy_effects": collector.get_net_vacancy_effects,
                  "agent_sets_sizes": collector.get_agent_sets_sizes,
                  "count_vacancies_still_in_system": collector.get_count_vacancies_still_in_system,
                  "percent_female_actors": collector.get_percent_female_actors,
                  "percent_actors_from_before_shock": collector.get_percent_actors_from_before_shock}

# Toy example; the vacancy transition probability matrix is roughly the same as reported by White for
# the 1922-1937 Methodist Church system (White 1970:125)

# start with twenty percent women, and after step 40 start increasing this percentage to fifty percent
fem_entry_per_step = {i: .2 for i in range(total_actor_steps)}
fem_entry_per_step.update({i: .5 for i in range(40, total_actor_steps + 1)})
fixed_params = {"shock_step": 40,
                "positions_per_level": [50, 100, 150],
                "vacancy_move_period": 30,
                "actor_retire_probs": [0.1, 0.075, 0.05],
                "vacancy_trans_prob_matrix": [[.5, .3, .1, .1],
                                              [.1, .4, .3, .2],
                                              [0., .1, .3, .6]],
                "vacancy_benefit_deficit_matrix": [[1, 2, 3, 4],
                                                   [-1, 3, 4, 5],
                                                   [-2, -1, 5, 6]],
                "firing_schedule": {"steps": {}, "actor retirement probs": []},
                "growth_orders": {"steps": {50, 51, 52}, "extra positions": [5, 10, 15]},
                "start_fraction_female": .2,
                "prob_female_entry_per_step": fem_entry_per_step}

if __name__ == "__main__":
    variable_params = None
    fixed_params["data_collector"] = data_collector
    model_reporters = {"Data Collector": lambda m: m.data_collector}
    simulation = model.VacancyChainAgentBasedModel
    total_simulation_steps = total_actor_steps * fixed_params["vacancy_move_period"] + 1

    # run without firing schedule
    b_runs = [batchrun.get_batchrun(simulation, variable_params, fixed_params, iterations, total_simulation_steps,
                                    model_reporters)]

    # run with firing schedule
    fixed_params["firing_schedule"] = {"steps": {40},
                                       "actor retirement probs": [.6, .5, .4, .3]}

    b_runs.append(batchrun.get_batchrun(simulation, variable_params, fixed_params, iterations, total_simulation_steps,
                                        model_reporters))

    timeseries.make_time_series_figures(b_runs, out_directory, level_names, burn_in_steps=burn_in_steps, stdev=True,
                                        ncol=4, shock_step=fixed_params["shock_step"], experiment_name=experiment_name)

    measures = data_collector.keys()
    convergence.make_convergence_table(b_runs, measures, out_directory, fixed_params["vacancy_trans_prob_matrix"],
                                       burn_in_steps=burn_in_steps)
