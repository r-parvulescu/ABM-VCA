"""Top-level script form which we run the various simulation experiments."""

from vc_abm import model, batchrun
from data import collector, timeseries, convergence
from local import out_dir

# toy example, featuring no downward mobility of people (therefore only downward mobility of vacancies)
total_steps = 1500
fem_entry_per_step = {i: 0.5 for i in range(total_steps)}
toy_outdir = out_dir + 'toy/'
toy_exp_name = "Toy Example"
toy = {"positions_per_level": [50, 100, 150],
       "vacancy_move_period": 15,
       "actor_retire_probs": [0.1, 0.075, 0.05],
       "vacancy_trans_prob_matrix": [[.5, .3, .1, .1],
                                     [.1, .4, .3, .2],
                                     [0., .1, .3, .6]],
       "vacancy_benefit_deficit_matrix": [[1, 2, 3, 4],
                                          [-1, 3, 4, 5],
                                          [-2, -1, 5, 6]],
       "firing_schedule": {"steps": {}, "actor retirement probs": []},
       "growth_orders": {"steps": {}, "extra positions": []},
       "start_percent_female": 0.,
       "percent_female_entry_per_step": fem_entry_per_step}

if __name__ == "__main__":

    data_collector = {"average_career_length": collector.get_average_actor_career_length,
                      "average_vacancy_chain_length": collector.get_average_vacancy_chain_length,
                      "actor_turnover_rate": collector.get_actor_turnover_rate,
                      "time_to_promotion_from_last_level": collector.get_time_to_promotion_from_last_level,
                      "time_to_retirement_from_last_level": collector.get_time_to_retirement_from_last_level,
                      "net_vacancy_effects": collector.get_net_vacancy_effects,
                      "percent_female_actors": collector.get_percent_female_actors,
                      "percent_actors_from_before_first_shock": collector.get_percent_actors_from_before_shock,
                      "agent_sets_sizes": collector.get_agent_sets_sizes,
                      "count_vacancies_still_in_system": collector.get_count_vacancies_still_in_system
                      }

    iterations = 10

    experiments = [(toy, toy_outdir, toy_exp_name)]

    for exp in experiments:
        out_directory = exp[1]
        variable_params, fixed_params = None, exp[0]
        fixed_params["data_collector"] = data_collector
        model_reporters = {"Data Collector": lambda m: m.data_collector}
        simulation = model.VacancyChainAgentBasedModel
        fixed_params["shock_step"] = 50

        # run without firing schedule
        b_runs = [batchrun.get_batchrun(simulation, variable_params, fixed_params, iterations, total_steps,
                                        model_reporters)]

        # run with firing schedule
        fixed_params["firing_schedule"] = {"steps": {50 * fixed_params["vacancy_move_period"]},
                                           "actor retirement probs": [1, 1, 1]}

        # run with growth orders
        fixed_params["growth_orders"] = {"steps": {55 * fixed_params["vacancy_move_period"]},
                                         "extra positions": [10, 20, 30]}

        b_runs.append(batchrun.get_batchrun(simulation, variable_params, fixed_params, iterations, total_steps,
                                            model_reporters))

        timeseries.make_time_series_figures(b_runs, out_directory, burn_in_steps=0,
                                            shock_step=fixed_params["shock_step"], experiment_name=exp[2], stdev=False)

        measures = data_collector.keys()
        convergence.make_convergence_table(b_runs, measures, out_directory, fixed_params["vacancy_trans_prob_matrix"],
                                           burn_in_steps=20)
