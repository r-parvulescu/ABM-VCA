"""Top-level script form which we run the various simulation experiments."""

import model
import batchrun
import plotter
import collector
from local import out_dir

"""
positions_per_level, actor_retire_probs, vacancy_trans_prob_matrix, firing_schedule,
                 growth_orders, start_percent_female, percent_female_entry_per_step, actor_datacollector,
                 vacancy_datacollector, shock_step=None, vacancy_move_period=53
"""

# Use White's figures on vacancy chain mobility in the 1922-1937 Methodist Church system (White 1970:125)
# no shocks, only men
total_steps = 5000
fem_entry_per_step = {i: 0 for i in range(total_steps)}
whites_methodist_church_outdir = out_dir + 'methodists_1922-1937/'
whites_methodist_churh_exp_name = "White's Methodist Church, 1922-1937"
whites_methodist_church = {"positions_per_level": [20, 40, 80],
                           "actor_retire_probs": [0.1, 0.05, 0.05],
                           "vacancy_trans_prob_matrix": [[.46, .33, .05, .16],
                                                         [.11, .41, .12, .36],
                                                         [.02, .16, .24, .58]],
                           "firing_schedule": {"steps": {}, "actor retirement probs": []},
                           "growth_orders": {"steps": {}, "level growths": {}},
                           "start_percent_female": 0.,
                           "percent_female_entry_per_step": fem_entry_per_step,
                           }

# RO Judges 2014
"""
For 2014, full system size was roughly [110, 830, 1500, and 2200]
super trimmed down, we can do [10, 100, 150, 200]  -- this runs fairly quickly, ~24 second per iteration for 1500 steps
at a third, we get roughly [35, 280, 500, 730], which takes 3.5 minutes per iteration
If 

"""
total_steps = 1500
fem_entry_per_step = {i: 0 for i in range(total_steps)}
ro_judges_2014_outdir = out_dir + 'ro_judges/'
ro_judges_2014_exp_name = "Romanian Judges 2014"
ro_judges_2014 = {"positions_per_level": [35, 280, 500, 730],  # the tenth = [10, 100, 150, 200]
                  "actor_retire_probs": [0.06, 0.04, 0.03, 0.04],
                  "vacancy_trans_prob_matrix": [[0., 100., 0., 0., 0.],
                                                [0., .06, .68, .1, .16],
                                                [0., 0.,  .21, .66, .13],
                                                [0., .01, .02, .36, .61]],
                  "firing_schedule": {"steps": {}, "actor retirement probs": []},
                  "growth_orders": {"steps": {}, "level growths": {}},
                  "start_percent_female": 0.,
                  "percent_female_entry_per_step": fem_entry_per_step,
                  }

if __name__ == "__main__":

    data_collector = {"average_career_length": collector.get_average_actor_career_length,
                      "average_vacancy_chain_length": collector.get_average_vacancy_chain_length,
                      "get_count_vacancies_per_step": collector.get_count_vacancies_per_step,
                      }

    iterations = 5

    experiments = [  # (whites_methodist_church, whites_methodist_church_outdir, whites_methodist_churh_exp_name),
        (ro_judges_2014, ro_judges_2014_outdir, ro_judges_2014_exp_name)
    ]

    for exp in experiments:
        variable_params, fixed_params = None, exp[0]
        fixed_params["data_collector"] = data_collector
        fixed_params["shock_step"] = 0
        fixed_params["vacancy_move_period"] = 15

        burn_in_steps = 0
        num_start_steps_to_show = 0
        out_directory = exp[1]
        model_reporters = {"Data Collector": lambda m: m.data_collector}

        simulation = model.VacancyChainAgentBasedModel

        b_runs = batchrun.compare_with_without_shocks(simulation, variable_params, fixed_params, iterations,
                                                      total_steps, model_reporters)

        plotter.make_time_series_figures(b_runs, out_directory, burn_in_steps=burn_in_steps,
                                         plot_shock_step=num_start_steps_to_show, experiment_name=exp[2], stdev=True, )

"""
FULL DATA COLLECTOR

    data_collector = {"average_career_length": collector.get_average_actor_career_length,
                      "average_vacancy_chain_length": collector.get_average_vacancy_chain_length,
                      "percent_female_actors": collector.get_percent_female_actors,
                      "percent_actors_from_before_first_shock": collector.get_percent_actors_from_before_shock,
                      "get_count_vacancies_in_system": collector.get_count_vacancies_still_in_system,
                      "get_count_vacancies_per_step": collector.get_count_vacancies_per_step,
                      "get_actor_counts": collector.get_actor_count,
                      "get_agent_sets_sizes": collector.get_agent_sets_sizes
                      }
"""

# TODO
#  - get the Markov estimates for chain length by departure stratum and compare to simulation results
#  - calculate distance between input and observed transition matrixes, average over all steps in one iteration, then
#    averaged over all iterations
#  - figure out what the system growth of retirement rates were for White's churches, or Chase's crabs
#  - figure out what extra metrics Chase wanted and get those
#  - make another table descriptor to spit out vacancy transition matrixes as you need them, plus average them
#    over pre-defined intervals
