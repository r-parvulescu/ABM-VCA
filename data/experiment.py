"""Top-level script form which we run the various simulation experiments."""

import model
import batchrun
import plotter
import collector
from local import out_dir

# Uses White's figures on vacancy mobility in the 1922-1937 Methodist Church system (White 1970:125)
# no shocks, only men
total_steps = 1500
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

# Uses transition matrix from hierarchical mobility among Romanian judges in 2014, from the Romanian Judicial
# Professions Database; for now, no shocks, only men
total_steps = 1500
fem_entry_per_step = {i: 0 for i in range(total_steps)}
ro_judges_2014_outdir = out_dir + 'ro_judges/'
ro_judges_2014_exp_name = "Romanian Judges 2014"
ro_judges_2014 = {"positions_per_level": [10, 100, 150, 200],
                  "actor_retire_probs": [0.06, 0.04, 0.03, 0.04],
                  "vacancy_trans_prob_matrix": [[0., 100., 0., 0., 0.],
                                                [0., .06, .68, .1, .16],
                                                [0., 0., .21, .66, .13],
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

    experiments = [(whites_methodist_church, whites_methodist_church_outdir, whites_methodist_churh_exp_name),
                   (ro_judges_2014, ro_judges_2014_outdir, ro_judges_2014_exp_name)]

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
                                         shock_step=num_start_steps_to_show, experiment_name=exp[2], stdev=True)

        plotter.theoretical_observed_chain_length(fixed_params["vacancy_trans_prob_matrix"], b_runs, out_directory)
