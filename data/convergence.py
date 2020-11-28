"""
Functions for making tables based on equilibrium, post-burn-in model behaviour of batchruns
of VacancyChainAgentBasedModel.
"""

from data import helpers, collector
import numpy as np
import csv


def make_convergence_table(batchruns, measures, out_dir, vacancy_transition_probability_matrix, burn_in_steps=0):
    """
    Make a table in which you report the values of measures averaged across equilibrium (i.e. post-burn-in) steps.

    :param batchruns:
    :param measures: dict, where keys are the (str) names of the measures that we're including in the name
    :param out_dir: str, directory where we want the comparison table to live
    :param vacancy_transition_probability_matrix: list of lists, the matrix showing the transition probabilities
                                              matrix of vacancies. The form is as in the example below

                                                         Level 1     Level 2     Level 3     Retire
                                              Level 1    [[0.3,         0.4,        0.1,        0.2],
                                              Level 2     [0.1,         0.4,        0.4,        0.1],
                                              Level 3     [0.05,        0.05,       0.2,        0.7]]

    :param burn_in_steps: int, how many beginning steps we treat as the model hitting convergence (i.e. "burning in")
                          and therefore don't consider for the metric calculations. Default is 0, i.e. no burn-in.
    """

    vals = {i: {measure_name: {"measure": {}, "stdev": {}} for measure_name in measures} for i in range(len(batchruns))}

    for idx, br in enumerate(batchruns):
        b_run_per_step_stats = helpers.get_means_std(br)
        for measure_name in measures:
            for line_name in b_run_per_step_stats[measure_name].keys():
                mean_stdev = helpers.get_brun_mean_stdev_lines(b_run_per_step_stats, measure_name, line_name,
                                                               burn_in_steps=burn_in_steps)
                vals[idx][measure_name]["measure"][line_name] = np.mean(mean_stdev[0])
                vals[idx][measure_name]["stdev"][line_name] = np.mean(mean_stdev[1])

    # save table to disk
    with open(out_dir + "equilibrium_values.csv", "w") as out_f:
        writer = csv.writer(out_f)
        writer.writerow(["Batchrun", "Measure Name", "Mean or StDev", "Level", "Value"])
        for br in vals:
            for measure_name in vals[br]:
                for measure_type in vals[br][measure_name]:
                    for line_name in vals[br][measure_name][measure_type]:
                        writer.writerow([br, measure_name, measure_type, line_name,
                                         vals[br][measure_name][measure_type][line_name]])

        for lvl, value in collector.markov_predicted_chain_length(vacancy_transition_probability_matrix).items():
            writer.writerow(["", "Markov Predicted Vacancy Chain Length", "", lvl, value])
