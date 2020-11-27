"""
Functions for making tables based on equilibrium, post-burn-in model behaviour of batchruns
of VacancyChainAgentBasedModel.
"""

from data import helpers, collector
import numpy as np
import csv


def theoretical_observed_chain_length(vacancy_transition_probability_matrix, batchruns, out_dir, burn_in_steps=0):
    """
    Given the inter-level vacancy transition matrix used to simulate the vacancy dynamics, compare the chain length
    as predicted by a Markov-chain model to the observed chain length, averaged across model steps and iterations.
    Results are disaggregated by the state in which the chain started, i.e. comparing predicted vs. observed lengths
    of chain started in level 1.

    :param vacancy_transition_probability_matrix: list of lists, the matrix showing the transition probabilities
                                                  matrix of vacancies. The form is as in the example below

                                                             Level 1     Level 2     Level 3     Retire
                                                  Level 1    [[0.3,         0.4,        0.1,        0.2],
                                                  Level 2     [0.1,         0.4,        0.4,        0.1],
                                                  Level 3     [0.05,        0.05,       0.2,        0.7]]

    :param batchruns: a list of batchruns, where each batchrun is the same model run in n-iterations
    :param out_dir: str, directory where we want the comparison table to live
    :param burn_in_steps: int, how many beginning steps we treat as the model hitting convergence (i.e. "burning in")
                          and therefore don't consider for the metric calculations. Default is 0, i.e. no burn-in.
    """

    # by default, consider all steps, i.e. don't throw away the burn-ins
    for b_run in batchruns:
        predicted_chain_lengths = collector.markov_predicted_chain_length(vacancy_transition_probability_matrix)

        observed_average_chain_lengths = []
        b_run_per_step_stats = helpers.get_means_std(b_run)
        for line_name in b_run_per_step_stats["average_vacancy_chain_length"].keys():
            b_run_mean_line = helpers.get_batch_run_mean_stdev_lines(b_run_per_step_stats,
                                                                     "average_vacancy_chain_length", line_name,
                                                                     burn_in_steps)[0]
            observed_average_chain_lengths.append(np.mean(b_run_mean_line))

        # save to disk the comparisons, in a csv table
        with open(out_dir + "theoretical_vs_simulated_chain_lengths.csv", "w") as out_f:
            writer = csv.writer(out_f)
            writer.writerow(["Level", "Markov Chain Predictions", "Average of Observed Chain Lengths"])
            for i in range(0, len(predicted_chain_lengths)):
                writer.writerow([i, predicted_chain_lengths[i], observed_average_chain_lengths[i]])
