"""
tools for plotting per-step means and standard deviations of metrics from batch runs of VacancyChainAgentBasedModel.
the data structure is of a list containing onested dicts, with structure
    [models in the order that they ran
        {actor-steps per model
            {metrics per actor-step
                {submetrics (one per level of the vacancy chain system, plus pooled)
                    {means and standard deviations of the submetric (across model runs at the same actor-step)
                    }
                }
            }
        }
    ]

"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from string import punctuation
from copy import deepcopy
import csv


def make_time_series_figures(batchruns, out_dir, burn_in_steps=0, shock_step=0, stdev=True, experiment_name=''):
    """
    Makes time series figures and save. them to disk
    If input is more than one batchrun, overlay the output of the two batchruns in the same figure, so you can see the
    difference between runs with and without shocks.

    NB: if dealing with multiple batchruns, each batchrun must have the exact same parametrisation but for the shocks.

    NB: if dealing with multple batchruns, each must have identical nesting and key:value structure, o
        only bottom level data/values may differ

    :param batchruns: list, each of a batchrun, i.e. one model run n-iterations
    :param out_dir: str, where we want the figures to live
    :param burn_in_steps: int, number of steps at the beginning of model runs that we want to exclude from the figures
    :param shock_step: int, when (if at all) we draw a dashed, vertical line to signify the step at which a system
                            shock occurred; 0 by default, i.e. no shock
    :param stdev: bool, whether we plot shaded areas around the mean line, representing two standard deviations away
                  from the mean; False by default
    :param experiment_name: str, the name of the particular simulation experiment, e.g. "The Big Purge"
    :return None
    """

    batchruns_per_step_stats = [get_means_std(br) for br in batchruns]
    prototype_br_stats = batchruns_per_step_stats[0]
    for br_per_step_stats in batchruns_per_step_stats:
        # look into the metrics
        for metric_name in prototype_br_stats.keys():
            colour_counter = 0
            for line_name in prototype_br_stats[metric_name].keys():  # look into the submetrics
                b_run_mean_line, b_run_stdev_line = get_batch_run_mean_stdev_lines(br_per_step_stats, metric_name,
                                                                                   line_name, burn_in_steps)
                # plot the lines
                plot_mean_line(shock_step, metric_name, line_name, b_run_mean_line, b_run_stdev_line, colour_counter,
                               "-", stdev)
                colour_counter += 1

            # name the plot, save the figure to disk
            save_figure(metric_name, batchruns[0], out_dir, shock_step=shock_step, experiment_name=experiment_name)


def get_batch_run_mean_stdev_lines(batch_run_per_step_stats, metric_name, line_name, burn_in_steps):
    """
    Get the mean or stdev line (as a list) of per-step metric values averaged or stdeve'd across all model
    iterations. So each value of the list is e.g. an average of metric values at that step, across all model iterations,
    and the values are in the order of model steps, e.g. 1,2,3...

    :param batch_run_per_step_stats:
    :param metric_name: str, the name of a metric, e.g. average_vacancy_chain_length
    :param line_name: str, the name of a particular line in that metric, e.g. "Level 1"; this is important because
                      most metrics are disaggregated, usually by hierarchical level, and we want to plot multiple of
                      these lines in one graph.
    :param burn_in_steps: the number of steps from the beginning of the model that we DO NOT want to see
    :return: a 2-tuple of lists, the mean line and stdev line for that metric, for that line
    """
    br_mean_line, br_stdev_line = None, None
    # NB: "measure type" is either "Mean Across Runs" or "StDev Across Runs"
    for measure_type in batch_run_per_step_stats[metric_name][line_name].keys():
        # NB: ignore burn-in steps
        if measure_type == "Mean Across Runs":
            br_mean_line = batch_run_per_step_stats[metric_name][line_name][measure_type][burn_in_steps:]
        else:
            br_stdev_line = batch_run_per_step_stats[metric_name][line_name][measure_type][burn_in_steps:]
    return br_mean_line, br_stdev_line


# HELPERS FOR TRANSFORMING TIME SERIES DATA

# helper function for make_timeseries_figures
def get_means_std(batchrun):
    """for each step, get a submetric's mean and starndard deviations across all model runs"""
    metrics = get_metrics_timeseries_dataframes(batchrun)
    per_step_stats = deepcopy(metrics)  # keep nested dict structure but replace dataframes with means and stdevs
    for k in metrics.keys():  # look into the metrics
        for l in metrics[k].items():  # look into the submetrics
            # calculate means and standard deviations, across all model runs
            mean = l[1].mean(axis=0)
            stdev = l[1].std(axis=0)
            if isinstance(per_step_stats[k][l[0]], pd.DataFrame):
                per_step_stats[k][l[0]] = {}
            # put into appropriate place
            per_step_stats[k][l[0]]["Mean Across Runs"] = mean
            per_step_stats[k][l[0]]["StDev Across Runs"] = stdev
    return per_step_stats


# helper function for get_means_std
def get_metrics_timeseries_dataframes(batchrun):
    """take a batchrun and return a dict of pd.DataFrames, one for each metric
    each dataframe shows metric timeseries across steps, for each run; example output below
                    METRIC 1
    eg.         step 1  step 2
        run 1     4       5
        run 2    4.5      6
        run 3    2.2      3
    """
    models_in_run_order = get_data_of_models_in_run_order(batchrun)
    metric_names = models_in_run_order[0].columns.values
    metric_dataframes = {m: {} for m in metric_names}
    for one_run in models_in_run_order:
        for metric in one_run.columns.values:
            values_across_steps = one_run.loc[:, metric]
            flat_values_across_steps = flatten_dict(values_across_steps)
            flatten_dicts_into_df(flat_values_across_steps, metric_dataframes, metric)
    return metric_dataframes


# helper function for get_metrics_timeseries_dataframes
def get_data_of_models_in_run_order(batchrun):
    """returns all per-step model data (as a pd.DataFrame) for each run, in the order of the runs"""
    tuples_in_run_order = [(int(str(k).translate(str.maketrans('', '', punctuation))),
                            v["Data Collector"]) for k, v in batchrun.model_vars.items()]
    tuples_in_run_order.sort()
    models_in_run_order = [i[1].get_model_vars_dataframe() for i in tuples_in_run_order]
    return models_in_run_order


# helper function for get_metrics_timeseries_dataframes
def flatten_dict(pandas_series):
    """
    flatten a pandas series of dicts, where each dict has the same keys, into a dict whose keys are a list
    of values across (former) subdicts
    e.g. pd.Series({"me": you, "her": him}, {"me": Thou, "her": jim}) => {"me": you, Thou, "her": him, jim}
    """
    # TODO this might be messing up order, need to look into it again
    keys = pandas_series[0].keys()
    values_across_steps = {k: [] for k in keys}
    for step in pandas_series:
        for base_value in step.items():
            values_across_steps[base_value[0]].append(base_value[1])
    return values_across_steps


# helper function for get_metrics_timeseries_dataframes
def flatten_dicts_into_df(input_dict, output_dict, output_dict_key):
    """
    for a set of dicts, each with the same keys and whose values are lists of equal length,
    stacks the lists in a pd.DataFrame named after the key, and insert the new key:values into some dict.
    e.g. {"me": [you, Thou], "her": [him, jim]}
         {"me": [Pradeep, King], "her": [without, slim]}"
         => some_dict{
         "me" : pd.DataFrame(
            you         thou
            Pradeep     King),
         "her" : pd.DataFrame(
            him          jim
            without      slim) }
    """
    # if output_dict empty at certain key, make its value a dict, full of input_dict_key : pd.DataFrame
    if output_dict[output_dict_key] == {}:
        input_dict_keys = list(input_dict.keys())
        output_dict[output_dict_key] = {k: pd.DataFrame() for k in input_dict_keys}
    # turn the lists into pd.Series and append them to the dataframe
    for i in input_dict.items():
        output_dict[output_dict_key][i[0]] = output_dict[output_dict_key][i[0]].append(pd.Series(i[1]),
                                                                                       ignore_index=True)


# HELPERS FOR PLOTTING

def plot_mean_line(shock_step, metric_name, line_name, mean_line, stdev_line, colour_counter, linestyle, stdev=True):
    """Plot a mean line, and optionally a shaded area of two standard deviations around the mean."""

    # since cohorts have total turnover in 30 years, use only the thirty years after the shock step (inclusive)
    if metric_name == "percent_actors_from_before_first_shock":
        mean_line = mean_line[shock_step:shock_step+31]
        stdev_line = stdev_line[shock_step:shock_step+31]

    colours = ['r-', 'b-', 'g-', 'k-', 'y-', 'c-', 'm-']
    x = np.linspace(0, len(mean_line) - 1, len(mean_line))
    plt.plot(x, mean_line, colours[colour_counter], linestyle=linestyle, label=line_name)

    if stdev:
        # make sure lower stdev doesn't go below zero
        stdev_lowbound = mean_line - stdev_line * 2
        stdev_lowbound[stdev_lowbound < 0] = 0

        # and if dealing with percentages, that higher stdev line doesn't go above 100
        stdev_highbound = mean_line + stdev_line * 2
        if "percent" in metric_name:
            stdev_highbound[stdev_highbound > 100] = 100

        plt.fill_between(x, stdev_lowbound, stdev_highbound, color=colours[colour_counter][0], alpha=0.2)


# plotting helper for make_time_series_figures
def save_figure(metric_name, batchrun, out_dir, shock_step=0, experiment_name=''):
    """Complete a figure with titles, legend, extra line-markers and grid, then saves the figure to disk."""

    y_axis_labels = {"average_career_length": "years", "average_vacancy_chain_length": "steps",
                     "percent_female_actors": "percent", "percent_actors_from_before_first_shock": "percent",
                     "get_count_vacancies_in_system": "count", "get_count_vacancies_per_step": "count",
                     "get_actor_counts": "count", "get_agent_sets_sizes": "count"}

    # if no title name provided, use function name
    if experiment_name:
        plt.title(str(experiment_name + '\n' + metric_name.replace('_', ' ')).title())
    else:
        plt.title(metric_name.replace('_', ' ').title())

    #
    ax = plt.subplot(111)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1,
                     box.width, box.height * 0.9])

    # if there is a shock, draw a vertical, dashed line indicating at which step the shock occurred
    if shock_step:
        ax.axvline(x=shock_step, lw=2, color='k', linestyle='--')

    # add the legend, label the x and y axes, and add gridlines
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.14), fancybox=True, shadow=True, ncol=5, fontsize='medium')
    plt.xlabel("periods")
    plt.ylabel(y_axis_labels[metric_name])
    plt.grid()

    # name and save the figure
    iterations, steps = str(batchrun.iterations), str(batchrun.max_steps)
    figure_filename = out_dir + metric_name + "_" + iterations + "runs_" + steps + "steps.png"
    plt.savefig(figure_filename)

    plt.close()


# HELPERS FOR TABLE METRICS

def theoretical_observed_chain_length(vacancy_transition_probability_matrix, batchruns, out_dir):
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
    """

    # by default, consider all steps, i.e. don't throw away the burn-ins
    burn_in_steps = 0
    for b_run in batchruns:
        predicted_chain_lengths = markov_predicted_chain_length(vacancy_transition_probability_matrix)

        observed_average_chain_lengths = []
        b_run_per_step_stats = get_means_std(b_run)
        for line_name in b_run_per_step_stats["average_vacancy_chain_length"].keys():
            b_run_mean_line = get_batch_run_mean_stdev_lines(b_run_per_step_stats, "average_vacancy_chain_length",
                                                             line_name, burn_in_steps)[0]
            observed_average_chain_lengths.append(np.mean(b_run_mean_line))

    # save to disk the comparisons, in a csv table
        with open(out_dir + "theoretical_vs_simulated_chain_lengths.csv", "w") as out_f:
            writer = csv.writer(out_f)
            writer.writerow(["Level", "Markov Chain Predictions", "Average of Observed Chain Lengths"])
            for i in range(0, len(predicted_chain_lengths)):
                writer.writerow([i, predicted_chain_lengths[i], observed_average_chain_lengths[i]])


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
    return pred_chain_lengths
