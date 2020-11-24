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


def make_time_series_figures(batchruns, out_directory, burn_in_steps=0, plot_shock_step=0,
                             stdev=True, experiment_name=''):
    """
    Makes time series figures and save. them to disk
    If input is more than one batchrun, overlay the output of the two batchruns in the same figure, so you can see the
    difference between runs with and without shocks.
    NB: if dealing with multiple batchruns, each batchrun must have the exact same parametrisation but for the shocks.

    :param batchruns
    :param out_directory:
    :param burn_in_steps:
    :param plot_shock_step
    :param stdev:
    :param experiment_name:
    :return None

    """

    br1_per_step_stats = get_means_std(batchruns[0])
    if len(batchruns) == 2:
        br2_per_step_stats = get_means_std(batchruns[1])
    # look into the metrics
    # NB: all batchruns must have identical nesting and key:value structure, only bottom level data/values differ
    for k in br1_per_step_stats.keys():
        colour_counter = 0
        for l in br1_per_step_stats[k].keys():  # look into the submetrics
            br1_mean_line, br1_stdev_line = None, None
            br2_mean_line, br2_stdev_line = None, None
            for m in br1_per_step_stats[k][l].keys():  # look into means and stdevs
                if m == "Mean Across Runs":
                    br1_mean_line = br1_per_step_stats[k][l][m][burn_in_steps:]  # NB: ignore burn-in steps
                    if len(batchruns) == 2:
                        br2_mean_line = br2_per_step_stats[k][l][m][burn_in_steps:]
                else:
                    br1_stdev_line = br1_per_step_stats[k][l][m][burn_in_steps:]
                    if len(batchruns) == 2:
                        br2_stdev_line = br2_per_step_stats[k][l][m][burn_in_steps:]
            # plot the line
            plot_mean_line(k, l, br1_mean_line, br1_stdev_line, colour_counter, "-", stdev)
            if len(batchruns) == 2:
                plot_mean_line(k, "shock", br2_mean_line, br2_stdev_line, colour_counter, "--", stdev)
            colour_counter += 1

        # name the plot, save the figure to disk
        save_figure(k, batchruns[0], out_directory, plot_shock_step=plot_shock_step, experiment_name=experiment_name)


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

# plotting helper for make_time_series_figures
def plot_mean_line(metric_name, line_name, mean_line, stdev_line, colour_counter, linestyle, stdev=True):
    """given line name, mean and associated stdev values, and a colour counter, plots a line"""
    #if "shock" in metric_name:  # since cohorts vanish in max 30 years, need a tight window for a useful plot
    #    mean_line = mean_line[15:-24]
    #    stdev_line = stdev_line[15:-24]
    colours = ['r-', 'b-', 'k-', 'g-', 'c-', 'm-', 'y-']
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
        plt.fill_between(x, stdev_lowbound, stdev_highbound, alpha=0.2)
    y_axis_labels = {"average_career_length": "years",
                     "average_vacancy_chain_length": "steps",
                     "percent_female_actors": "percent",
                     "percent_actors_from_before_first_shock": "percent",
                     "get_count_vacancies_in_system": "count",
                     "get_count_vacancies_per_step": "count",
                     "get_actor_counts": "count",
                     "get_agent_sets_sizes": "count"}
    plt.ylabel(y_axis_labels[metric_name])


# plotting helper for make_time_series_figures
def save_figure(metric_name, batchrun, out_directory, plot_shock_step=0, experiment_name=''):
    """given a title for the figure and a batchrun, completes figure, saves it to .png, and closes open plots"""
    # if no title name provided, use function name
    if experiment_name:
        plt.title(str(experiment_name + '\n' + metric_name.replace('_', ' ')).title())
    else:
        plt.title(metric_name.replace('_', ' ').title())
    ax = plt.subplot(111)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1,
                     box.width, box.height * 0.9])
    # unless we're dealing with the before-first-shock metrics (which are plotted differently)
    # add a dashed, black line at the first shock step
    #if "shock" not in metric_name:
    #    ax.axvline(x=plot_shock_step, lw=2, color='k', linestyle='--')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), fancybox=True, shadow=True, ncol=5, fontsize='medium')
    plt.xlabel("periods")
    figure_filename = out_directory + metric_name + "_" + str(batchrun.iterations) + "runs_" + \
                      str(batchrun.max_steps) + "steps.png"
    plt.grid()  # add gridlines
    plt.savefig(figure_filename)
    plt.close()
