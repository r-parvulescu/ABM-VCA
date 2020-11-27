"""
Functions for plotting per-step means and standard deviations of metrics from batchruns of VacancyChainAgentBasedModel.
"""

import matplotlib.pyplot as plt
import numpy as np
from data import helpers


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
    for br in batchruns:
        br_per_step_stats = helpers.get_means_std(br)
        # look into the metrics
        for metric_name in br_per_step_stats.keys():
            colour_counter = 0
            for line_name in br_per_step_stats[metric_name].keys():  # look into the submetrics
                b_run_mean_line, b_run_stdev_line = helpers.get_batch_run_mean_stdev_lines(br_per_step_stats,
                                                                                           metric_name, line_name,
                                                                                           burn_in_steps)
                # plot the lines
                plot_mean_line(shock_step, metric_name, line_name, b_run_mean_line, b_run_stdev_line, colour_counter,
                               "-", stdev)
                colour_counter += 1

            # name the plot, save the figure to disk
            save_figure(metric_name, batchruns[0], out_dir, shock_step=shock_step, experiment_name=experiment_name)


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
