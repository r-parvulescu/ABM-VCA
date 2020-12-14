"""
Functions for plotting per-step means and standard deviations of metrics from batchruns of VacancyChainAgentBasedModel.
"""

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from data import helpers


def make_time_series_figures(batchruns, out_dir, level_names, experiment_names, burn_in_steps=0, shock_step=0,
                             stdev=True, fig_title='', new_level_info=None):
    """
    Makes time series figures and save them to disk. If the input is more than one batchrun, overlay the output of the
    two batchruns in the same figure, so you can see the difference between batchruns.

    NB: if dealing with multiple batchruns, each must have identical nesting and key:value structure, only bottom level
        data/values may differ

    NB: burn_in_steps <= shock_step, otherwise you get garbage plotting, cannot promise it'll work at all

    :param batchruns: list, each of a batchrun, i.e. one model run n-iterations
    :param out_dir: str, where we want the figures to live
    :param level_names: dict, the names associated with different hierachical levels, e.g. {1: "Top", 2: "Bottom"}
    :param experiment_names: list, containing strings of the names of experiments that are compared in said graph,
                             e.g. ["baseline", "with purge"]
    :param burn_in_steps: int, number of steps at the beginning of model runs that we want to exclude from the figures
    :param shock_step: int, when (if at all) we draw a dashed, vertical line to signify the step at which a system
                            shock occurred; 0 by default, i.e. no shock
    :param fig_title: str, the title of the particular figure, e.g. "Baseline vs. The Big Purge"
    :param stdev: bool, whether we plot shaded areas around the mean line, representing two standard deviations away
                  from the mean; False by default
    :param new_level_info: dict, with four keys:
                            a) "level addition batchrun index" : the position in the list of batchruns in which the
                               experiment with the level addition is located. The value is an int >= 0
                            b) "new level rank" : where in the hierarchy of levels the new level enters; int > 0, and
                               NB that hierarchical levels are 1-indexed and reverse order, i.e. 1 > 2 > 3 > 4 ...
                            c) "new level addition step" : int, the actor-step at which the new level is added, e.g. 20
                            d) "total actor stess" : int, the total number of actor steps in the simulation, e.g. 100
    :return None
    """

    linestyles = ['-', '--', ':', '-.']

    batch_runs_per_step_stats = [helpers.get_means_std(br) for br in batchruns]

    metric_names = batch_runs_per_step_stats[0].keys()

    for m_name in metric_names:

        plt.figure(figsize=(7, 7))  # set the size of the figure

        for idx, br_per_step_stats in enumerate(batch_runs_per_step_stats):

            colour_counter = 0
            line_style = linestyles[idx]
            # handle proper truncating with burn-in-steps if experiment features the introduction of a new level
            total_steps = new_level_info["total actor steps"] if new_level_info else None

            for line_name in br_per_step_stats[m_name].keys():
                b_run_mean_line, b_run_stdev_line = helpers.get_brun_mean_stdev_lines(br_per_step_stats, m_name,
                                                                                      line_name, burn_in_steps,
                                                                                      new_level_total_steps=total_steps)

                if new_level_info and m_name != "agent_sets_sizes":
                    handle_level_addition(new_level_info, idx, m_name, line_name, b_run_mean_line, b_run_stdev_line,
                                          level_names, line_style, colour_counter, stdev=stdev, shock_step=shock_step,
                                          burn_in_steps=burn_in_steps)

                else:  # when not dealing with experimental comparisons in which we have level introductions
                    plot_mean_line(m_name, line_name, b_run_mean_line, b_run_stdev_line, colour_counter,
                                   line_style, level_names, stdev, shock_step=shock_step,
                                   burn_in_steps=burn_in_steps)

                colour_counter += 1

        # name the plot, save the figure to disk
        save_figure(m_name, batchruns[0], out_dir, level_names, experiment_names, burn_in_steps=burn_in_steps,
                    shock_step=shock_step, fig_title=fig_title)


def handle_level_addition(new_level_info, br_idx, measure_name, line_name, b_run_mean_line, b_run_stdev_line,
                          level_names, line_style, colour_counter, stdev=True, shock_step=0, burn_in_steps=0):
    """
    Things get much more complicated when an experiment features the addition of a new hierarchical level. Essentially,
    the problem is what the datacollector functions go from a typology with n-levels to one with n+1 levels, in which
    the new level can be squeezed in between the old ones. For example, if we introduce a new level in rank 2 (i.e.
    second from top position) in a three level system, we end up with the following mapping of level names:

    Old Level Typology    New Level Typology
            1                     1
                                  2 (the new level added in)
            2                     3
            3                     4

    The problem is that as far as the computer is concerned, levels 2 and 3 continue throughout and leven 4 is the
    genuinely new one, but really level 2 is new and 3 and 4 are just re-namings. To obtain non-spurious graphing, we
    need to ensure, for example, that the time-series from the old level 2 connects seamlessly to the time-series from
    the new level 3 (which is just the old level 2, but with a new name). Likewise, we need to assign a distinct colour
    for the genuinely new level (the new level 2).

    This function achieves this by segmenting the lines at the point of new level introduction, then
    re-colouring and padding them to ensure smooth transitions and clear identification.

    It's all very cluge-y, but trying to handle this issue upstream in the datacollectors risks messing up dependencies.

    NB: levels are hierachically ranked in reverese order, i.e. 1 > 2 > 3 > 4 ...

    NB: recall that the actual data lines are pandas.Series objects
    """

    # for the batchruns that DON'T include level additions
    if br_idx != new_level_info["level addition batchrun index"]:

        # bump down in colour the lines at or below the rank of the level that we're introducing
        if line_name >= new_level_info["new level rank"]:  # recall, ranks are in reverse order, i.e., 1 > 2 > 3
            plot_mean_line(measure_name, line_name + 1, b_run_mean_line, b_run_stdev_line,
                           colour_counter + 1, line_style, level_names, stdev, shock_step=shock_step,
                           burn_in_steps=burn_in_steps)
        else:  # leave intact the lines higher than the level that we're introducing
            plot_mean_line(measure_name, line_name, b_run_mean_line, b_run_stdev_line,
                           colour_counter, line_style, level_names, stdev, shock_step=shock_step,
                           burn_in_steps=burn_in_steps)

    else:  # these are the batchruns that DO include level additions

        # if we're dealing with an apparently "new" line, that starts partway through
        if len(b_run_mean_line) <= new_level_info["total actor steps"]:

            # the apparently new bottom line (e.g. if we expand a three-level system by one level, then
            # the new level 4) is actually a continuation of the old bottom line (the old level 3). Thereore, this new
            # line needs to be padded so that it seamlessly starts at the step of new level introduction
            nones = (new_level_info["new level addition-step"]) * [None]  # pad with Nones
            b_run_mean_line = pd.Series(nones + b_run_mean_line.tolist())
            b_run_stdev_line = pd.Series(nones + b_run_stdev_line.tolist())
            plot_mean_line(measure_name, line_name, b_run_mean_line, b_run_stdev_line, colour_counter,
                           line_style, level_names, stdev, shock_step=shock_step, burn_in_steps=burn_in_steps)

        else:  # else, we're dealing with the other apparently "old" lines

            # the "old" lines at or below the new rank need to be bumped down BEFORE the level addition, since after the
            # addition the data on that line reflect measures from a genuinely new level. and needs a distinct colour.
            # So we bump down the colour scheme of the pre-introduction data, but keep post-introduction scheme as is.
            if line_name >= new_level_info["new level rank"]:  # again, ranks are reversed
                # deal with the line segment from BEFORE the introduction of the new level
                pre_lvl_intro_mean_line = b_run_mean_line[:new_level_info["new level addition-step"] + 1]
                pre_lvl_intro_stdev_line = b_run_stdev_line[:new_level_info["new level addition-step"] + 1]
                plot_mean_line(measure_name, line_name + 1, pre_lvl_intro_mean_line, pre_lvl_intro_stdev_line,
                               colour_counter + 1, line_style, level_names, stdev,
                               shock_step=shock_step, burn_in_steps=burn_in_steps)

                # deal with the line segment from AFTER the introduction of the new level
                post_lvl_intro_mean_line = b_run_mean_line[new_level_info["new level addition-step"] + 1:]
                post_lvl_intro_stdev_line = b_run_stdev_line[new_level_info["new level addition-step"] + 1:]
                # need to pad these post-introduction segments so they start at the introduction point
                nones = (new_level_info["new level addition-step"]) * [None]
                post_lvl_intro_mean_line = pd.Series(nones + post_lvl_intro_mean_line.tolist())
                post_lvl_intro_stdev_line = pd.Series(nones + post_lvl_intro_stdev_line.tolist())
                plot_mean_line(measure_name, line_name, post_lvl_intro_mean_line, post_lvl_intro_stdev_line,
                               colour_counter, line_style, level_names, stdev, shock_step=shock_step,
                               burn_in_steps=burn_in_steps)

            else:  # leave untouched the lines hierarchically superior to the level we introduce partway through

                plot_mean_line(measure_name, line_name, b_run_mean_line, b_run_stdev_line, colour_counter, line_style,
                               level_names, stdev, shock_step=shock_step, burn_in_steps=burn_in_steps)


def plot_mean_line(metric_name, line_name, mean_line, stdev_line, colour_counter, linestyle, level_names, stdev=True,
                   shock_step=0, burn_in_steps=0):
    """Plot a mean line, and optionally a shaded area of two standard deviations around the mean."""

    # since cohorts have total turnover in 32 years, use only the first thirty-two years after the shock step
    # NB: if the shock step is zero, then this limits our view to the initialisation cohort of actors.
    if metric_name == "percent_actors_from_before_shock":
        window_start = shock_step - burn_in_steps
        mean_line = mean_line[window_start:window_start+33]
        stdev_line = stdev_line[window_start:window_start+33]

    # if the line name refers to a hierarchical level (e.g. 1, 2), give it a proper name
    if isinstance(line_name, int):
        line_name = level_names[line_name] if line_name else "Level " + str(line_name)

    colours = ['r', 'b', 'g', 'k', 'y', 'c', 'm']
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


def save_figure(metric_name, batchrun, out_dir, level_names, experiment_names, shock_step=0, fig_title='',
                burn_in_steps=0):
    """Complete a figure with titles, legend, extra line-markers and grid, then saves the figure to disk."""

    y_axis_labels = {"average_career_length": "actor steps", "average_vacancy_chain_length": "positions",
                     "percent_female_actors": "percent", "percent_actors_from_before_shock": "percent",
                     "count_vacancies_still_in_system": "count", "count_vacancies_per_step": "count",
                     "actor_counts": "count", "agent_sets_sizes": "count",
                     "actor_turnover_rate": "actor turnover per position",
                     "time_to_promotion_from_last_level": "actor steps",
                     "time_to_retirement_from_last_level": "actor steps",
                     "net_vacancy_effects": "value unit"}

    # if no title name provided, use function name
    if fig_title:
        plt.title(str(fig_title + '\n' + metric_name.replace('_', ' ')).title())
    else:
        plt.title(metric_name.replace('_', ' ').title())

    # make the legend box and set the axis for drawing the vertical line
    ax = plt.subplot(111)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1,
                     box.width, box.height * 0.9])

    # needs separate treatment for "agents_sets_sizes" since we don't disaggregate this measure by level
    if metric_name == "agent_sets_sizes":
        level_names = {1: "actor", 2: "vacancy", 3: "position"}

    # if there is a shock, draw a vertical, dashed line at shock step, while accounting for burn-in steps
    if shock_step:  # percent_actors_from_before_shock needs own treatment since I limit it to a ~30 year window
        x = 0 if metric_name == "percent_actors_from_before_shock" else shock_step - burn_in_steps
        ax.axvline(x=x, lw=2, color='k', linestyle='--')

    # add the legend that identifies levels/strata of the system by color
    colours = ['r', 'b', 'g', 'k', 'y', 'c', 'm']
    lgnd1_lines = [Line2D([0], [0], c=colours[i], lw=2, ls="-") for i in range(len(level_names))]
    ncol_colours = 1 if len(level_names) <= 3 else 2
    # NB: I'm being risky here since the level come as dict values and I'm just list-ing the values so in principle
    # this could scramble up their order and mislabel the legend lines, but in practice it never does; still, risky
    color_legend = plt.legend(lgnd1_lines, list(level_names.values()), loc='upper right', bbox_to_anchor=(1.1, -0.09),
                              ncol=ncol_colours)
    plt.gca().add_artist(color_legend)

    # add another legend for linestyles, where each style indicates a different simulation case
    linestyles = ['-', '--', ':', '-.']
    lgnd2_lines = [Line2D([0], [0], c='k', lw=2, ls=linestyles[i]) for i in range(len(experiment_names))]
    ncol_exp_cases = 1 if len(experiment_names) <= 3 else 2
    linestyle_legend = plt.legend(lgnd2_lines, experiment_names, loc='upper left', bbox_to_anchor=(-0.1, -0.09),
                                  ncol=ncol_exp_cases)
    plt.gca().add_artist(linestyle_legend)

    # add label the x and y axes, add gridlines
    plt.xlabel("actor steps")
    plt.ylabel(y_axis_labels[metric_name])
    plt.grid()

    # name and save the figure
    iterations, steps = str(batchrun.iterations), str(batchrun.max_steps)
    figure_filename = out_dir + metric_name + "_" + iterations + "runs_" + steps + "steps.png"
    plt.savefig(figure_filename)

    plt.close()
