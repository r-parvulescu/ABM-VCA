"""
Helpers for getting nicely formatted data out from the batchrun data harvester of VacancyChainAgentBasedModel.

The data structure from a batchrun harvester is of a list containing nested dicts, with structure
    [models in the order that they ran
        {actor-steps per model
            {metrics per actor-step
                {submetrics (usually one per level of the vacancy chain system)
                    {means and standard deviations of the submetric (across model runs at the same actor-step)
                    }
                }
            }
        }
    ]
"""

import pandas as pd
from string import punctuation
from copy import deepcopy


def get_brun_mean_stdev_lines(batch_run_per_step_stats, metric_name, line_name, burn_in_steps, new_level_total_steps):
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
    :param new_level_total_steps: int or None: if int, then it's the number of total actor steps in the model FOR THE
                                  CASE IN WHICH A NEW LEVEL IS INTRODUCED PARTWAY; if None, then no new level is added
    :return: a 2-tuple of lists, the mean line and stdev line for that metric, for that line
    """

    br_mean_line, br_stdev_line = None, None
    # NB: "measure type" is either "Mean Across Runs" or "StDev Across Runs"
    for measure_type in batch_run_per_step_stats[metric_name][line_name].keys():
        # NB: ignore burn-in steps
        if measure_type == "Mean Across Runs":
            br_mean_line = batch_run_per_step_stats[metric_name][line_name][measure_type]
        else:
            br_stdev_line = batch_run_per_step_stats[metric_name][line_name][measure_type]
    # if a new level is introduced partway (and therefore does not have observations for a number of actor steps at the
    # start) then burn-in-steps do not apply to that level's observations; otherwise, throw out the burn-in-steps
    if new_level_total_steps and len(br_mean_line) <= new_level_total_steps:
        return br_mean_line, br_stdev_line
    else:
        return br_mean_line[burn_in_steps:], br_stdev_line[burn_in_steps:]


def get_means_std(batchrun):
    """For each step, get a submetric's mean and starndard deviations across all model runs."""
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


def get_metrics_timeseries_dataframes(batchrun):
    """
    Take a batchrun and return a dict of pd.DataFrames, one for each metric
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


def get_data_of_models_in_run_order(batchrun):
    """Return all per-step model data (as a pd.DataFrame) for each run, in the order of the runs"""
    tuples_in_run_order = [(int(str(k).translate(str.maketrans('', '', punctuation))),
                            v["Data Collector"]) for k, v in batchrun.model_vars.items()]
    tuples_in_run_order.sort()
    models_in_run_order = [i[1].get_model_vars_dataframe() for i in tuples_in_run_order]
    return models_in_run_order


def flatten_dict(pandas_series):
    """
    Flatten a pandas series of dicts, where each dict has the same keys, into a dict whose keys are a list
    of values across (former) subdicts

    e.g. pd.Series({"me": you, "her": him}, {"me": Thou, "her": jim}) => {"me": you, Thou, "her": him, jim}
    """
    # TODO this might be messing up order, need to look into it again
    keys = pandas_series[len(pandas_series) - 1].keys()
    values_across_steps = {k: [] for k in keys}
    for step in pandas_series:
        for base_value in step.items():
            values_across_steps[base_value[0]].append(base_value[1])
    return values_across_steps


def flatten_dicts_into_df(input_dict, output_dict, output_dict_key):
    """
    For a set of dicts, each with the same keys and whose values are lists of equal length,
    stacks the lists in a pd.DataFrame named after the key, and insert the new key:values into some dict.

    e.g. {{"me": [you, Thou], "her": [him, jim]}
          {"me": [Pradeep, King], "her": [without, slim]}}

         => some_dict{"me" : pd.DataFrame(you         thou
                                          Pradeep     King),
                      "her" : pd.DataFrame(him          jim
                                           without      slim)}
    """
    # if output_dict empty at certain key, make its value a dict, full of input_dict_key : pd.DataFrame
    if output_dict[output_dict_key] == {}:
        input_dict_keys = list(input_dict.keys())
        output_dict[output_dict_key] = {k: pd.DataFrame() for k in input_dict_keys}
    # turn the lists into pd.Series and append them to the dataframe
    for i in input_dict.items():
        output_dict[output_dict_key][i[0]] = output_dict[output_dict_key][i[0]].append(pd.Series(i[1]),
                                                                                       ignore_index=True)
