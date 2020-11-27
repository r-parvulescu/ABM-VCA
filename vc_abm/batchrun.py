"""Script to run the vacancy chain, agent-based model in batches."""

from mesa.batchrunner import BatchRunner


def get_batchrun(model, variable_params, fixed_params, iterations, max_steps, model_reporters):
    """Return one batchrun."""
    batchrun = BatchRunner(
        model,
        variable_params,
        fixed_params,
        iterations=iterations,
        max_steps=max_steps,
        model_reporters=model_reporters)
    batchrun.run_all()
    return batchrun


def compare_with_without_shocks(model, variable_params, fixed_params, iterations, max_steps, model_reporters):
    """
    If there are firing schedules or growth orders (i.e. shocks), return two batchruns, one without the shock and one
    with. If there are no shocks, just return the baseline/shockless batchrun."""

    if fixed_params["firing_schedule"]["steps"] or fixed_params["growth_orders"]["steps"]:

        shock_run = get_batchrun(model, variable_params, fixed_params, iterations, max_steps, model_reporters)

        # now get the shockless run, in order to compare the shock run to a baseline
        fixed_params["firing_schedule"]["steps"], fixed_params["growth_orders"]["steps"] = {}, {}
        no_shocks_run = get_batchrun(model, variable_params, fixed_params, iterations, max_steps, model_reporters)

        return [no_shocks_run, shock_run]
    else:
        return [get_batchrun(model, variable_params, fixed_params, iterations, max_steps, model_reporters)]
