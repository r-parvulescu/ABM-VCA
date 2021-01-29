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
