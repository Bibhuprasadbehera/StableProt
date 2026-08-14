"""This path used to contain a fabricated ΔTm evaluation (random labels + injected noise).

The real mutation benchmark is:

    experiments/run_real_mutation_benchmark.py

The old script is quarantined at:

    experiments/src/eval/_do_not_run/evaluate_mutation_deltatm.py

Do not import or run that file for any published figure.
"""

raise RuntimeError(
    "evaluate_mutation_deltatm.py is retired. Use experiments/run_real_mutation_benchmark.py."
)
