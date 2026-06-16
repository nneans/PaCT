"""Dataset configuration for the PaCT benchmark.

This module centralizes everything that is specific to the benchmark event
logs: which XES event attributes are encoded as categorical prefix channels
(DATASET_CONFIG), how benchmark file names map to those config keys
(FILENAME_TO_CONFIG_KEY), and the order datasets are run in (DATASET_ORDER).
experiment.py reads these mappings; a dataset absent from DATASET_CONFIG falls
back to activity-only prefix input.
"""

# Categorical prefix attributes per dataset (config key -> XES attribute keys).
DATASET_CONFIG = {
    "helpdesk":                        ["org:resource"],
    "Sepsis":                          ["org:group"],
    "Env Permit":                      ["org:group", "org:resource"],
    "BPIC13_closed":                   ["org:group", "org:resource"],
    "BPIC13_incidents":                ["impact", "org:group", "org:resource", "org:role"],
    "BPIC12_A":                        ["org:resource"],
    "BPIC12_O":                        ["org:resource"],
}

# Benchmark file name -> DATASET_CONFIG key.
FILENAME_TO_CONFIG_KEY = {
    "Helpdesk.xes.gz": "helpdesk",
    "SEPSIS.xes.gz": "Sepsis",
    "env_permit.xes.gz": "Env Permit",
    "BPI_Challenge_2013_closed_problems.xes.gz": "BPIC13_closed",
    "nasa.xes.gz": "nasa",
    "BPI_Challenge_2012_A.xes.gz": "BPIC12_A",
    "BPI_Challenge_2012_O.xes.gz": "BPIC12_O",
    "bpi_challenge_2013_incidents.xes.gz": "BPIC13_incidents",
    "BPI_Challenge_2012_W_Complete.xes.gz": "BPIC12_WC",
    "BPI_Challenge_2012_Complete.xes.gz": "BPIC12_Complete",
    "BPI_Challenge_2012_W.xes.gz": "BPIC12_W",
    "BPI_Challenge_2012.xes.gz": "BPIC12",
}

# Order datasets are processed in when none are explicitly requested.
DATASET_ORDER = [
    "BPI_Challenge_2013_closed_problems.xes.gz",
    "env_permit.xes.gz",
    "Helpdesk.xes.gz",
    "SEPSIS.xes.gz",
    "BPI_Challenge_2012_O.xes.gz",
    "BPI_Challenge_2012_A.xes.gz",
    "nasa.xes.gz",
    "bpi_challenge_2013_incidents.xes.gz",
    "BPI_Challenge_2012_W_Complete.xes.gz",
    "BPI_Challenge_2012_Complete.xes.gz",
    "BPI_Challenge_2012_W.xes.gz",
    "BPI_Challenge_2012.xes.gz",
]
