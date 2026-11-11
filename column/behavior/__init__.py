"""Custom behaviors, grouped by their framework responsibility."""

from conex.nn.priority import ALL_PRIORITIES

ALL_PRIORITIES.update(
    {
        "IF": 260,
        "MarginWTA": 300,
        "DriveDendriticInput": 180,
        "DenseDendriticInput": 180,
        "Conv2dWTASTDP": 500,
    }
)
