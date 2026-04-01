TASK_SEQS = {
    "CW10": [
        "hammer-v3",
        "push-wall-v3",
        "faucet-close-v3",
        "push-back-v3",
        "stick-pull-v3",
        "handle-press-side-v3",
        "push-v3",
        "shelf-place-v3",
        "window-close-v3",
        "peg-unplug-side-v3",
    ],
}

TASK_SEQS["CW20"] = TASK_SEQS["CW10"] + TASK_SEQS["CW10"]
