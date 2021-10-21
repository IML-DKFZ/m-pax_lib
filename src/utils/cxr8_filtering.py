import pandas as pd
import numpy as np

#### OOD Test Data ####

csv_path = "./data/CXR8/CheXpert-v1.0-small/train.csv"
targets = pd.read_csv(csv_path)
targets = targets[targets["Frontal/Lateral"] == "Frontal"]
targets2 = pd.read_csv(csv_path)
targets2 = targets2[targets2["Frontal/Lateral"] == "Frontal"]


targets = targets[
    [
        "Path",
        "Atelectasis",
        "Cardiomegaly",
        "Pneumonia",
        "Pneumothorax",
        "Consolidation",
        "Edema",
    ]
]

targets = targets.fillna(0)

mask = np.logical_or(
    np.logical_or(
        np.array(np.sum(targets2.iloc[:, [6, 8, 9, 15, 16, 17]], axis=1) >= 1),
        np.array(np.sum(targets2.iloc[:, [7, 10, 11, 12, 13, 14]], axis=1) > 1),
    ),
    np.array(targets2.iloc[:, [7, 10, 11, 12, 13, 14]] == -1).any(axis=1),
)

targets[~mask].to_csv(
    "./data/CXR8/CheXpert-v1.0-small/train_filtered.csv", sep=",", index=None
)

#### Train/Val/Test Data ####
datasets = ["train", "val", "test"]
for set in datasets:
    csv_path = "./data/CXR8/" + set + "_label.csv"
    targets = pd.read_csv(csv_path)

    mask = np.logical_or(
        np.array(
            np.sum(
                targets.iloc[
                    :,
                    [3, 4, 5, 10, 11, 12, 13, 14],
                ],
                axis=1,
            )
            >= 1
        ),
        np.array(
            np.sum(
                targets.iloc[
                    :,
                    [1, 2, 6, 7, 8, 9],
                ],
                axis=1,
            )
            > 1
        ),
    )

    targets[~mask][
        [
            "FileName",
            "Atelectasis",
            "Cardiomegaly",
            "Pneumonia",
            "Pneumothorax",
            "Consolidation",
            "Edema",
        ]
    ].to_csv("./data/CXR8/" + set + "_label_filtered.csv", sep=",", index=None)
