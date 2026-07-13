#!/usr/bin/env python3

from itertools import combinations
from pathlib import Path

import pandas as pd


DATASET_BASE = "/homes/j244s673/documents/wsu/phd/dataset"

datasets = [
    {
        "Dataset": "EARTHQUAKE-TURKEY",
        "Folder": "EARTHQUAKE-TURKEY",
        "Path": f"{DATASET_BASE}/EARTHQUAKE-TURKEY",
    },
    {
        "Dataset": "HURRICANE-DELTA",
        "Folder": "HURRICANE-DELTA",
        "Path": f"{DATASET_BASE}/HURRICANE-DELTA",
    },
    {
        "Dataset": "HURRICANE-IAN",
        "Folder": "HURRICANE-IAN",
        "Path": f"{DATASET_BASE}/HURRICANE-IAN",
    },
    {
        "Dataset": "HURRICANE-LAURA",
        "Folder": "HURRICANE-LAURA",
        "Path": f"{DATASET_BASE}/HURRICANE-LAURA",
    },
    {
        "Dataset": "MOUNT-SEMERU-ERUPTION",
        "Folder": "MOUNT-SEMERU-ERUPTION",
        "Path": f"{DATASET_BASE}/MOUNT-SEMERU-ERUPTION",
    },
    {
        "Dataset": "STVINCENT-VOLCANO",
        "Folder": "STVINCENT-VOLCANO",
        "Path": f"{DATASET_BASE}/STVINCENT-VOLCANO",
    },
    {
        "Dataset": "TEXAS-TORNADOES",
        "Folder": "TEXAS-TORNADOES",
        "Path": f"{DATASET_BASE}/TEXAS-TORNADOES",
    },
    {
        "Dataset": "TONGA-VOLCANO",
        "Folder": "TONGA-VOLCANO",
        "Path": f"{DATASET_BASE}/TONGA-VOLCANO",
    },
    {
        "Dataset": "xBD",
        "Folder": "xBD",
        "Path": f"{DATASET_BASE}/xBD",
    },
]


transfer_modes = [
    {
        "Transfer_Learning": "No",
        "Training_Setup": "Train HRTBDA from scratch on Dataset A + Dataset B",
        "Initialization": "Random initialization",
        "Pretrain_Source": "None",
    },
    {
        "Transfer_Learning": "Yes",
        "Training_Setup": "Initialize HRTBDA using xBD-pretrained weights, then fine-tune on Dataset A + Dataset B",
        "Initialization": "xBD-pretrained HRTBDA checkpoint",
        "Pretrain_Source": "xBD",
    },
]


rows = []
exp_id = 1

dataset_lookup = {d["Dataset"]: d for d in datasets}

for train_a, train_b in combinations(datasets, 2):
    for test_c in datasets:
        for mode in transfer_modes:
            train_combo = f"{train_a['Dataset']} + {train_b['Dataset']}"
            test_dataset = test_c["Dataset"]

            if test_dataset in [train_a["Dataset"], train_b["Dataset"]]:
                evaluation_type = "In-domain or mixed-domain evaluation"
            else:
                evaluation_type = "Cross-disaster generalization"

            if mode["Transfer_Learning"] == "Yes":
                mode_tag = "TL"
                output_name = (
                    f"HRTBDA_TL_xBDPretrain_Train_{train_a['Dataset']}_PLUS_{train_b['Dataset']}"
                    f"_Test_{test_dataset}"
                )
            else:
                mode_tag = "Scratch"
                output_name = (
                    f"HRTBDA_Scratch_Train_{train_a['Dataset']}_PLUS_{train_b['Dataset']}"
                    f"_Test_{test_dataset}"
                )

            rows.append(
                {
                    "Experiment_ID": f"HRTBDA_COMBO_{exp_id:04d}_{mode_tag}",
                    "Model": "HRTBDA",
                    "Model_Full_Name": "High Resolution Transformer Building Damage Assessment",
                    "Transfer_Learning": mode["Transfer_Learning"],
                    "Training_Setup": mode["Training_Setup"],
                    "Initialization": mode["Initialization"],
                    "Pretrain_Source": mode["Pretrain_Source"],
                    "Train_Dataset_A": train_a["Dataset"],
                    "Train_Dataset_B": train_b["Dataset"],
                    "Train_Combination": train_combo,
                    "Test_Dataset_C": test_dataset,
                    "Evaluation_Type": evaluation_type,
                    "Train_A_Path": train_a["Path"],
                    "Train_B_Path": train_b["Path"],
                    "Test_C_Path": test_c["Path"],
                    "Train_Splits": "train",
                    "Validation_Splits": "val from Dataset A + val from Dataset B",
                    "Test_Split": "test from Dataset C",
                    "Output_Folder_Name": output_name,
                    "Phase_I_Setup": "Train/fine-tune building localization stage on Dataset A + Dataset B",
                    "Phase_II_Setup": "Train/fine-tune damage classification stage on Dataset A + Dataset B",
                    "Final_Testing": "Use trained HRTBDA cascade and evaluate on Dataset C test set",
                    "Localization_F1": "",
                    "No_Damage_F1": "",
                    "Minor_Damage_F1": "",
                    "Major_Damage_F1": "",
                    "Destroyed_F1": "",
                    "Damage_Macro_F1": "",
                    "Overall_Score": "",
                    "Run_Status": "Not started",
                    "Notes": "",
                }
            )

            exp_id += 1


df_datasets = pd.DataFrame(datasets)
df_experiments = pd.DataFrame(rows)

summary_rows = [
    {"Item": "Model", "Value": "HRTBDA"},
    {"Item": "Full model name", "Value": "High Resolution Transformer Building Damage Assessment"},
    {"Item": "Number of datasets", "Value": len(datasets)},
    {"Item": "Number of train dataset pairs", "Value": len(list(combinations(datasets, 2)))},
    {"Item": "Number of test datasets", "Value": len(datasets)},
    {"Item": "Transfer modes", "Value": 2},
    {"Item": "Total experiments", "Value": len(df_experiments)},
    {
        "Item": "Experiment design",
        "Value": "Train on Dataset A + Dataset B and test on Dataset C, repeated with and without xBD transfer learning.",
    },
]

df_summary = pd.DataFrame(summary_rows)

output_path = Path("HRTBDA_all_dataset_combination_experiment_plan.xlsx")

with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
    df_summary.to_excel(writer, index=False, sheet_name="README")
    df_datasets.to_excel(writer, index=False, sheet_name="Datasets")
    df_experiments.to_excel(writer, index=False, sheet_name="All_Experiments")

    workbook = writer.book

    for sheet_name in ["README", "Datasets", "All_Experiments"]:
        ws = workbook[sheet_name]
        ws.freeze_panes = "A2"

        for column_cells in ws.columns:
            max_length = 0
            column_letter = column_cells[0].column_letter

            for cell in column_cells:
                value = str(cell.value) if cell.value is not None else ""
                max_length = max(max_length, len(value))

            ws.column_dimensions[column_letter].width = min(max_length + 2, 55)

        ws.auto_filter.ref = ws.dimensions

print(f"Wrote Excel file: {output_path.resolve()}")
print(f"Total experiments: {len(df_experiments)}")
