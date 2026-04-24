import json
from pathlib import Path
import pandas as pd


def save_summary_json(summary_dict, save_path):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    with open(save_path, "w") as f:
        json.dump(summary_dict, f, indent=2)


def load_summary_json(load_path):
    load_path = Path(load_path)

    with open(load_path, "r") as f:
        return json.load(f)


def save_dataframe_csv(df, save_path):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(save_path, index=False)


def load_dataframe_csv(load_path):
    load_path = Path(load_path)
    return pd.read_csv(load_path)