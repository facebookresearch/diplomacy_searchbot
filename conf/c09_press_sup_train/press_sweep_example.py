"""
This sample script runs a sweep over 15 different learning rates in the range of 1e-2 to 1e-6 on nodes with 8 gpus each
See heyhi/README.md section on `Running group runs` on instructions to run this script.
"""

import getpass
import json
import pathlib

import numpy as np
import pandas as pd

import heyhi.gsheets
import heyhi.run
import run

THIS_FILE = pathlib.Path(__file__).resolve()
THIS_DIR = THIS_FILE.parent


def loguniform(rng, low, high):
    low, high = min(low, high), max(low, high)
    return np.exp(rng.uniform(np.log(low), np.log(high)))


def redefines_from_dict(redefines):
    chunks = []
    for k, v in redefines.items():
        chunks.append(f"{k}={v}")
    return chunks


def yield_sweep():
    cfg = THIS_DIR / "press_sl.prototxt"
    size = 15
    for i in range(size):
        rng = np.random.RandomState(i)
        redefines = {}
        redefines["I.launcher"] = "slurm_8gpus"
        redefines["dipnet_train_params.lr"] = loguniform(rng, 1e-2, 1e-6)
        yield cfg, redefines


def get_logs_data(handle):
    valid_columns = [
        "valid_loss",
        "valid_p_loss",
        "valid_v_loss",
        "valid_p_accuracy",
        "valid_v_accuracy",
    ]
    metrics = {k: None for k in ["epoch", "train_loss"] + valid_columns}
    result_path = handle.exp_path / "metrics.jsonl"
    if not result_path.exists():
        return metrics
    data = []
    with result_path.open() as stream:
        for line in stream:
            try:
                record = json.loads(line)
            except:
                break
            data.append(record)
    data = pd.DataFrame(data)

    if "epoch" in data.columns:
        metrics["epoch"] = data["epoch"].values.max()
        metrics["train_loss"] = data["loss"].dropna().values.min()

        for col in valid_columns:
            if col in data.columns:
                if "_loss" in col:
                    metrics[col] = data[col].dropna().values.min()
                else:
                    metrics[col] = data[col].dropna().values.max()

    return metrics


def main(mode, use_gsheet):
    handles = []
    sheet_file = "parldipnet/%s" % THIS_DIR.name
    sheet_name = "%s" % (THIS_FILE.name.split(".")[0])
    exp_id = "%s_%s/sweep" % (THIS_DIR.name, sheet_name)
    for cfg, override_dict in yield_sweep():
        h = heyhi.run.maybe_launch(
            run.main,
            exp_root=heyhi.run.get_exp_dir(heyhi.run.PROJECT_NAME),
            overrides=redefines_from_dict(override_dict),
            cfg=cfg,
            mode=mode,
            exp_id_pattern_override=f"%(prefix)s/{exp_id}_%(redefines)s_%(redefines_hash)s",
        )
        handles.append((h, override_dict))
    df = []
    col_order = {}
    for (h, od) in handles:
        datum = {}
        datum["status"] = str(h.get_status())
        datum.update(get_logs_data(h))
        datum.update(od)
        col_order.update(datum)
        # Not adding these feilds to col_order to out them at the end.
        datum["exp_id"] = h.exp_id.split("/")[-1].replace("sweep@launcher@slurm@", "")
        datum["job_id"] = h.maybe_get_job_id()
        datum["folder"] = h.exp_path
        df.append(datum)
    if df:
        # Puting meta fields to the col order.
        col_order.update(df[0])
        df = pd.DataFrame(df, columns=list(col_order))
        if "valid_loss" in df.columns:
            df = df.sort_values("valid_loss")
        if use_gsheet:
            heyhi.gsheets.save_pandas_table(df, sheet_file, sheet_name, offset=1)
        del df["folder"]
        print(df.head(50).to_string(index=False))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=heyhi.MODES, default="gentle_start")
    parser.add_argument("--use_gsheet", type=int, default=int("yolo" == getpass.getuser()))
    main(**vars(parser.parse_args()))
