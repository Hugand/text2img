#!/usr/bin/env python3

import glob
import os
import pprint
import traceback

import click
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


# Extraction function
def tflog2pandas(path: str) -> pd.DataFrame:
    """convert single tensorflow log file to pandas DataFrame

    Parameters
    ----------
    path : str
        path to tensorflow log file

    Returns
    -------
    pd.DataFrame
        converted dataframe
    """
    DEFAULT_SIZE_GUIDANCE = {
        "compressedHistograms": 1,
        "images": 1,
        "scalars": 0,  # 0 means load all
        "histograms": 1,
    }
    runlog_data = pd.DataFrame({"metric": [], "value": [], "step": []})
    try:
        event_acc = EventAccumulator(path, DEFAULT_SIZE_GUIDANCE)
        event_acc.Reload()
        tags = event_acc.Tags()["scalars"]
        for tag in tags:
            event_list = event_acc.Scalars(tag)
            values = list(map(lambda x: x.value, event_list))
            step = list(map(lambda x: x.step, event_list))
            r = {"metric": [tag] * len(step), "value": values, "step": step}
            r = pd.DataFrame(r)
            runlog_data = pd.concat([runlog_data, r])
    # Dirty catch of DataLossError
    except Exception:
        print("Event file possibly corrupt: {}".format(path))
        traceback.print_exc()
    return runlog_data


def many_logs2pandas(event_paths):
    all_logs = pd.DataFrame()
    for path in event_paths:
        log = tflog2pandas(path)
        if log is not None:
            if all_logs.shape[0] == 0:
                all_logs = log
            else:
                # all_logs = all_logs.append(log, ignore_index=True)
                all_logs = pd.concat([all_logs, pd.DataFrame(log)], ignore_index=True)
    return all_logs

def main():
    write_csv = True
    write_pkl = False
    out_dir = "./"
    event_paths = glob.glob(os.path.join("lightning_logs/version_154/", "event*"))

    # Call & append
    if event_paths:
        all_logs = many_logs2pandas(event_paths)
        os.makedirs(out_dir, exist_ok=True)
        if write_csv:
            print("saving to csv file")
            out_file = os.path.join(out_dir, "all_training_logs_in_one_file.csv")
            print(out_file)
            all_logs.to_csv(out_file, index=None)
        if write_pkl:
            print("saving to pickle file")
            out_file = os.path.join(out_dir, "all_training_logs_in_one_file.pkl")
            print(out_file)
            all_logs.to_pickle(out_file)
    else:
        print("No event paths have been found.")


if __name__ == "__main__":
    main()