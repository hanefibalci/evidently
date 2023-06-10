#!/usr/bin/env python3

import argparse
import json
import os
import time
from typing import Dict

import numpy as np
import pandas as pd
import requests


# the encoder helps to convert NumPy types in source data to JSON-compatible types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.void):
            return None

        if isinstance(obj, (np.generic, np.bool_)):
            return obj.item()

        if isinstance(obj, np.ndarray):
            return obj.tolist()

        return obj


def send_data_row(dataset_name: str, data: Dict) -> None:
    print(f"Send a data item for {dataset_name}")

    try:
        response = requests.post(
            f"http://localhost:8085/iterate/{dataset_name}",
            data=json.dumps([data], cls=NumpyEncoder),
            headers={"content-type": "application/json"},
        )

        if response.status_code == 200:
            print(f"Success.")

        else:
            print(
                f"Got an error code {response.status_code} for the data chunk. "
                f"Reason: {response.reason}, error text: {response.text}"
            )

    except requests.exceptions.ConnectionError as error:
        print(f"Cannot reach a metrics application, error: {error}, data: {data}")


def main(sleep_timeout: int) -> None:
    datasets_path = os.path.abspath("datasets")
    if not os.path.exists(datasets_path):
        exit("Cannot find datasets, try to run run_example.py script for initial setup")

    print(
        f"Get production data from {datasets_path} and send the data to monitoring service each {args.timeout} seconds"
    )
    datasets = {}
    max_index = 0


    df_ref = pd.read_csv(datasets_path+"/reference.csv")
    df_prod = pd.read_csv(datasets_path+"/production.csv")

    for i in range(df_ref.shape[0]):
        data = df_ref.iloc[i].to_dict()
        print(data)
        send_data_row("reference", data)
        time.sleep(0.1)

    for i in range(df_prod.shape[0]):
        data = df_prod.iloc[i].to_dict()
        send_data_row("production", data)
        time.sleep(0.1)
    




if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script for data sending to Evidently metrics integration demo service"
    )
    parser.add_argument(
        "-t",
        "--timeout",
        type=float,
        default=2,
        help="Sleep timeout between data send tries in seconds.",
    )
    args = parser.parse_args()
    main(args.timeout)
