#!/usr/bin/env python3
#hanefi balcı
import argparse
import io
import logging
import os
import shutil
import zipfile
from typing import Tuple

import pandas as pd
import requests

# suppress SettingWithCopyWarning: warning
pd.options.mode.chained_assignment = None


BIKE_DATA_SOURCE_URL = "https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip"


def setup_logger() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", handlers=[logging.StreamHandler()]
    )


def get_data_telco_churn() -> Tuple[pd.DataFrame, pd.DataFrame]:
    from sklearn import model_selection
    from sklearn import neighbors
    print("telco metoduna girdi")

    data = pd.read_csv("telco_churn.csv")
    features = list(set(data.columns) - {"Churn"})
    
    reference_data, production_data = model_selection.train_test_split(
        data, random_state=0, train_size=0.2, test_size=0.1
    )
    target = "Churn"

    classification_model = neighbors.KNeighborsClassifier(n_neighbors=1)
    classification_model.fit(reference_data[features], reference_data[target])

    reference_data["prediction"] = classification_model.predict(reference_data[features])
    production_data["prediction"] = classification_model.predict(production_data[features])
    print("tahminler yapıldı")
    return reference_data[features + [target, "prediction"]], production_data


def main(dataset_name: str, dataset_path: str) -> None:
    logging.info("Generate test data for dataset %s", dataset_name)
    dataset_path = os.path.abspath(dataset_path)
    logging.info("Maine girdik")
 #   if os.path.exists(dataset_path):
#        logging.info("Path %s already exists, remove it", dataset_path)
    #    shutil.rmtree(dataset_path)

    #os.makedirs(dataset_path)

    reference_data, production_data = DATA_SOURCES[dataset_name]()
    logging.info("Save datasets to %s", dataset_path)
    reference_data.to_csv(os.path.join(dataset_path, "reference.csv"), index=False)
    production_data.to_csv(os.path.join(dataset_path, "production.csv"), index=False)

    logging.info("Reference dataset was created with %s rows", reference_data.shape[0])
    logging.info("Production dataset was created with %s rows", production_data.shape[0])


DATA_SOURCES = {
    "telco_churn": get_data_telco_churn,
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script for data and config generation for demo Evidently metrics integration with Grafana"
    )
    parser.add_argument(
        "-d",
        "--dataset",
        choices=DATA_SOURCES.keys(),
        type=str,
        help="Dataset name for reference.csv= and production.csv files generation.",
    )
    parser.add_argument(
        "-p",
        "--path",
        type=str,
        help="Path for saving dataset files.",
    )

    args = parser.parse_args()
    setup_logger()
    if args.dataset not in DATA_SOURCES:
        exit(f"Incorrect dataset name {args.dataset}, try to see correct names with --help")
    main(args.dataset, args.path)
#hanefi