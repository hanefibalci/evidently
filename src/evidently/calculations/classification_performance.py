from typing import Dict, Union, Optional, List
from typing import List

import dataclasses
import numpy as np
import pandas as pd
from dataclasses import dataclass
from numpy import dtype
from pandas.core.dtypes.common import is_float_dtype, is_string_dtype, is_object_dtype

from evidently import ColumnMapping


@dataclass
class ConfusionMatrix:
    labels: List[str]
    values: list


def calculate_confusion_by_classes(confusion_matrix: pd.DataFrame, class_names: List[str]) -> Dict[str, Dict[str, int]]:
    """Calculate metrics
        TP (true positive)
        TN (true negative)
        FP (false positive)
        FN (false negative)
    for each class from confusion matrix.

    Returns a dict like:
    {
        "class_1_name": {
            "tp": 1,
            "tn": 5,
            "fp": 0,
            "fn": 3,
        },
        ...
    }
    """
    true_positive = np.diag(confusion_matrix)
    false_positive = confusion_matrix.sum(axis=0) - np.diag(confusion_matrix)
    false_negative = confusion_matrix.sum(axis=1) - np.diag(confusion_matrix)
    true_negative = confusion_matrix.sum() - (false_positive + false_negative + true_positive)
    confusion_by_classes = {}

    for idx, class_name in enumerate(class_names):
        confusion_by_classes[str(class_name)] = {
            "tp": true_positive[idx],
            "tn": true_negative[idx],
            "fp": false_positive[idx],
            "fn": false_negative[idx],
        }

    return confusion_by_classes


def k_probability_threshold(prediction_probas: pd.DataFrame, k: Union[int, float]) -> float:
    probas = prediction_probas.iloc[:, 0].sort_values(ascending=False)
    if isinstance(k, float):
        if k < 0.0 or k > 1.0:
            raise ValueError(f"K should be in range [0.0, 1.0] but was {k}")
        return probas.iloc[max(int(np.ceil(k * prediction_probas.shape[0])) - 1, 0)]
    if isinstance(k, int):
        return probas.iloc[min(k, prediction_probas.shape[0] - 1)]
    raise ValueError(f"K has unexpected type {type(k)}")


@dataclasses.dataclass
class PredictionData:
    predictions: pd.Series
    prediction_probas: Optional[pd.DataFrame]
    labels: List[Union[str, int]]


def get_prediction_data(data: pd.DataFrame, mapping: ColumnMapping, threshold: float = 0.5) -> PredictionData:
    """Get predicted values and optional prediction probabilities from source data.
    Also take into account a threshold value - if a probability is less than the value, do not take it into account.

    Return and object with predicted values and an optional prediction probabilities.
    """
    # binary or multiclass classification
    # for binary prediction_probas has column order [pos_label, neg_label]
    # for multiclass classification return just values and probas
    if isinstance(mapping.prediction, list) and len(mapping.prediction) > 2:
        # list of columns with prediction probas, should be same as target labels
        return PredictionData(
            predictions=data[mapping.prediction].idxmax(axis=1),
            prediction_probas=data[mapping.prediction],
            labels=mapping.prediction,
        )

    # calculate labels as np.array - for better negative label calculations for binary classification
    if mapping.target_names is not None:
        # if target_names is specified, get labels from it
        labels = np.array(mapping.target_names)

    else:
        # if target_names is not specified, try to get labels from target and/or prediction
        if isinstance(mapping.prediction, str) and not is_float_dtype(data[mapping.prediction]):
            # if prediction is not probas, get unique values from it and target
            labels = np.union1d(data[mapping.target].unique(), data[mapping.prediction].unique())

        else:
            # if prediction is probas, get unique values from target only
            labels = data[mapping.target].unique()

    # binary classification
    # prediction in mapping is a list of two columns:
    # one is positive value probabilities, second is negative value probabilities
    if isinstance(mapping.prediction, list) and len(mapping.prediction) == 2:
        pos_label = _check_pos_labels(mapping.pos_label, labels)

        # get negative label for binary classification
        neg_label = labels[labels != mapping.pos_label][0]

        predictions = threshold_probability_labels(data[mapping.prediction], pos_label, neg_label, threshold)
        return PredictionData(
            predictions=predictions,
            prediction_probas=data[[pos_label, neg_label]],
            labels=[pos_label, neg_label],
        )

    # binary classification
    # target is strings or other values, prediction is a string with positive label name, one column with probabilities
    if (
        isinstance(mapping.prediction, str)
        and (is_string_dtype(data[mapping.target]) or is_object_dtype(data[mapping.target]))
        and is_float_dtype(data[mapping.prediction])
    ):
        pos_label = _check_pos_labels(mapping.pos_label, labels)

        if mapping.prediction not in labels:
            raise ValueError(
                "No prediction for the target labels were found. "
                "Consider to rename columns with the prediction to match target labels."
            )

        # get negative label for binary classification
        neg_label = labels[labels != pos_label][0]

        if pos_label == mapping.prediction:
            pos_preds = data[mapping.prediction]

        else:
            pos_preds = data[mapping.prediction].apply(lambda x: 1.0 - x)

        prediction_probas = pd.DataFrame.from_dict(
            {
                pos_label: pos_preds,
                neg_label: pos_preds.apply(lambda x: 1.0 - x),
            }
        )
        predictions = threshold_probability_labels(prediction_probas, pos_label, neg_label, threshold)
        return PredictionData(
            predictions=predictions,
            prediction_probas=prediction_probas,
            labels=[pos_label, neg_label],
        )

    # binary target and preds are numbers and prediction is a label
    if not isinstance(mapping.prediction, list) and mapping.prediction in [0, 1, "0", "1"] and mapping.pos_label == 0:
        if mapping.prediction in [0, "0"]:
            pos_preds = data[mapping.prediction]
        else:
            pos_preds = data[mapping.prediction].apply(lambda x: 1.0 - x)
        predictions = pos_preds.apply(lambda x: 0 if x >= threshold else 1)
        prediction_probas = pd.DataFrame.from_dict(
            {
                0: pos_preds,
                1: pos_preds.apply(lambda x: 1.0 - x),
            }
        )
        return PredictionData(
            predictions=predictions,
            prediction_probas=prediction_probas,
            labels=[0, 1],
        )

    # binary target and preds are numbers
    elif (
        isinstance(mapping.prediction, str)
        and data[mapping.target].dtype == dtype("int")
        and data[mapping.prediction].dtype == dtype("float")
    ):
        predictions = (data[mapping.prediction] >= threshold).astype(int)
        prediction_probas = pd.DataFrame.from_dict(
            {
                1: data[mapping.prediction],
                0: data[mapping.prediction].apply(lambda x: 1.0 - x),
            }
        )
        return PredictionData(predictions=predictions, prediction_probas=prediction_probas)

    # for other cases return just prediction values, probabilities are None by default
    return PredictionData(
        predictions=data[mapping.prediction],
        prediction_probas=None,
        labels=data[mapping.prediction].unique().tolist()
    )


def _check_pos_labels(pos_label: Optional[Union[str, int]], labels: List[str]) -> Union[str, int]:
    if pos_label is None:
        raise ValueError("Undefined pos_label.")

    if pos_label not in labels:
        raise ValueError(f"Cannot find pos_label '{pos_label}' in labels {labels}")

    return pos_label


def threshold_probability_labels(
    prediction_probas: pd.DataFrame, pos_label: Union[str, int], neg_label: Union[str, int], threshold: float
) -> pd.Series:
    """Get prediction values by probabilities with the threshold apply"""
    return prediction_probas[pos_label].apply(lambda x: pos_label if x >= threshold else neg_label)
