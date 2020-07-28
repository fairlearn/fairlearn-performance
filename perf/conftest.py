# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import pytest

from fairlearn.postprocessing import ThresholdOptimizer
from fairlearn.reductions import ExponentiatedGradient, GridSearch

from workspace import get_workspace
from environment_setup import build_package


THRESHOLD_OPTIMIZER = ThresholdOptimizer.__name__
EXPONENTIATED_GRADIENT = ExponentiatedGradient.__name__
GRID_SEARCH = GridSearch.__name__

MEMORY = "memory"
TIME = "time"

ADULT_UCI = 'adult_uci'
COMPAS = 'compas'

RBM_SVM = 'SVC()'
DECISION_TREE_CLASSIFIER = 'DecisionTreeClassifier()'

DATASETS = [ADULT_UCI, COMPAS]
ESTIMATORS = [RBM_SVM, DECISION_TREE_CLASSIFIER]
MITIGATORS = [THRESHOLD_OPTIMIZER, EXPONENTIATED_GRADIENT, GRID_SEARCH]


class PerfTestConfiguration:
    def __init__(self, dataset, estimator, mitigator, disparity_metric):
        self.dataset = dataset
        self.estimator = estimator
        self.mitigator = mitigator
        self.disparity_metric = disparity_metric

    def __repr__(self):
        return "[dataset: {}, estimator: {}, mitigator: {}, disparity_metric: {}]" \
               .format(self.dataset, self.estimator, self.mitigator, self.disparity_metric)


def get_all_perf_test_configurations():
    perf_test_configurations = []
    for dataset in DATASETS:
        for estimator in ESTIMATORS:
            for mitigator in MITIGATORS:
                if mitigator == THRESHOLD_OPTIMIZER:
                    disparity_metrics = ["'equalized_odds'", "'demographic_parity'"]
                elif mitigator == EXPONENTIATED_GRADIENT:
                    disparity_metrics = ["EqualizedOdds()", "DemographicParity()", "ErrorRateParity()", "FalsePositiveRateParity()", "TruePositiveRateParity()"]
                elif mitigator == GRID_SEARCH:
                    disparity_metrics = ["EqualizedOdds()", "DemographicParity()", "ErrorRateParity()", "FalsePositiveRateParity()", "TruePositiveRateParity()"]
                else:
                    raise Exception("Unknown mitigator {}".format(mitigator))

                for disparity_metric in disparity_metrics:
                    perf_test_configurations.append(
                        PerfTestConfiguration(dataset, estimator, mitigator, disparity_metric))

    return perf_test_configurations


@pytest.fixture(scope="session")
def workspace():
    return get_workspace()


@pytest.fixture(scope="session")
def wheel_file():
    return build_package()
