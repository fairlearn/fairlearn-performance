# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os

from fairlearn.postprocessing import ThresholdOptimizer
from fairlearn.reductions import ExponentiatedGradient, GridSearch

from timed_execution import TimedExecution


_MITIGATION = "mitigation"
_ESTIMATOR_FIT = 'estimator_fit'


def generate_script(request, perf_test_configuration, script_name, script_directory, workspace):
    if not os.path.exists(script_directory):
        os.makedirs(script_directory)

    script_lines = []
    add_imports(script_lines, workspace)
    script_lines.append("")
    
    if workspace:
        script_lines.append("run = Run.get_context()")
    else:
        # patch Run class so that we print out metrics if AzureML isn't used.
        script_lines.append("class Run:")
        script_lines.append("    def log(self, msg, *args):")
        script_lines.append("        print(msg, *args)")
        script_lines.append("    def log_list(self, msg, lst):")
        script_lines.append("        print(msg, *lst)")
        script_lines.append("run = Run()")
    
    add_dataset_setup(script_lines, perf_test_configuration)
    add_unconstrained_estimator_fitting(script_lines, perf_test_configuration)
    add_mitigation(script_lines, perf_test_configuration)
    add_additional_metric_calculation(script_lines, perf_test_configuration)
    script_lines.append("")

    print(f"\n\n{'='*100}\n\n")

    full_script_name = os.path.join(script_directory, script_name)
    with open(full_script_name, 'w') as script_file:  # noqa: E501
        script_file.write("\n".join(script_lines))
    print(f"wrote script to {full_script_name}")


def add_imports(script_lines, workspace):
    script_lines.append('from time import time')
    script_lines.append('from tempeh.configurations import datasets')
    script_lines.append('from fairlearn.postprocessing import ThresholdOptimizer')
    script_lines.append('from fairlearn.reductions import ExponentiatedGradient, GridSearch')
    script_lines.append('from fairlearn.reductions import DemographicParity, EqualizedOdds, ErrorRateParity, TruePositiveRateParity, FalsePositiveRateParity')
    script_lines.append('from sklearn.svm import SVC')
    script_lines.append('from sklearn.tree import DecisionTreeClassifier')
    if workspace:
        script_lines.append('from azureml.core.run import Run')


def add_dataset_setup(script_lines, perf_test_configuration):
    script_lines.append('print("Downloading dataset")')
    script_lines.append(f'dataset = datasets["{perf_test_configuration.dataset}"]()')
    script_lines.append('X_train, X_test = dataset.get_X()')
    script_lines.append('y_train, y_test = dataset.get_y()')
    script_lines.append('print("Done downloading dataset")')

    if perf_test_configuration.dataset == "adult_uci":
        # sensitive feature is 8th column (sex)
        script_lines.append('sensitive_features_train = X_train[:, 7]')
        script_lines.append('sensitive_features_test = X_test[:, 7]')
    elif perf_test_configuration.dataset == "diabetes_sklearn":
        # sensitive feature is 2nd column (sex)
        # features have been scaled, but sensitive feature needs to be str or int
        script_lines.append('sensitive_features_train = X_train[:, 1].astype(str)')
        script_lines.append('sensitive_features_test = X_test[:, 1].astype(str)')
        # labels can't be floats as of now
        script_lines.append('y_train = y_train.astype(int)')
        script_lines.append('y_test = y_test.astype(int)')
    elif perf_test_configuration.dataset == "compas":
        # sensitive feature is either race or sex
        # TODO add another case where we use sex as well, or both (?)
        script_lines.append('sensitive_features_train, sensitive_features_test = dataset.get_sensitive_features("race")')
        script_lines.append('y_train = y_train.astype(int)')
        script_lines.append('y_test = y_test.astype(int)')
    else:
        raise ValueError(f"Sensitive features unknown for dataset {perf_test_configuration.dataset}")


def add_unconstrained_estimator_fitting(script_lines, perf_test_configuration):
    with TimedExecution(_ESTIMATOR_FIT, script_lines):
        script_lines.append(f'estimator = {perf_test_configuration.estimator}')
        script_lines.append(f'unconstrained_estimator = {perf_test_configuration.estimator}')
        script_lines.append('unconstrained_estimator.fit(X_train, y_train)')


def add_mitigation(script_lines, perf_test_configuration):
    with TimedExecution(_MITIGATION, script_lines):
        if perf_test_configuration.mitigator == ThresholdOptimizer.__name__:
            script_lines.append('mitigator = ThresholdOptimizer('
                                'estimator=unconstrained_estimator, '
                                'prefit=True, '
                                f'constraints={perf_test_configuration.disparity_metric})')
        elif perf_test_configuration.mitigator in [
                ExponentiatedGradient.__name__,
                GridSearch.__name__]:
            script_lines.append(f'mitigator = {perf_test_configuration.mitigator}('
                                'estimator=estimator, '
                                f'constraints={perf_test_configuration.disparity_metric})')
        else:
            raise Exception("Unknown mitigation technique.")

        script_lines.append('mitigator.fit(X_train, y_train, sensitive_features=sensitive_features_train)')

    if perf_test_configuration.mitigator == ThresholdOptimizer.__name__:
        script_lines.append('mitigator.predict('
                            'X_test, '
                            'sensitive_features=sensitive_features_test, '
                            'random_state=1)')
    else:
        script_lines.append('predictions = mitigator.predict(X_test)')


def add_additional_metric_calculation(script_lines, perf_test_configuration):
    # In certain mitigation methods we re-run the estimators many times.
    # For that reason we need metrics to compare the mitigation time with the time that the
    # estimators took since Fairlearn only controls the mitigation overhead and not the estimator
    # training time.
    if perf_test_configuration.mitigator in [
            ExponentiatedGradient.__name__,
            GridSearch.__name__]:
        add_metric_logging_script(script_lines, "metric_logging_script_expgrad_gridsearch.txt")
    elif perf_test_configuration.mitigator in [ThresholdOptimizer.__name__]:
        add_metric_logging_script(script_lines, "metric_logging_script_postprocessing.txt")


def add_metric_logging_script(script_lines, metric_logging_script_file_name):
    skip_lines = [
        "# Copyright (c) Microsoft Corporation. All rights reserved."
        "# Licensed under the MIT License."
    ]
    script_directory = os.path.dirname(__file__)
    with open(os.path.join(script_directory, metric_logging_script_file_name), 'r') as metric_logging_script_file:
        for line in metric_logging_script_file.readlines():
            if line not in skip_lines:
                script_lines.append(line.replace("\n", ""))
