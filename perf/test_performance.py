# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import logging
import os
import pytest

from azureml.core import Experiment, RunConfiguration, ScriptRunConfig

from conftest import get_all_perf_test_configurations
from environment_setup import configure_environment
from script_generation import generate_script

all_perf_test_configurations = get_all_perf_test_configurations()
all_perf_test_configurations_descriptions = \
    [config.__repr__().replace(' ', '') for config in all_perf_test_configurations]

SCRIPT_DIRECTORY = os.path.join('perf', 'scripts')
EXPERIMENT_NAME = "perftest"

logging.basicConfig(level=logging.DEBUG)


# ensure the tests are run from the fairlearn repository base directory
if not os.path.exists(os.path.join("fairlearn-performance", "perf")):
    raise Exception("Please run perf tests from outside the fairlearn-performance repository "
                    "base directory. Current working directory: {}".format(os.getcwd()))


@pytest.mark.parametrize("perf_test_configuration", all_perf_test_configurations,
                         ids=all_perf_test_configurations_descriptions)
@pytest.mark.perf
def test_perf(perf_test_configuration, workspace, request, wheel_file):
    print("Starting with test case {}".format(request.node.name))

    script_name = determine_script_name(request.node.name)
    generate_script(request, perf_test_configuration, script_name, SCRIPT_DIRECTORY)

    experiment = Experiment(workspace=workspace, name=EXPERIMENT_NAME)
    compute_target = workspace.compute_targets['cpu-cluster']
    run_config = RunConfiguration()
    run_config.target = compute_target

    environment = configure_environment(workspace, wheel_file=wheel_file,
                                        requirements_file=os.path.join("fairlearn", "requirements.txt"))
    run_config.environment = environment
    environment.register(workspace=workspace)
    script_run_config = ScriptRunConfig(source_directory=SCRIPT_DIRECTORY,
                                        script=script_name,
                                        run_config=run_config)
    print("submitting run")
    experiment.submit(config=script_run_config, tags=perf_test_configuration.__dict__)
    print("submitted run")


def determine_script_name(test_case_name):
    hashed_test_case_name = hash(test_case_name)
    hashed_test_case_name = hashed_test_case_name if hashed_test_case_name >= 0 \
        else -1 * hashed_test_case_name
    return "{}.py".format(str(hashed_test_case_name))
