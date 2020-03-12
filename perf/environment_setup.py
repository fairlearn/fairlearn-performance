# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from azureml.core import Environment
import os
import shutil
import subprocess
import time


def build_package():
    print('removing build directory')
    shutil.rmtree(os.path.join("fairlearn", "build"), True)
    print('removing fairlearn.egg-info')
    shutil.rmtree(os.path.join("fairlearn", "fairlearn.egg-info"), True)
    print('removing dist directory')
    shutil.rmtree(os.path.join("fairlearn", "dist"), True)

    print('running python setup.py bdist_wheel')
    subprocess.Popen(["python", "setup.py", "bdist_wheel"],
                     cwd=os.path.join(os.getcwd(), "fairlearn"),
                     shell=True).wait()
    for root, dirs, files in os.walk(os.path.join("fairlearn", "dist")):
        for file_ in files:
            if file_.endswith(".whl"):
                print("Found wheel {}".format(file_))
                # change wheel name to be unique for every run
                src = os.path.join("fairlearn", "dist", file_)
                dst = os.path.join("fairlearn", "dist", "fairlearn-v{}-py3-none-any.whl"
                                   .format(time.time()))
                shutil.copy(src, dst)
                return dst

    raise Exception("Couldn't find wheel file.")


def configure_environment(workspace, wheel_file=None, requirements_file=None):
    # collect external requirements from requirements file
    if requirements_file is None:
        requirements_file = 'requirements.txt'
    environment = Environment.from_pip_requirements(name="env", file_path=requirements_file)

    # add private pip wheel to blob if provided
    if wheel_file:
        private_pkg = environment.add_private_pip_wheel(workspace, file_path=wheel_file,
                                                        exist_ok=True)
        environment.python.conda_dependencies.add_pip_package(private_pkg)

    # add azureml-sdk to log metrics
    environment.python.conda_dependencies.add_pip_package("azureml-sdk")

    # set docker to enabled for AmlCompute
    environment.docker.enabled = True
    print("environment successfully configured")
    return environment
