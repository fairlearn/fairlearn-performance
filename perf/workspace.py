# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

""" Utilities to use an Azure Machine Learning workspace to track the performance test results
over time.
"""

import logging
import os

from azureml.core import Workspace
from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.authentication import ServicePrincipalAuthentication
from azureml._base_sdk_common.common import resource_client_factory
from azure.mgmt.resource.resources.models import ResourceGroup


# Users may define their own variable names where they store the following data.
# We use VARIABLE_NAME_* to identify where they stored it, and subsequently look it up.
# The reason for these levels of indirection are mostly key restrictions:
# Azure Key Vault doesn't allow underscores, Powershell doesn't allow dashes.
# Furthermore, user-specific secrets in Key Vault should have use case specific names such as
# <use-case>-sp-id as opposed to SERVICE_PRINCIPAL_ID.
TENANT_ID = os.getenv("TENANT_ID")
SERVICE_PRINCIPAL_ID = os.getenv("SERVICE_PRINCIPAL_ID")
SERVICE_PRINCIPAL_PASSWORD = os.getenv("SERVICE_PRINCIPAL_PASSWORD")
SUBSCRIPTION_ID = os.getenv("SUBSCRIPTION_ID")
RESOURCE_GROUP_NAME = os.getenv("RESOURCE_GROUP_NAME")
WORKSPACE_NAME = os.getenv("WORKSPACE_NAME")
WORKSPACE_LOCATION = os.getenv("WORKSPACE_LOCATION")

logger = logging.getLogger(__file__)


def get_workspace(compute_name=None, compute_config=None):
    logger.info("Logging in as service principal.")
    auth = ServicePrincipalAuthentication(TENANT_ID,
                                          SERVICE_PRINCIPAL_ID,
                                          SERVICE_PRINCIPAL_PASSWORD)
    logger.info("Successfully logged in as service principal.")

    logger.info("Ensuring resource group {} exists.".format(RESOURCE_GROUP_NAME))
    resource_management_client = resource_client_factory(auth, SUBSCRIPTION_ID)
    resource_group_properties = ResourceGroup(location=WORKSPACE_LOCATION)
    resource_management_client.resource_groups.create_or_update(WORKSPACE_NAME,
                                                                resource_group_properties)
    logger.info("Ensured resource group {} exists.".format(RESOURCE_GROUP_NAME))

    logger.info("Ensuring workspace {} exists.".format(WORKSPACE_NAME))
    workspace = Workspace.create(name=WORKSPACE_NAME, auth=auth, subscription_id=SUBSCRIPTION_ID,
                                 resource_group=RESOURCE_GROUP_NAME, location=WORKSPACE_LOCATION,
                                 create_resource_group=False, exist_ok=True)
    logger.info("Ensured workspace {} exists.".format(WORKSPACE_NAME))
    logger.info("Ensuring compute exists.")
    get_or_create_compute_cluster(workspace, compute_name, compute_config)
    logger.info("Ensured compute exists.")
    return workspace


def get_or_create_compute_cluster(workspace, compute_name=None, config=None):
    if compute_name is None:
        compute_name = "cpu-cluster"
    if compute_name not in workspace.compute_targets:
        if config is None:
            config = AmlCompute.provisioning_configuration(
                vm_size='STANDARD_DS2_V2',
                vm_priority="dedicated",
                min_nodes=0,
                max_nodes=10,
                idle_seconds_before_scaledown=300
            )
        compute_target = ComputeTarget.create(workspace, compute_name, config)
        compute_target.wait_for_completion(show_output=True)
    else:
        compute_target = workspace.compute_targets[compute_name]
    return compute_target
