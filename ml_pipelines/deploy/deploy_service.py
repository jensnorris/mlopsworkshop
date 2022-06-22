#!/usr/bin/env python3
from azureml.core.model import InferenceConfig
from azureml.core import Workspace, Model
from ml_pipelines.utils import EnvironmentVariables, get_config_compute, get_environment
from azureml.core.webservice import AciWebservice
from pathlib import Path
from argparse import ArgumentParser

parser = ArgumentParser("Build Scoring Image")
parser.add_argument("--model-version", default=None)
parser.add_argument("--id", default=None)
parser.add_argument("--local", action="store_true")
parser.add_argument("--url-output", default=None)
args = parser.parse_args()

deployment_config = AciWebservice.deploy_configuration(cpu_cores=1, memory_gb=1)

workspace = Workspace.from_config()
env_vars = EnvironmentVariables()

environment = get_environment(workspace, env_vars)
inference_config = InferenceConfig(
    entry_script=env_vars.scoring_file,
    source_directory=env_vars.scoring_dir,
    environment=environment,
)

cluster = get_config_compute(workspace, env_vars)

model = Model(workspace, name='diamonds-regressor', version=None)

service = Model.deploy(
    workspace=workspace,
    name=env_vars.service_name,
    models=[model],
    inference_config=inference_config,
    deployment_config=deployment_config,
    overwrite=True,
    deployment_target=cluster
)

service.wait_for_deployment(show_output=True)
if args.url_output is not None:
    Path(args.url_output).write_text(service.scoring_uri)
    f"SCORING_URL={service.scoring_uri}\n"