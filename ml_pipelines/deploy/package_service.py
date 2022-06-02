#!/usr/bin/env python3
from azureml.core.model import InferenceConfig
from azureml.core import Workspace, Model
from ml_pipelines.utils import EnvironmentVariables, get_environment

workspace = Workspace.from_config()
env_vars = EnvironmentVariables()

environment = get_environment(workspace, env_vars)
inference_config = InferenceConfig(
    entry_script=env_vars.scoring_file,
    source_directory=env_vars.scoring_dir,
    environment=environment,
)

# No model here yet!!!
package = Model.package(
    workspace, models=[], inference_config=inference_config, generate_dockerfile=True
)
package.wait_for_creation(show_output=True)

# Save the dockerfile and related resources here
package.save("./imagefiles")

# Get the Azure container registry that the model/Dockerfile uses.
acr = package.get_container_registry()

# Give the user info on how they can build the service locally
info_string = f"""
Build and start the docker by running:
echo  {acr.password} | docker login {acr.address} -u {acr.username} --password-stdin
docker build --tag my-service-container ./imagefiles
docker run --rm -p 6789:5001 --name mycontainer my-service-container

Send request to the service on:
http://localhost:6789/score

stop the service by running this in a new terminal:
docker stop mycontainer"""
print(info_string)
