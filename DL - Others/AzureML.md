
Centermask Model to AzureML

1. /AIModels/train_centermask_tim.cmd  (1)

   EXPERIMENT_NAME

   ENVIRONMENT_NAME=/AIModels/env_support/centermask_core_local_env.yml (2)

    --> setup python environment

   WORKSPACE_CONFIG=/AIModels/azure_support/config_ai_confluence.json (3)

                    --> setup subscription_id, resource_group, workspace_name

   SCRIPT_CONFIG=/AIModels/Centermask/config_script_aml_tim.json (4)

                    --> run /CenterMask/centerMask_aml_train.py

{

 "name": "./CenterMask/centerMask_aml_train.py", (5)

 "arguments": {

     "--config-file":"./CenterMask/CenterWrapRoot/configs/cellseg/panoptic_phase_v3_aml.yaml", (6)

     "--num-gpus": "4"

  }

}

   COMPUTE_TYPE="local"

   AUTH_TYPE="interactive"

   COMPUTE_NAME="clu-4-k80-tj-1"

   WHEELS_DIR="./CenterMask/wheels/"

python azure_support/azureModelWrapper.py (7)

 --experiment_name %EXPERIMENT_NAME%

 --env_name %ENVIRONMENT_NAME%

 --workspace_config %WORKSPACE_CONFIG%

 --script_config %SCRIPT_CONFIG%

--compute_type %COMPUTE_TYPE%

--auth_type %AUTH_TYPE%

--custom_wheels_dir %WHEELS_DIR%

--compute_target %COMPUTE_NAME%

DINOv2 Model to AzureML

1. /dinov2_docker/train_dinov2_aml.cmd  (1)

   EXPERIMENT_NAME

   ENVIRONMENT_NAME=/dinov2_docker/env_support/dinov2_core_env.yml (2)

    --> setup python environment

   WORKSPACE_CONFIG=/dinov2_docker/azure_support/config_workspace_dinov2.json (3)

                    --> setup subscription_id, resource_group, workspace_name

   SCRIPT_CONFIG=/dinov2_dockers/dinov2_support/config_script_aml.json (4)

                    --> run /dinov2_support/dinov2_aml_train.py

{

 "name": "./dinov2_support/dinov2_aml_train.py", (5)

 "arguments": {

     "--config-file":"./dinov2_support/config_dinov2_model.yaml", (6)

     "--num-gpus": "4"

  }

}

   COMPUTE_TYPE="local"

   AUTH_TYPE="interactive"

   COMPUTE_NAME="clu-4-k80-tj-1"

   WHEELS_DIR="./dinov2_support/wheels/"

python azure_support/azureModelWrapper.py (7)

 --experiment_name %EXPERIMENT_NAME%

 --env_name %ENVIRONMENT_NAME%

 --workspace_config %WORKSPACE_CONFIG%

 --script_config %SCRIPT_CONFIG%

--compute_type %COMPUTE_TYPE%

--auth_type %AUTH_TYPE%

--custom_wheels_dir %WHEELS_DIR%

--compute_target %COMPUTE_NAME%