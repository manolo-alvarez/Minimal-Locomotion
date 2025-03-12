## First Time Setup
Initialize the git submodules:
```bash
git submodule update --init --recursive
```
Set channel priority to flexible to resolve package conflicts
``` bash
conda config --set channel_priority flexible
```

If on MacOS, create and configure the environment with:
```bash
conda env create -f environment_macos.yml
```
If on Linux, create and configure the environment with:
```bash
conda env create -f environment_linux.yml
```

Activate the environment:
```bash
conda activate minimal_walking
```

## Updating Environment
You'll want to pull the latest changes every time a teammate commits them. This way you are building off of the latest code and avoid merge conflicts. To do this, run:
```bash
git pull
```
As code is developed, new packages are likley to be added to support the new functionality, so you'll want to update the environment to be compatible with the pulled code by running:
```bash
conda env update --file environment.yml --prune
```

## Training
To kickoff a locomotion policy with the default params, run:
```bash
python genesis_playground/zbot/zbot_train.py -e zbot-walking
```

Checkout the available argument flags to configure the run in main()

## Evaluation
To evaluate, analyze, and visualize the policy you've trained with the default params, run:
```bash
python genesis_playground/zbot/zbot_eval.py -e zbot-walking --analyze --show_viewer
```
Checkout the available argument flags to configure the evaluation in main()

## Real To Sim
To run a policy in the `zbot_unit_test` repo, you'll need to convert the policy to an onnx model. 

```bash
python genesis_playground/zbot/export_model.py --input_dir <path_to_checkpoint_dir> --output_dir <path_to_output_dir> --checkpoint <checkpoint_number>
```

## Debugging
Use, change, or add to the `launch.json` configuration in the `.vscode` folder to debug your session in vscode. You can step, jump, watch, etc., with this function.
