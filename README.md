## Setup
Create and configure the environment:
```bash
conda env create -f environment.yml
```

Activate the environment:
```bash
conda activate minimal_walking
```

Train:
```bash
python genesis_playground/zbot/zbot_train.py -e zbot-walking --num_envs 4096
```