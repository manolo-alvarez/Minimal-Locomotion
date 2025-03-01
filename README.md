## First Time Setup
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

## Updating Environment
You'll first want to pull the latest changes with:
```bash
git pull
```
Then, you'll want to update the environment to be compatible with the pulled code:
```bash
conda env update --file environment.yml --prune
```