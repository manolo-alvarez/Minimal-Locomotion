# MAC vs Linux

eval "$(micromamba shell hook --shell zsh)"
source ~/.zshrc

micromamba activate genesis


micromamba create -n genesis python=3.12

conda create -n genesis python=3.12

cd sim/genesis

Install rsl_rl

git clone https://github.com/leggedrobotics/rsl_rl
cd rsl_rl && git checkout v1.0.2 && pip install -e .

Install TensorBoard

pip install tensorboard



python zbot_train.py


python zeroth_eval.py


python examples/locomotion_gpr/gpr_train.py -e gpr-walking -B 5
python examples/locomotion_gpr/gpr_eval.py -e gpr-walking -v --ckpt 100