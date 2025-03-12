from setuptools import setup, find_packages

setup(
    name="tdmpc2",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=1.13.0",
        "numpy>=1.20.0",
        "tensordict",
        "gym",
        "hydra-core",
        "termcolor",
    ],
    author="TDMPC2 Authors",
    author_email="",
    description="TD-MPC2 Reinforcement Learning Framework",
    keywords="reinforcement-learning, model-predictive-control",
    python_requires=">=3.7",
)