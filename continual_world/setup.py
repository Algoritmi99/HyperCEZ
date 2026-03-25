from setuptools import find_packages, setup

# Required dependencies
required = [
    "tensorflow>=2.0",
    "mujoco",
    "metaworld",
    "pandas",
    "matplotlib",
    "seaborn",
]
setup(
    name="continualworld",
    description="Continual World: A Robotic Benchmark For Continual Reinforcement Learning",
    packages=find_packages(),
    include_package_data=True,
    install_requires=required,
)
