from setuptools import setup, find_packages

setup(
    name="house_prediction",
    version="1.0.0",
    author="Agrim Agarwal",
    packages=find_packages(),
    install_requires=[
        "pandas",
        "scikit-learn",
        "mlflow"
    ]
)