import os
import subprocess

from setuptools import setup
from setuptools.command.build_ext import build_ext


class CMakeBuild(build_ext):
    def run(self):
        # Compile the C++ code
        if not os.path.exists("graph_gen_gym/orca"):
            subprocess.check_call(
                [
                    "g++",
                    "-O2",
                    "-std=c++11",
                    "-o",
                    "graph_gen_gym/orca",
                    "orca_src/orca.cpp",
                ]
            )
        build_ext.run(self)


setup(
    name="graph_gen_gym",
    version="0.1",
    packages=["graph_gen_gym"],
    cmdclass={"build_ext": CMakeBuild},
    install_requires=[
        "numpy",
        "torch",
        "torch_geometric",
        "scipy",
        "pydantic",
        "networkx",
        "joblib",
        "appdirs",
    ],  # Add your Python dependencies here
)
