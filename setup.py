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
                    "graph_gen_gym/orca_src/orca.cpp",
                ]
            )
        build_ext.run(self)


setup(
    cmdclass={"build_ext": CMakeBuild},
)
