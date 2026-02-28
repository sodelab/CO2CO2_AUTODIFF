import os
import sys
import subprocess
import shutil
from setuptools import setup, find_packages, Command, Distribution
from wheel.bdist_wheel import bdist_wheel

class MakeBuild(Command):
    description = "Build the C++ shared library using Makefile"
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        cwd = os.path.join(os.path.dirname(__file__), "src")
        subprocess.check_call(["make", "clean"], cwd=cwd)
        subprocess.check_call(["make"], cwd=cwd)
        # Copy the built shared library into the package folder
        pkg_dir = os.path.join(os.path.dirname(__file__), "co2_potential")
        src_lib = os.path.join(cwd, "libCO2CO2.so")
        if os.path.exists(src_lib):
            shutil.copy(src_lib, os.path.join(pkg_dir, "libCO2CO2.so"))

    def get_source_files(self):
        return []

    def get_outputs(self):
        return []

    def get_output_mapping(self):
        return {}

class BinaryDistribution(Distribution):
    """Force platform-specific distribution (platlib, not purelib)."""
    def has_ext_modules(self):
        return True

class BdistWheelPlatform(bdist_wheel):
    """Mark the wheel as platform-specific (not pure Python)."""
    def finalize_options(self):
        bdist_wheel.finalize_options(self)
        self.root_is_pure = False

    def get_tag(self):
        python, abi, plat = bdist_wheel.get_tag(self)
        # Use generic Python 3 tag since we use ctypes, not a C extension
        python, abi = "py3", "none"
        return python, abi, plat

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="co2_potential",
    version="0.4.4",
    author="Olaseni Sode",
    license="MIT",
    author_email="osode@calstatela.edu",
    description="A Python package interfacing with the CO2CO2 shared library.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    include_package_data=True,
    package_data={
        "co2_potential": ["libCO2CO2.so"],
    },
    install_requires=[
        "numpy",
    ],
    distclass=BinaryDistribution,
    cmdclass={
        "build_ext": MakeBuild,
        "bdist_wheel": BdistWheelPlatform,
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: C++",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: POSIX :: Linux",
    ],
    python_requires=">=3.6",
    entry_points={
        'console_scripts': [
            'co2-benchmark=co2_potential.benchmark:main',
        ],
    },
)