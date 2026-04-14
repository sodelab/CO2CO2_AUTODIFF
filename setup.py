import os
import sys
import subprocess
import shutil
from setuptools import setup, find_packages
from setuptools.command.build_ext import build_ext  # Import the real build_ext
from setuptools.dist import Distribution

# --- Platform-aware library name ---
if sys.platform == "darwin":
    LIB_NAME = "libCO2CO2.dylib"
else:
    LIB_NAME = "libCO2CO2.so"

# Inherit from the real build_ext command
class MakeBuild(build_ext):
    def run(self):
        # This command is now primarily for local development builds.
        # The CI/CD pipeline uses cibuildwheel's `before-build` hook instead.
        print("--- Building C++ shared library ---")
        src_dir = os.path.join(os.path.dirname(__file__), "src")
        
        # Run make
        subprocess.check_call(["make", "clean"], cwd=src_dir)
        subprocess.check_call(["make"], cwd=src_dir)

        # Copy the built library into the package folder so it's included in the wheel
        pkg_dir = os.path.join(os.path.dirname(__file__), "co2_potential")
        src_lib_path = os.path.join(src_dir, LIB_NAME)
        dest_lib_path = os.path.join(pkg_dir, LIB_NAME)

        if os.path.exists(src_lib_path):
            print(f"--- Copying {src_lib_path} to {dest_lib_path} ---")
            shutil.copy(src_lib_path, dest_lib_path)
        else:
            # This check is important for CI/CD where the file might already be in place
            print(f"--- Library {src_lib_path} not found, assuming it's already in package dir ---")
        
        # It's important to call the superclass's run method
        super().run()

class BinaryDistribution(Distribution):
    """Force platform-specific distribution."""
    def has_ext_modules(self):
        return True

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="co2_potential",
    version="0.6.4",
    author="Olaseni Sode",
    license="MIT",
    author_email="osode@calstatela.edu",
    description="A Python package interfacing with the CO2CO2 shared library.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    # Use include_package_data and MANIFEST.in to grab the library
    include_package_data=True,
    # This tells setuptools that the wheel is not pure Python
    distclass=BinaryDistribution,
    # This custom command is mostly for `pip install -e .`
    cmdclass={
        "build_ext": MakeBuild,
    },
    install_requires=[
        "numpy",
    ],
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