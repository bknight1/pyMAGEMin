from setuptools import setup, find_packages
from setuptools.command.install import install
import subprocess
import sys

class CustomInstall(install):
    def run(self):
        # Check for Julia installation and install if necessary
        try:
            subprocess.check_call(['julia', '--version'])
            print("Julia is already installed.")
            self.announce("Julia is already installed.", level=2)
        except FileNotFoundError:
            self.announce("Julia not found. Please install Julia before proceeding.", level=4)
            sys.exit(1)
        
        # Install MAGEMin Julia package
        try:
            subprocess.check_call(['julia', '-e', 'using Pkg; Pkg.add("MAGEMin_C")'])
            self.announce("MAGEMin Julia package installed successfully.", level=2)
        except Exception as e:
            self.announce(f"Failed to install Julia package: {e}", level=4)
            sys.exit(1)

        # Now call the standard install behavior
        install.run(self)

setup(
    name='PyMAGEMin',
    version='0.0.1',
    author='Ben Knight',
    author_email='ben.knight@curtin.edu.au',
    description='A python package to perform MAGEMin calculations in python',
    # long_description=open('README.md').read(),
    # long_description_content_type='text/markdown',
    # url='https://github.com/bknight1/GarnetDiffusion',
    packages=find_packages(where='src'),  # Automatically find and include all packages
    package_dir={'': 'src'},  # Tell distutils packages are under src
    cmdclass={
        'install': CustomInstall,
    },
    install_requires=[
        'pandas>=1.0.0',
        'numpy>=1.18.0',
        'scipy>=1.4.0',
        'juliacall>=0.9',
    ],
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: Academic Free License (AFL)',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: Scientific/Engineering',
        'Natural Language :: English',
    ],
    python_requires='>=3.10',
)
