from setuptools import setup, find_packages
from setuptools.command.install import install
import subprocess
import sys


class CustomInstall(install):
    def run(self):
        install.run(self)
        # Check for Julia installation and install if necessary
        try:
            subprocess.check_call(['julia', '--version'])
            print("Julia is already installed.")
        except FileNotFoundError:
            print("Julia not found. Please install Julia.")
        
        # Install MAGEMin Julia package
        try:
            subprocess.check_call(['julia', '-e', 'using Pkg; Pkg.add("MAGEMin_C")'])
        except Exception as e:
            print(f"Failed to install Julia package: {e}")

        try:
            subprocess.check_call([sys.executable, '-c', 'import julia; julia.install()'])
        except Exception as e:
            print(f"Failed to install pyJulia: {e}")


setup(
    name='GarnetDiffusion',
    version='0.0.1',  # Consider using semantic versioning
    author='Ben Knight',
    author_email='ben.knight@curtin.edu.au',
    description='A collection of functions to generate garnets using MAGEMin',
    # long_description=open('README.md').read(),
    # long_description_content_type='text/markdown',  # If your README is Markdown
    # url='https://github.com/bknight1/GarnetDiffusion',  # Optional
    packages=find_packages(where='src'),  # Automatically find and include all packages
    package_dir={'':'src'},  # Tell distutils packages are under src
    install_requires=[
        'pandas>=1.0.0',  
        'numpy>=1.18.0',
        'scipy>=1.4.0',
        'juliacall>=0.9',  
    ],
    cmdclass={
        'install': CustomInstall,
    },
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',  # Change as appropriate
        'Intended Audience :: Developers',
        'License :: OSI Approved :: Academic Free License (AFL)',  # Change as appropriate
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: Scientific/Engineering',
        'Natural Language :: English',

        
    ],
    python_requires='>=3.10',  # Specify compatible Python versions
)
