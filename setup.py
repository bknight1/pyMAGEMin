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
        
        # Install Julia packages
        # You should replace `PackageName` with the actual name of the Julia package
        try:
            subprocess.check_call(['julia', '-e', 'using Pkg; Pkg.add("MAGEMin_C")'])
        except Exception as e:
            print(f"Failed to install Julia package: {e}")

        try:
            subprocess.check_call([sys.executable, '-c', 'import julia; julia.install()'])
        except Exception as e:
            print(f"Failed to install pyJulia: {e}")


setup(
    name='UWGarnetDiffusion',
    version='0.1.0',  # Consider using semantic versioning
    author='Your Name',
    author_email='your.email@example.com',
    description='A brief description of your package',
    # long_description=open('README.md').read(),
    # long_description_content_type='text/markdown',  # If your README is Markdown
    # url='https://github.com/yourusername/UWGarnetDiffusion',  # Optional
    packages=find_packages(),  # Automatically find and include all packages
    install_requires=[
        'pandas>=1.0.0',  # Specify minimum versions
        'numpy>=1.18.0',
        'scipy>=1.4.0',
        'julia>=0.5.6',  # Ensure this matches the PyPI name for pyjulia
    ],
    cmdclass={
        'install': CustomInstall,
    },
    classifiers=[
        'Development Status :: 3 - Alpha',  # Change as appropriate
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',  # Change as appropriate
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
    python_requires='>=3.7',  # Specify compatible Python versions
)
