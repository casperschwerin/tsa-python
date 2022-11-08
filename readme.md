# Embryo for python package for time series analysis course

# Intro
So far, all there is in this repo is an implementation of PEM as well as a short example on how to work with the scipy signal processing toolbox and make some plots.

## Installation
I assume you have python 3.8 or later installed on your computer.
### 1. Create virtual environment
Install virtualenv if not already installed. This is done by typing `pip install virtualenv` in your terminal.

Open terminal and navigate to your time-series directory. Type `python -m venv tsa` to create the virtual environment
### 2. Activate the virtual environment
In the newly created directory, there is a bin-folder. Among other things, this contains an activation script: `<tsa-folder>/bin/activate`. Type `source <path-to-activate-file>` to activate your virtual environment.

*Tip! You can create an alias for this step. If you're on mac, you can open the file `~/.bash_profile` and add a line: `alias tsa="source <path-to-activate-file>"`. Next time you open terminal, you only need to write "tsa" to activate the virtual environment.*
### 3. Install this package
Navigate to the root of this project and type `pip install -r requirements.txt` followed by `pip install --editable .`.

Now, when you're in this virtual environment you can always import files from this repo, where ever you are on the computer.

## Contributing
Read the `contributing.md` file before pushing code.