# Repository for UCU summer school

See instructions on environment setup below:

## Installing Anaconda

Anaconda is available for Windows, Mac OS X, and Linux. You can find the installers and installation instructions at [https://www.continuum.io/downloads](https://www.continuum.io/downloads).

If you already have Python installed on your computer, this won't break anything. Instead, the default Python used by your scripts and programs will be the one that comes with Anaconda.

Choose the Python 3.6 version, you can install Python 2 versions later. Also, choose the 64-bit installer if you have a 64-bit operating system, otherwise go with the 32-bit installer. Go ahead and choose the appropriate version, then install it. Continue on afterwards!

After installation, you’re automatically in the default conda environment with all packages installed which you can see below. You can check out your own install by entering into your terminal  `conda list`

### On Windows
A bunch of applications are installed along with Anaconda:

- **Anaconda Navigator**, a GUI for managing your environments and packages
- **Anaconda Prompt**, a terminal where you can use the command line interface to manage your environments and packages
- **Spyder**, an IDE geared toward scientific development

To avoid errors later, it's best to update all the packages in the default environment. Open the Anaconda Prompt application. In the prompt, run the following commands:
```
conda upgrade conda
conda upgrade --all
```

and answer yes when asked if you want to install the packages. The packages that come with the initial install tend to be out of date, so updating them now will prevent future errors from out of date software.

**Note:** In the previous step, running ```conda upgrade conda``` should not be necessary because ```--all``` includes the conda package itself, but some users have encountered errors without it.


## Install working environment

Follow these instructions to create and configure your environment. An environment file for supported OSes, which will install Python 3 and all the necessary packages used in this course.

**Linux**

Download the _ucu-environment-unix.yml_ file.

```conda env create -f ucu-environment-unix.yml``` to create the environment.

```source activate ucu-sentiment``` to enter the environment.

**Mac OS X**

Download the _ucu-environment-osx.yml_ file.

```conda env create -f ucu-environment-osx.yml``` to create the environment.

```source activate ucu-sentiment``` to enter the environment.

**Windows**

Download the _ucu-environment-windows.yml_ file.

```conda env create -f ucu-environment-windows.yml``` to create the environment.

```activate ucu-sentiment``` to enter the environment.
