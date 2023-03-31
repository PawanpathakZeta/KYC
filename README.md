# Streamlit Application - Probabilistic Record Linkage

If you want to skip the 'Instructions for usage' section to know how to set up the virtual environment necessary to run the application,
please jump directly to the 'How to run the application' one.


## Instructions for usage

### Clone the repository
Using a terminal of choice, navigate to a folder you wish to clone the repository using 

`cd the_folder_of_your_choice`

then clone the repository by using (in case of several keys, please use the one appropriate)

 `git clone git@gitlabmaster.bo3.e-dialog.com:GDSA_P/probabilistic-matching-streamlit-app.git`

### Useful git commands

* switch to an ***existing*** branch: `git checkout the_name_of_the_branch`
* switch to a ***new*** branch: `git checkout -b the_name_of_the_branch`
* update your local copy of the repository with remote changes: 
```
git pull --all
git fetch --all
```
* push your local changes to the remote branch:

Always check your current status with : `git status`

This will give information about the current branch you are working on as well as information about which files are changed

```
# Add the files you have changed to the commit
git add file_name
OR add all the changed files
git add .

# Commit your changes
git commit -m "add_a_description_here"

# Push the changes
git push origin the_name_of_your_branch
```


## How to run the application 

### Version of python
The application is built with **`python 3.7.6`**, this is the one that should be used for further developments.

In case your system version of python is not the recommended one, we encourage you the use of [`pyenv`](https://github.com/pyenv/pyenv) to handle several versions of python on the same machine. The first sub-section of the following section - about the set up of virtual environment - is dedicated to `pyenv`

### How to set up a virtual environment

#### Use of `pyenv-virtualenv`

Here, we assume that `pyenv` and `pyenv-virtualenv` are installed on your machine. In case not, and you are a Mac user, we recommend the use of `brew` for the installation. See the code snippet below, and the [github repository](https://github.com/pyenv) for more information.

```
brew install pyenv
brew install pyenv-virtualenv
```

Regarding the set up of the environment itself, please follow these steps in the shown order:
1. In case `python 3.7.6` is not installed, run `pyenv install 3.7.6` to install it
2. Create an empty virtual environment this way : `pyenv virtualenv 3.7.6 <environment_name>`
3. Activate the environment you just created : `pyenv activate <environment_name>`. After that, you should notice in the terminal that the line is prefixed with `(<environment_name>)`, like in the following example : `(prob_match_streamlit_app_env) M110587:probabilistic-matching-streamlit-app msaidi$ `
4. Install the necessary libraries thanks to `requirements.txt` : run `pip install -r requirements.txt`
5. Basic checks, if you run the command `pip show <library_name>`, you should be able to see the version (and more) of `<library_name>`:
      * Version of `streamlit` should be `1.11.1`
      * Version of `pandas` should be `1.3.2`

You can find more detailed information regarding the set up of virtual environment with pyenv in:
* The official [Github repository](https://github.com/pyenv)
* One of several web pages about this topic like [this one](https://akrabat.com/creating-virtual-environments-with-pyenv/)


#### Use of `venv`

Here, the assumption is - in addition to the fact that you do not use `pyenv` - `python 3.7.6` is your system python.

From there, the steps to follow are very similar to the ones described on the `pyenv` subsection. You have the details in the [official python documentation](https://docs.python.org/3/tutorial/venv.html) and a quick usable summary below:
1. Create an empty virtual environment this way : `python3 -m venv <environment_name>`. **Careful** : unlike `pyenv` the environment is physically created where you run the command, the choose the location mindfully
2. Activate the environment you just created : `source <environment_name>/bin/activate`. After that, you should notice in the terminal that the line is prefixed with `(<environment_name>)`, like in the following example : `(prob_match_streamlit_app_env) M110587:probabilistic-matching-streamlit-app msaidi$ `
3. Install the necessary libraries thanks to `requirements.txt` : run `pip install -r requirements.txt`
4. Basic checks, if you run the command `pip show <library_name>`, you should be able to see the version (and more) of `<library_name>`:
      * Version of `streamlit` should be `0.89.0`
      * Version of `pandas` should be `1.3.2`


#### Use of `conda`

### Run the application

Once the environment is set up, you can run the application this way:
* In case you are not in the project, get there with `cd <path_to_the_project>`
* Run `streamlit run probabilistic_matching_app.py`

The application should appears in a new tab of your default web browser.






