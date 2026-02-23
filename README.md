# futuregreen_DS
Repo for data science team

To run this project, we will use virtual environments to ensure that library versions are consistent across the entire team
- before doing so, make sure you have python 3.11 downloaded ([link](https://www.python.org/downloads/release/python-3119/))
  - this seems to be the best version of python you can use

How to activate your virtual environment:
1) From the project root, run `python3.11 -m venv .venv`
2) Press Ctrl+Shift+P (Windows) or Cmd+Shift+P (Mac) and click on "Python: Select Interpreter"
   - choose the one with `.venv/bin/python` (Windows) or `.venv/bin/python` (Mac) in its name
3) Activate the virtual environment with `.venv\Scripts\activate` (Windows) or `source .venv/bin/activate` (Mac)
   - you should see `(.venv)` in the terminal after activating
4) Run `pip install --upgrade pip && pip install -r requirements.txt` to install the libraries into the virtual environment

You are now in the virutal environment! This is where you run the code with the correct libraries
- to get out of the virtual environment, run `deactivate`