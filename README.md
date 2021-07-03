# Athena_Stocks

## Poetry commands
- Install poetry if you havn't already with below command
    - `curl -SSL https://raw.githubusercontent.com/sdispater/poetry/master/get-poetry.py | python`

- configure poetry to create virtual environment in project path
    - `poetry config virtualenvs.in-project true`

- Install dependencies and create python virtual environment 
    - `poetry install`

- Add new python library to virtual environment 
    - `poetry add <library>`

- Removed  python library from the dependency list
    - `poetry remove <library>`
  
- To use the already created python virtual environment 
    - `poetry env use python`

- Start server with command(with poetry)
    - `poetry run python <python file>`

