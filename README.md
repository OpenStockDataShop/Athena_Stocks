# **Athena_Stocks**
Aditya Sharoff, Felina Kang, Haeyoon Chang and Vinodh Kotipalli 2021

## **High Level Description**
One stop shop for collection of Data Analytics and Data Visualization tool help with stock market trading decision making. 
## **Roadmap**
- [ ] **Web-Scraper** for Data Collection
  - [ ] Gather historic/current prices of selected stock(s) and store to DB
  - [ ] Gather historic/current news relevant to selected stock(s) `+` URL(s) and store to DB
- [ ] **AI/ML models** to process collected data and provide predictions
  - [ ] Generate Abstracted-Text summary of new articles collected by Web-Scraper
  - [ ] LSTM model for buy/sell/hold predictions for a given stock
- [ ] FrontEnd UI
  - [ ] Ability for user to enter the stock(s) they are interested in for day-to-day tracking 
  - [ ] Ability for user to enter there personal portfolio across multiple platforms through CSV file
  - [ ] Data Visualization tool(s) for statistics on User portfolio
  - [ ] Data Visualization tool(s) for performance statistics on selected stock(s)
  
## **Web Development Stack**
* Frontend
  * HTML
  * CSS
  * ReactJS
* Backend
  * Python Flask
  * Python AI/ML, Web-Scrapper support libraries
* Database
  * ???
## **Acknowledgements**

## **License**

This work is made available under the "MIT License". Please
see the file `LICENSE` in this distribution for license
terms.
## **Poetry commands**
* Install poetry if you havn't already with below command
    - `curl -SSL https://raw.githubusercontent.com/sdispater/poetry/master/get-poetry.py | python`

* configure poetry to create virtual environment in project path
    - `poetry config virtualenvs.in-project true`

* Install dependencies and create python virtual environment 
    - `poetry install`

* Add new python library to virtual environment 
    - `poetry add <library>`

- Removed  python library from the dependency list
    - `poetry remove <library>`
  
- To use the already created python virtual environment 
    - `poetry env use python`

- Start server with command(with poetry)
    - `poetry run python <python file>`

