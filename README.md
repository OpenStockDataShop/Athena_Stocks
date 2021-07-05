# **Athena_Stocks**

by Aditya Sharoff, Felina Kang, Haeyoon Chang and [Vinodh Kotipalli](vkotipa2@pdx.edu) 2021


## **Project Description**
---
One stop shop for collection of Data Analytics and Data Visualization tool(s) to help with stock market trading decision making.We came across several open source AI/ML models and they were reported to have varying degrees of success on the historical data. However the real test is to see how well these models function current real world data and test the performance against real and/or simulated investments. For this end, we have two pronged approach.
  * Use custom web-scraper to collect the relevant data about selected stock(s) needed AI/ML models  and store then in database.
  * Run the stored time series datasets in the database through collection of AI/ML models to provide predictions on future stock trends which can be useful buy/sell/hold kind of decision making. 

In order to make process more user friendly and get quick enough feedback to help make real-world decisions we plan to build a web application with a simple UI containing follow minimum features
* Users can create personal user accounts to enter list of stock(s) and/or there detailed stock portfolio.
* Provide data visualization tools like time series graphs, pie charts etc that give visual feedback on individual stock and/or complete portfolio.
* Provide the best possible predictions on future stock performance and accordingly recommend buy/sell/hold decisions. 

**Note**: This application doesn't support features to make the actual financial transactions. 
## **Roadmap**
---
Due to limited timeline for the course project, we want focus on UI and data pipeline for collecting using web-scrapper first. We hope to have infrastructure and support for one prediction model. We plan to build the application modular enough to swap/add more models in future.

Here are the high level summary of tasks required achieve the said goal(s) for the project.  

- [ ] **Web-Scraper** for Data Collection
  - [ ] Gather historic/current prices of selected stock(s) and store to DB
  - [ ] Gather historic/current news relevant to selected stock(s) `+` URL(s) and store to DB
- [ ] **AI/ML models** to process collected data and provide predictions
  - [ ] Generate Abstracted-Text summary of new articles collected by Web-Scraper
  - [ ] LSTM model for buy/sell/hold predictions for a given stock
- [ ] **FrontEnd UI**
  - [ ] Ability for user to enter the stock(s) they are interested in for day-to-day tracking 
  - [ ] Ability for user to enter there personal portfolio across multiple platforms through CSV file
  - [ ] Data Visualization tool(s) for statistics on User portfolio
  - [ ] Data Visualization tool(s) for performance statistics on selected stock(s)
- [ ] **Database**
- [ ] **Backend**
  
## **Web Development Stack**
---
* Frontend
  * HTML
  * CSS
  * ReactJS
* Backend
  * Python Flask
  * Python AI/ML, Web-Scrapper support libraries
* Database
  * PostgreSQL
## **Acknowledgements**
---
* [Open Source Collection Machine Learning models of Stock Prediction](https://awesomeopensource.com/project/huseinzol05/Stock-Prediction-Models)

## **License**

This work is made available under the "MIT License". Please
see the file `LICENSE` in this distribution for license
terms.

---
## Build and Run (WIP)
---
### **Poetry commands**
* **Install poetry** if you havn't already with below command
    - `curl -SSL https://raw.githubusercontent.com/sdispater/poetry/master/get-poetry.py | python`

* **configure poetry** to create virtual environment in project path
    - `poetry config virtualenvs.in-project true`

* **Install dependencies** and create python virtual environment 
    - `poetry install`

* **Add new python library** to virtual environment 
    - `poetry add <library>`

- **Removed python library** from the dependency list
    - `poetry remove <library>`
  
- To **use** the already created **python virtual environment** 
    - `poetry env use python`

- **Run** python code  using **python virtual environment** 
    - `poetry run python <python file>`

