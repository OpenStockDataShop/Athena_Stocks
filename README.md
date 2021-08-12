# **Athena_Stocks**

by [Aditya Sharoff](asharoff@pdx.edu), [Felina Kang](felina2@pdx.edu), [Haeyoon Chang](haeyoon@pdx.edu) and [Vinodh Kotipalli](vkotipa2@pdx.edu) 2021

``` html
<span style="font-size: 16px"><b style="color:red">Disclaimer:</b> Stocks investing in general are risky and can result in considerable loss. This is a academic project and we are not financial experts and are not responsible for any profit/loss from provided recommendations. Please don't use the information for trading.</span>
```
## **Project Description**
---
One stop shop for collection of Data Analytics and Data Visualization tool(s) to help with stock market trading decision making. We came across several open source AI/ML models and they were reported to have varying degrees of success on the historical data. However the real test is to see how well these models function current real world data and test the performance against real and/or simulated investments. For this end, we have two pronged approach.
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

- [x] **Web-Scraper** for Data Collection
  - [x] Gather historic/current prices of selected stock(s) and store to DB
  - [x] Gather historic/current news relevant to selected stock(s) `+` URL(s) and store to DB
- [x] **AI/ML models** to process collected data and provide predictions
  - [x] Generate Abstracted-Text summary of new articles collected by Web-Scraper
  - [x] Generate sentiment scores based on the generated summary
  - [x] LSTM model for buy/sell/hold predictions for a given stock
- [ ] **Frontend UI** on client side
  - [x] Ability for user to enter the stock(s) they are interested in for day-to-day tracking 
  - [ ] Ability for user to enter there personal portfolio across multiple platforms through CSV file
  - [ ] Data Visualization tool(s) for statistics on user portfolio
  - [ ] Data Visualization tool(s) for performance statistics on selected stock(s)
- [ ] **Backend** on server side
  - [ ] Create email for the organization and heroku account associated with it.
  - [x] Build backend server using Django library
  - [ ] Deploy the application using the organization's heroku account. 
- [x] **Database**
  - [x] Define Schema for storing User Accounts and time series dataset for the stock(s)
  - [x] Implement database interaction based on the Schema 

## **Web Development Stack**
---
* Frontend
  * HTML
  * CSS/Bootstrap
  * ReactJS
* Backend
  * Python Flask
  * Python AI/ML, Web-Scrapper support libraries
* Database
  * PostgreSQL

**Note**: We are exploring the possibility of building the entire stack in Django for 2-3 days, if successful it would allow us to implement entire project with python. 
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

