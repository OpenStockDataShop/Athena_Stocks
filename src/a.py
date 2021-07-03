from numpy import recfromtxt
from pandas.core.frame import DataFrame
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from bs4 import BeautifulSoup
import time
import pandas as pd
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
import re


def scrape_for_google_finance(the_stock_name):
    print("stock name = ", the_stock_name)
    driver = webdriver.Chrome('./chromedriver')
    driver.get("https://www.google.com/finance/")
    driver.implicitly_wait(20)
    search = driver.find_elements_by_xpath("//*[@id='yDmH0d']/c-wiz/div/div[3]/div[3]/div/div/div/div[1]/input[2]")
    the_search = search[0]
    the_search.send_keys(the_stock_name)
    #time.sleep(5)
    the_search.send_keys(Keys.ENTER)
    time.sleep(5)
    the_html = driver.page_source
    soup = BeautifulSoup(the_html, 'lxml')
    the_soup = soup.prettify()
    myTable = soup.find_all('div', {'class': 'P6K39c'})
    recTable = soup.find_all('a', {'class': 'iFOBwb'})
    temp_rec_string = ""
    driver.close()
    if(len(myTable) == 0):
        print("noo")
        bad_list = []
        bad_list.append(-1)
        return bad_list, bad_list
    rec_list_of_companies= []
    for i in range(len(recTable)):
        temp_rec_string = str(recTable[i])
        #print(recTable[i])
        #print()
        if(temp_rec_string.find('href') != -1):
            #print(temp_rec_string)
            #print(temp_rec_string)
            r1 = re.findall(r"(href=(.)+jslog)", temp_rec_string)
            rec_list_of_companies.append(r1[0])
            #print(temp_rec_string)
    rec_list_parser = ""
    find_quote_slash = ""
    temp_num = 0
    for i in range(len(rec_list_of_companies)):
        rec_list_of_companies[i] = str(rec_list_of_companies[i])
        rec_list_parser = rec_list_of_companies[i]
        temp_num = rec_list_parser.rfind("/quote/")
        temp_num = temp_num + 7
        end_num = rec_list_parser.rfind("jslog")
        rec_list_of_companies[i] = rec_list_parser[(temp_num):(end_num-2)]
        #while(rec_list_parser[temp_num] != "\""):
        #    rec_list_of_companies[i] = rec_list_of_companies[i] + rec_list_parser[temp_num]

        
    #print(rec_list_of_companies)
    the_string = ""
    temp_string = ""
    sub_string = " a class"
    counter = 0
    myList = []
    for i in range(len(myTable)):
        temp_string = str(myTable[i])
        if(temp_string.find("a class=") != -1):
            continue
        myList.append(str(myTable[i].get_text()))
        counter = counter + 1
    #driver.close()
    return myList, rec_list_of_companies

def askUser(take_recs):
    finished = 0
    theList = []
    list_of_lists = []
    rec_list = []
    while(finished == 0 and len(take_recs) == 0):
        theList.append(input("enter name of stock: "))
        finished = int(input("enter 1 if you are done and 0 if not: "))
    if(len(take_recs) > 0):
        theList = take_recs
    for i in range(len(theList)):
        returnList = []
        list_of_lists.append(theList[i])
        returnList, rec_list = scrape_for_google_finance(theList[i])
        #print("pay attention: ", returnList)
        failSafe = 0
        while(returnList[0] == -1 or failSafe == 5):
            returnList, rec_list=scrape_for_google_finance(theList[i])
            failSafe = failSafe + 1
            #print(returnList)
        #print(returnList[0])
        list_for_the_company = []
        for q in range(len(returnList)):
            print(returnList[q])
            if(returnList[q].find(',') != -1):
                continue
            #print(returnList[i])
            list_for_the_company.append(returnList[q])
            #print("pay attention here", list_for_the_company)
        list_of_lists.append(list_for_the_company)
    #print("pay attention", list_of_lists)
    return list_of_lists, rec_list

def get_previous_close(previous_close):
    temp_string = ""
    number = float(0)
    for i in range(len(previous_close)):
        if(previous_close[i].isnumeric() == True or previous_close[i] == "."):
            temp_string = temp_string + previous_close[i]
    number = float(temp_string)
    return number
def get_day_ranges_or_year_range(my_range):
    the_list = []
    the_string = ""
    num1 = float(0)
    num2 = float(0)
    for i in range(len(my_range)):
        if(my_range[i] == "-"):
            num1 = float(the_string)
            the_list.append(num1)
            the_string = ""
        if(i == (len(my_range)-1)):
            num2 = float(the_string)
            if(my_range[i].isnumeric() == True):
                the_string = the_string + my_range[i]
            num2 = float(the_string)
            the_list.append(num2)
        if(my_range[i].isnumeric() == False and my_range[i] != "."):
            continue
        if(my_range[i].isnumeric() == True or my_range[i] == "."):
            the_string = the_string + my_range[i]
    if(len(the_list) < 1):
        the_list.append(False)
        the_list.append(False)
    if(len(the_list) < 2):
        the_list.append(False)
    return the_list

def main_test(rec_list):
    final_test_list = []
    final_rec_list = rec_list
    final_test_list, final_rec_list = askUser(final_rec_list)
    #print("final_rec_list =", final_rec_list)
    my_dict = {"Company_Name": [], "Previous_Close": [], "Day_Range_Low": [], "Day_Range_High": [],
    "Year_Range_Low": [], "Year_Range_High": [], "Market_Cap_USD": [], "Volume": [], "P/E_Ratio": [],
    "Dividend_Yield": [], "Primary_Exchange": []}
    
    the_temporary = []
    
    for i in range(len(final_test_list)):
        if(type(final_test_list[i]) == str):
            my_dict["Company_Name"].append(final_test_list[i])
            continue
        
        the_temporary = final_test_list[i]
        number = float(0)
        number_list = []
        other_number = float(0)
        counter = 0
        
        for i in range(len(the_temporary)):
            if(i == 0):
                number = get_previous_close(the_temporary[i])
                my_dict["Previous_Close"].append(number)
            if(i == 1):
                number_list = get_day_ranges_or_year_range(the_temporary[i])
                my_dict["Day_Range_Low"].append(number_list[0])
                my_dict["Day_Range_High"].append(number_list[1])
            if(i == 2):
                number_list = get_day_ranges_or_year_range(the_temporary[i])
                my_dict["Year_Range_Low"].append(number_list[0])
                my_dict["Year_Range_High"].append(number_list[1])
            if(i == 3):
                my_dict["Market_Cap_USD"].append(the_temporary[i])
            if(i == 4):
                my_dict["Volume"].append(the_temporary[i])
            if(i == 5):
                my_dict["P/E_Ratio"].append(float(the_temporary[i]))
            if(i == 6):
                if(the_temporary[i] == "-"):
                    my_dict["Dividend_Yield"].append(float(0))
                else:
                    y = the_temporary[i]
                    my_dict["Dividend_Yield"].append(float(y[0:(len(y)-1)]))
            if(i == 7):
                my_dict["Primary_Exchange"].append((the_temporary[i]))    
        
    repair_list = []
    counter = 0
    proper_length = 0
    temp = 0
    
    for key in my_dict:
        if(counter == 0):
            proper_length = len(my_dict["Company_Name"])
            print("proper_length = ", proper_length)
            counter = counter + 1
        if(len(my_dict[key]) < proper_length):
            while(len(my_dict[key]) < proper_length):
                my_dict[key].append(False)
            continue
        if(len(my_dict[key]) < proper_length):
            while(len(my_dict[key]) < proper_length):
                my_dict[key].append(False)
            my_dict[key].append(False)

    #print(my_dict)
    return my_dict, final_rec_list

def final_function_of_web():
    rec_of_recs= []
    the_final_dict, rec_of_recs = main_test(rec_of_recs)
    df = DataFrame(the_final_dict)
    print(df)
    print("rec_of_recs = ", rec_of_recs)
    print(rec_of_recs)
    the_final_dict, rec_of_recs = main_test(rec_of_recs[5:10])
    df1 = DataFrame(the_final_dict)
    #print(df1)
    all_frames = [df, df1]
    final_result = pd.concat(all_frames)
    print(final_result)

final_function_of_web()