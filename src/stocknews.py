import os
import sys
import datetime
import pandas as pd
import numpy as np
from timerit import Timerit
import calendar
import multiprocessing
globalLock = multiprocessing.Lock()

pyenvdir = sys.prefix
download_dir= pyenvdir.replace('\\', '/') + '/lib/nltk_data'
if not os.path.isdir(download_dir):
    os.mkdir(download_dir)

import nltk
nltk.download('vader_lexicon',download_dir=download_dir)

projdir = os.path.dirname(os.path.abspath(__file__)).replace('\\', '/').replace('C:', '')

datadir = projdir.replace('\\', '/') + '/data'

if not os.path.isdir(datadir):
    os.mkdir(datadir)

# inspired from ref: https://stackoverflow.com/questions/12566152/python-x-days-ago-to-datetime
import re
import dateparser
def getDateFromString(dateString):
    try:
        date = dateparser.parse(dateString)
    except:
        date = getDateFromPeriod(dateString)
    return date

from datetime import date
def getDayInWeekFromString(dateString,day=None):
    d = getDateFromString(dateString)
    d = datetime.date.toordinal(d)
    if day is None:
        day = 0
    if day < 0:
        day = 0
    if day > 6:
        day = 6

    if (d % 7) == 0:
        d = d - 7
    d = d - (d % 7) + day

    return extractDateFromDatetime(datetime.date.fromordinal(d))

def getDateFromPeriod(period):
    periodSplit = re.findall('(\d+|[A-Za-z]+)',period)
    for i in range(len(periodSplit)):
        text = periodSplit[i]
        if text in ['h','hr','hours', 'hour']:
            periodSplit[i] = 'hours'
        if text in ['d','days','day']:
            periodSplit[i] = 'days'
        if text in ['w','wk','wks','weeks','week']:
            periodSplit[i] = 'weeks'
        if text in ['m','mon','mons','months','month']:
            periodSplit[i] = 'months'
        if text in ['y','yr','yrs','years','year']:
            periodSplit[i] = 'years'
    dateString = ' '.join(periodSplit) + ' ago'

    date = dateparser.parse(dateString)
    return date

def extractDateFromDatetime(date):
    return date.strftime('%Y-%m-%d')

# inspired from example code from ref: https://pypi.org/project/newspaper3k/
from newspaper import Article
def getFullArticleTextFromURL(url):
    article = Article(url)
    article.download()
    try:
        article.parse()
    except:
        return "ERROR"
    return article.text

# inspired from example code from ref: https://pypi.org/project/newspaper3k/
def getArticleSummaryFromURL(url):
    if url:
        article = Article(url)
        article.download()
        try:
            article.parse()
        except:
            return "ERROR"
    else:
        return "ERROR"
    article.nlp()
    return article.summary


# inspired from ref: https://realpython.com/python-nltk-sentiment-analysis/
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import sent_tokenize
def getSentimentFromText(text,scoreType=None):
    sia = SentimentIntensityAnalyzer()
    sentimentScores = dict()
    keys = ['neg','neu','pos','compound']
    for key in keys:
        sentimentScores[key] = 0

    count = 0
    for sentance in sent_tokenize(text):
        scores = sia.polarity_scores(sentance)
        for key in keys:
            sentimentScores[key] = ((count*sentimentScores[key]) + scores[key]) / (count + 1)
        count += 1

    return [sentimentScores[key] for key in keys]

def writeStockHistoryToCSV(symbol,period=None,getHistory=None):
    csvpath = datadir + '/StocknewsDataset' + symbol + '.csv'
    if getHistory is None:
        stockHistory = getHistoricalDataSetWithPyGoogleNews(symbol,period)
    else:
        stockHistory = getHistory(symbol,period)
    if stockHistory.shape[0] > 0:
        if os.path.exists(csvpath):
            stockHistory.to_csv(csvpath,index=False,mode='a',header=False)
        else:
            stockHistory.to_csv(csvpath,index=False)
    return stockHistory

#inspired from example code from ref: https://pypi.org/project/yfinance/
#inspired from ref: https://stackoverflow.com/questions/45545110/make-pandas-dataframe-apply-use-all-cores
import yfinance as yf
from multiprocessing.dummy import Pool as ThreadPool

def getHistoricalDataSetWithPyGoogleNews(symbol,period=None):
    #print('symbol = %s period = %s\n' %(symbol,period))
    endTime = datetime.datetime.now()
    if period is None or (type(period) == list and len(period) == 0):
        startTime = getDateFromString('yesterday')
        endTime = datetime.datetime.now()
    elif type(period) != list:
        startTime = getDateFromString(period)
        endTime = datetime.datetime.now()
    else:
        startTime = getDateFromString(period[0])
        if len(period) > 1:
            endTime = getDateFromString(period[1])

    startDate = extractDateFromDatetime(startTime)
    endDate = extractDateFromDatetime(endTime)

    start = datetime.datetime.strptime(startDate,"%Y-%m-%d")
    end = datetime.datetime.strptime(endDate,"%Y-%m-%d")

    globalLock.acquire()
    stockPrices = yf.download(symbol,start=start,end=end,progress=False)
    globalLock.release()

    stockPrices = stockPrices.reset_index(level=0)
    stockPrices['ActualDate'] = stockPrices['Date'].apply(extractDateFromDatetime)


    stockNewsArgs = pd.DataFrame()
    stockNewsArgs['ActualDate'] = stockPrices['ActualDate']
    stockNewsArgs['symbol'] = symbol

    pool = ThreadPool(stockNewsArgs.shape[0])
    stockNews = pool.map(getArticlesOnDateParallel, stockNewsArgs.to_numpy().tolist())
    pool.close()
    pool.join()
    stockNews = np.array(stockNews)
    stockNews = np.stack(stockNews) 

    if len(stockNews.shape) == 3:
        stockNews = stockNews.reshape((stockNews.shape[0]*stockNews.shape[1],stockNews.shape[2]))
    stockNews = pd.DataFrame(stockNews,columns=['ActualDate','link'])
    stockNews = stockNews.reset_index()

    pool = ThreadPool(stockNews.shape[0])
    stockNews['Summary'] = pool.map(getArticleSummaryFromURL,stockNews['link'].tolist())
    pool.close()
    pool.join()

    stockNews = stockNews[stockNews['Summary'] != "ERROR"]

    columns = ['neg','neu','pos','compound']
    scores = pd.DataFrame(np.stack(np.array(stockNews['Summary'].apply(getSentimentFromText))),columns=columns)

    for key in columns:
        stockNews[key] = scores[key]
    stockNews = stockNews.dropna()

    sentimentScores = getPerDaySentimentScoresFromDataSet(stockNews)
    try:
        stockHistory = sentimentScores.merge(stockPrices,how='outer',on='ActualDate')
    except:
        stockHistory = stockPrices
        for key in columns:
            stockHistory[key] = 0.0

    stockHistory = stockHistory.drop(['Date'],axis=1)
    stockHistory = stockHistory.rename(columns={'ActualDate':'Date'})

    stockHistory['Symbol'] = symbol

    stockHistory = stockHistory[['Symbol','Date','Open','High','Low','Close','Adj Close','Volume','neg','neu','pos','compound']]
    stockHistory = stockHistory.dropna(subset=['Symbol','Date','Open','High','Low','Close','Adj Close','Volume'])
    stockHistory = stockHistory.fillna(0.0)


    return stockHistory


def getPerDaySentimentScoresFromDataSet(newsArticles):
    keys = ['ActualDate','neg','neu','pos','compound']
    sentimentScores = pd.DataFrame()
    if newsArticles.shape[0] > 0:
        for key in keys:
            sentimentScores[key] = newsArticles[key]

        sentimentScores = sentimentScores.groupby(['ActualDate']).mean()
        sentimentScores = sentimentScores.reset_index(level=0)
    
    return sentimentScores

# inspired from example code from ref: https://pypi.org/project/pygooglenews/
import pygooglenews 
def getArticlesOnDate(dateString,symbol,maxArticles=10):
    googlenews = pygooglenews.GoogleNews()
    try:
        pageResults = googlenews.search(symbol,from_=getStartEndTimeOfDate(dateString)['start'],to_=getStartEndTimeOfDate(dateString)['end'])
        df = pd.DataFrame(pageResults['entries'])
        if df.shape[0] > 0:
            df['FullDate'] = df['published'].apply(dateparser.parse)
            df['ActualDate'] = df['FullDate'].apply(extractDateFromDatetime)
            df = df[df['ActualDate'] == dateString]
            df = df[['ActualDate','link']]
        else:
            df = pd.DataFrame([[dateString,None]], columns = ['ActualDate','link'])
    except:
        df = pd.DataFrame([[dateString,None]], columns = ['ActualDate','link'])

    if df.shape[0] > maxArticles:
        df = df.sample(n = maxArticles,replace=False)
    elif df.shape[0] < maxArticles:
        while (df.shape[0]< maxArticles):
            df = df.append(pd.DataFrame([[dateString,None]], columns = ['ActualDate','link']))
    return df.to_numpy()

def getArticlesOnDateParallel(args):
    dateString = args[0]
    symbol = args[1]
    if len(args) > 2:
        maxArticles = args[3]
        return getArticlesOnDate(dateString,symbol,maxArticles)
    else:
        return getArticlesOnDate(dateString,symbol)

def getSentimentScoresOnDate(dateString,symbol,maxArticles=10):
    stockNews = getArticlesOnDate(symbol,dateString,maxArticles)
    stockNews['Summary'] = stockNews['link'].apply(getArticleSummaryFromURL)
    stockNews = stockNews[stockNews['Summary'] != "ERROR"]
    columns = ['neg','neu','pos','compound']
    if stockNews.shape[0] > 0:
        scores = pd.DataFrame(np.stack(np.array(stockNews['Summary'].apply(getSentimentFromText))),columns=columns)
        for key in columns:
            stockNews[key] = scores[key]

        sentimentScores = stockNews[columns]
        sentimentScores = sentimentScores.dropna()

        if sentimentScores.shape[0] == 0:
            meanSentimentScores = [0.0,0.0,0.0,0.0]
        else:
            meanSentimentScores = sentimentScores[columns].to_numpy()
            meanSentimentScores = np.mean(meanSentimentScores,axis=0).tolist()
    else:
        meanSentimentScores = [0.0,0.0,0.0,0.0]
    meanSentimentScores.append(dateString)
    return meanSentimentScores


def getStartEndTimeOfDate(dateString):
    date = dateparser.parse(dateString)
    extractedDate = extractDateFromDatetime(date)
    result = dict()
    result['start'] = extractedDate
    result['end'] = extractDateFromDatetime(date + datetime.timedelta(days=1))
    return result

def writeStockHistoryToCSVparallel(args):
    symbol = args[0]
    period = args[1]
    getHistory = args[2]
    #print("\t==========================Start(%s)================================\n" % symbol)
    #for _ in Timerit(num=1,verbose=2):
    #    df = writeStockHistoryToCSV(symbol,period,getHistory)
    #print("\t==========================Finish(%s)================================\n" % symbol)

    df = writeStockHistoryToCSV(symbol,period,getHistory)
    return df

if __name__ == '__main__':
    #getFullArticleTextFromURL('https://www.businessinsider.com/apple-iphone-12-pro-review')
    #getFullArticleTextFromURL('https://www.businessinsider.com/kamala-harris-staffers-toxic-office-culture-dysfunction-2021-7')
    #scores = getPerDaySentimentScoresFromGoogleNews('TSLA','1d')
    #print(scores.head())

    weeks = 180
    dateRanges = []
    for i in range(weeks,0,-1):
        dateRange = [getDayInWeekFromString(str(i)+'w',0),getDayInWeekFromString(str(i)+'w',6)]
        dateRanges.append(dateRange)
    dateRange = [getDayInWeekFromString('today',0),getDayInWeekFromString('today',6)]
    if dateRange[0] != dateRanges[-1][0] and dateRange[1] != dateRanges[-1][1]:
        dateRanges.append(dateRange)
    
    print(getDayInWeekFromString('2021-08-08',0))

    stocks = ['AAPL','TSLA','MSFT','INTC','AMZN','WMT']
    #stocks = ['AAPL']
    #for dateRange in dateRanges:
    #    print("\n==========================DateRangeStart(%s)================================" % dateRange[0])
    #    stockArgs = pd.DataFrame()
    #    stockArgs['symbol'] = stocks
    #    stockArgs['period'] = [dateRange] * len(stocks)
    #    stockArgs['getHistory'] = getHistoricalDataSetWithPyGoogleNews
        
    #    #for _ in Timerit(num=1,verbose=2):
    #    #    for stock in stocks:
    #    #        writeStockHistoryToCSV(stock,dateRange,getHistoricalDataSetWithPyGoogleNews)

    #    for _ in Timerit(num=1,verbose=2):
    #        pool = ThreadPool(len(stocks))
    #        results = pool.map(writeStockHistoryToCSVparallel,stockArgs.to_numpy().tolist())
    #        pool.close()
    #        pool.join()

    #    print("==========================DateRangeEnd(%s)================================" % dateRange[1])

