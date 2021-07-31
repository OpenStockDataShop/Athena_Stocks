# inspired from ref: https://stackoverflow.com/questions/12566152/python-x-days-ago-to-datetime
import re
import dateparser
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
    return date.strftime('%y-%m-%d')
# inspired from example code from ref: https://pypi.org/project/GoogleNews/
from GoogleNews import GoogleNews
import datetime
import pandas as pd
import numpy as np
import time
import http.cookiejar, urllib.request
def getArticlesDataFrameFromGoogleNews(symbol,period=None):
    endDate = datetime.datetime.now()
    if period is None:
        startDate = getDateFromPeriod('yesterday')
    else:
        startDate = getDateFromPeriod(period)

    googlenews = GoogleNews()
    googlenews.clear()
    googlenews.search(symbol)
    pageResults = googlenews.results()
    df = pd.DataFrame(pageResults)
    i = 2
    while True:
        googlenews.get_page(i)
        pageResults = googlenews.results()
        df = pd.DataFrame(pageResults)
        lastDate = getDateFromPeriod(df['date'].iloc[-1])
        i += 1
        if lastDate < startDate:
            break
    df['FullDate'] = df['date'].apply(getDateFromPeriod)
    df['ActualDate'] = df['FullDate'].apply(extractDateFromDatetime)
    df = df[df['FullDate'] >= startDate]
    df['FullText'] = df['link'].apply(getFullArticleTextFromURL)
    df = df[df['FullText'] != "ERROR"]
    df['Summary'] = df['link'].apply(getArticleSummaryFromURL)

    columns = ['SummaryScore-neg','SummaryScore-neu','SummaryScore-pos','SummaryScore-compound']
    scores = pd.DataFrame(np.stack(np.array(df['Summary'].apply(getSentimentFromText))),columns=columns)
    for key in columns:
        df[key] = scores[key]

    columns = ['FullTextScore-neg','FullTextScore-neu','FullTextScore-pos','FullTextScore-compound']
    scores = pd.DataFrame(np.stack(np.array(df['FullText'].apply(getSentimentFromText))),columns=columns)
    for key in columns:
        df[key] = scores[key]

    googlenews.clear()
    return df

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
    article = Article(url)
    article.download()
    try:
        article.parse()
    except:
        return "ERROR"
    article.nlp()
    return article.summary


# inspired from ref: https://realpython.com/python-nltk-sentiment-analysis/
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import sent_tokenize
import nltk
nltk.download('vader_lexicon')
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

def getPerDaySentimentScoresFromGoogleNews(symbol,period=None):
    keys = ['ActualDate','SummaryScore-neg','SummaryScore-neu','SummaryScore-pos','SummaryScore-compound','FullTextScore-neg','FullTextScore-neu','FullTextScore-pos','FullTextScore-compound']

    newsArticles = getArticlesDataFrameFromGoogleNews(symbol,period)
    sentimentScores = pd.DataFrame()
    for key in keys:
        sentimentScores[key] = newsArticles[key]
    
    sentimentScores = sentimentScores.groupby(['ActualDate']).mean()
    sentimentScores = sentimentScores.reset_index(level=0)
    return sentimentScores
if __name__ == '__main__':
    #getFullArticleTextFromURL('https://www.businessinsider.com/apple-iphone-12-pro-review')
    #getFullArticleTextFromURL('https://www.businessinsider.com/kamala-harris-staffers-toxic-office-culture-dysfunction-2021-7')
    scores = getPerDaySentimentScoresFromGoogleNews('TSLA','1d')
    print(scores.head())


