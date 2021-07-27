import newspaper
from GoogleNews import GoogleNews
import pandas as pd
import re

import dateparser
import datetime

# inspired from ref: https://stackoverflow.com/questions/12566152/python-x-days-ago-to-datetime
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

def getArticlesDataFrameFromGoogleNews(symbol,period=None):
    endDate = datetime.datetime.now()
    if period is None:
        startDate = getDateFromPeriod('yesterday')
    else:
        startDate = getDateFromPeriod(period)

    googlenews = GoogleNews()
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
    df = df[df['FullDate'] >= startDate]
    df['FullText'] = df['link'].apply(getFullArticleTextFromURL)
    df = df[df['FullText'] != "ERROR"]
    df['Summary'] = df['link'].apply(getArticleSummaryFromURL)
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







if __name__ == '__main__':
    #getFullArticleTextFromURL('https://www.businessinsider.com/apple-iphone-12-pro-review')
    #getFullArticleTextFromURL('https://www.businessinsider.com/kamala-harris-staffers-toxic-office-culture-dysfunction-2021-7')
    df = getArticlesDataFrameFromGoogleNews('AAPL')
    print(df.head())



