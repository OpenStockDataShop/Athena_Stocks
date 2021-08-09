

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