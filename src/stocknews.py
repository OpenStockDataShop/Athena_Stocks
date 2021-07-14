
import nltk
nltk.download('punkt')

from newspaper import Article

def getArticleSummary(urllink):
    article = Article(urllink)
    article.download()
    article.parse()
    article.nlp()

    print(article.keywords)
    print(article.publish_date)
    print(article.summary)




if __name__ == '__main__':
    getArticleSummary('https://www.businessinsider.com/apple-iphone-12-pro-review')