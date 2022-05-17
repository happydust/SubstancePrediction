import text2emotion as te # for emotion
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer # for sentiment
from tqdm import tqdm
analyser = SentimentIntensityAnalyzer()

def sentiment_analysis(message,type="post"):
    #this function takes care of both sentiment and emotion analysis
    #takes a list of dictionaries
    #no need to do these two in different founctions
    #return message length, as well as sentiment and emtions
    score = analyser.polarity_scores(message)
    emotions = te.get_emotion(message)
    message_length = len(message)
    if message_length == 0:
        positive, negative, neutral,sentiment = -1,-1,-1,-1
        happy, angry, surprise, sad, fear = -1, -1, -1, -1, -1
    else:
        positive = score['pos']
        negative = score['neg']
        neutral = score['neu']
        sentiment = score['compound']
        happy = emotions['Happy']
        angry = emotions['Angry']
        surprise = emotions['Surprise']
        sad = emotions['Sad']
        fear = emotions['Fear']
    message_sentiment = {f"{type}message_len":message_length,
                         f"{type}_positive":positive, f"{type}_negative":negative,f"{type}_neautral":neutral,
                         f"{type}_sentiment":sentiment, f"{type}_happy":happy,f"{type}_angry":angry,
                         f"{type}_surprise":surprise,f"{type}_sad":sad,f"{type}_fear":fear}

    return message_sentiment
