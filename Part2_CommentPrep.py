import pandas as pd
import os
import re
from tqdm import tqdm
import text2emotion as te # for emotion
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer # for sentiment
import emoji
import nltk

#read the data
path_comment = os.path.join(os.path.dirname(os.getcwd()), "Data\\GSSWData-Full\\Raw\\comments.csv")
comment_df = pd.read_csv(path_comment,encoding = "ISO-8859-1")
post_comments = []
dfgrp =comment_df.groupby('parent_id')


for grp in tqdm(dfgrp):
    grpdf = grp[1]
    comment_number = len(grp[1])
    post_id = grp[0]
    msgs = ""
    for m in grpdf['message']:
        try:
            m = re.sub("\s+", " ", m)

        except TypeError as err:
            print(m)
        else:
            msgs += " " + m
    #sentiment
    analyser = SentimentIntensityAnalyzer()
    score = analyser.polarity_scores(msgs)
    sentiment = score['compound']

    #emotions
    emotions = te.get_emotion(msgs)
    happy = emotions['Happy']
    angry = emotions['Angry']
    surprise = emotions['Surprise']
    sad = emotions['Sad']
    fear = emotions['Fear']

    post_comments.append([post_id,sentiment,happy,angry,surprise, sad, fear,comment_number])

post_comment_df =pd.DataFrame(data=post_comments,columns=['post_id','comment_sentiment','comment_happy',
                                                          'comment_angry','comment_surprise',"comment_sad",
                                                     'comment_fear','comment_number'])

post_comment_df.to_csv("..\Data\Analysis Results\MISQ\comments_processed.csv")


