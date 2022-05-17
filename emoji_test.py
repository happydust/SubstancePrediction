import os
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer # for sentiment
#path = os.path.join(os.path.dirname(os.getcwd()), "Data\\GSSWData-Full\\Raw\\emoji_test.csv")
#path = os.path.join(os.path.dirname(os.getcwd()), "Data\\GSSWData-Full\\Raw\\emoji_test.xlsx")


post_df = pd.read_csv("vader_test.csv",encoding_errors='ignore')
test = ": face with tears of joy : : face with tears of joy : : face with tears of joy : : face with tears of joy : : face with tears of joy :<NL>#Gucci<NL>#fdTrump"
#post_df = pd.read_excel(path, engine = "openpyxl")
#print(post_df.head)
analyser = SentimentIntensityAnalyzer()
for rowidx in post_df.index:
    row = post_df.loc[rowidx]
    message = row['message']
    print(message)
    scores = analyser.polarity_scores(message)
    print(scores)
score = analyser.polarity_scores(test)
print(score)

import re
m = "www.youtube.com/watch?v=HVuNiCXa5WA"
#m = re.sub("htt"," ",m)
m= re.sub("(http|www).*?\s+"," ",m+" ").strip()
print("after removing the url, the message is", m)