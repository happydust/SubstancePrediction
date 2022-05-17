#We will perform word cloud
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
import pandas as pd
import os
import csv
from nltk import ngrams, FreqDist
from collections import Counter

path = os.path.join(os.path.dirname(os.getcwd()), "Data\Analysis Results\MISQ\post_processed.csv")
post_df = pd.read_csv(path)
mask0 = post_df['messageLength']>0
mask1 = post_df['author_by_user']>0
filtered = post_df.where(mask0 & mask1)
text = " ".join([str(i) for i in filtered['message'].to_list()])
# text = " ".join([midx for midx in post_df.index if post_df['messageLength'][midx]>0 and post_df['author_by_user']>0])

#Word cloud
#text = "Try try try and see"
#text = " ".join([review for review in df['description'] if df['binary'] == 1])
stopwords = stopwords.words('english')
stopwords.extend(['nan','CM','NL','QT'])
print(stopwords)
wordcloud = WordCloud(stopwords=stopwords).generate(text)


plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()
