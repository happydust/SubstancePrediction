# importing all necessary modules
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import pandas as pd
import spacy
from nltk import word_tokenize
from nltk.corpus import stopwords
import tqdm
nlp = spacy.load('en_core_web_sm')


# Read the data
post_df = pd.read_csv("C:\\Users\\tianjie.deng\\Dropbox\\PROF PREP\\DCB Facebook\\"
                 "Facebook Data Analysis\\Data\\Analysis Results\\MISQ\\aggregated_posts.csv")
# Clean the data
messages = post_df['all_posts'].to_list()

tokens = []
allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']
for m in messages:
    doc = nlp(m.lower())
    tokens.extend([token.lemma_ for token in doc if token.pos_ in allowed_postags and len(token.lemma_) >= 2])
print(tokens[:2])

words = " ".join(tokens)
# Prepare stopwords
stopwords = stopwords.words('english')
add_stopwords = ['hi', 'thanks', 'would', 'cheers', 'please', 'hey', "could","also", "be", "s", "do", "does", "doing",
                  "did","think",'shit','fuck', 'want','lol']
stopwords.extend(add_stopwords)

# Make the word cloud
wordcloud = WordCloud(width=800, height=800,
                      background_color='white',
                      stopwords=stopwords,
                      min_font_size=10).generate(words)

# Plot the WordCloud image
plt.figure(figsize=(8, 8), facecolor=None)
plt.imshow(wordcloud)
plt.axis("off")
plt.tight_layout(pad=0)
plt.savefig('C:\\Users\\tianjie.deng\\Dropbox\\PROF PREP\\DCB Facebook\\Facebook Data Analysis\\Data\\Analysis Results\\MISQ\\wordcloud.png')
plt.show()