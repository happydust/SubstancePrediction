import data
import csv
import pandas as pd
pd.set_option('display.max_columns', None)
from tqdm import tqdm
from analysis import sentiment, gsdmm_mod, prep
from gensim.models.coherencemodel import CoherenceModel
import sys
# read comments
# read posts
post_df = data.read_data("posts.csv","C:\\Users\\Tianjie.Deng\\Dropbox\\PROF PREP\\DCB Facebook\\Facebook "
                                     "Data Analysis\\Data\\Analysis Results\\MISQ\\posts_cleaned.csv",type="post")
comment_df = data.read_data("comments.csv","C:\\Users\\Tianjie.Deng\\Dropbox\\PROF PREP\\DCB Facebook\\Facebook "
                                     "Data Analysis\\Data\\Analysis Results\\MISQ\\comments_cleaned.csv",type="comment")
#read survey
survey_df = data.read_survey("SurveyData_RawWithDup.csv","C:\\Users\\Tianjie.Deng\\Dropbox\\PROF PREP\\DCB Facebook\\Facebook "
                                     "Data Analysis\\Data\\Analysis Results\\MISQ\\survey_cleaned.csv")

# data preprocessing
# sentiment analysis

# topic modeling - gsdmm
# Todo: de-emojize here becuase we need that for topic modelling. Right now the topic modeling is not using any emojis, which is fine.
#prepped_lemmas_msg = prep.prep_documents(post_df['text'].to_list(), nlp=prep.nlp, add_stops=['get'])
#gsd_topics = gsdmm_mod.run(prepped_lemmas_msg, cluster_number=10, beta=.5, alpha=.3)

# Aggregate comments first
# Note this is a list of dictionaires
comment_agg = data.aggregate_comments(comment_df)
#print("Aggregated comments look like: ", comment_agg[:10])
comment_agg_df = pd.DataFrame(comment_agg)
#print("The comment dataframe looks like:", comment_agg_df.head())
#scores_comments = sentiment.sentiment_analysis(comment_agg, column_name="all_comments")
#print("The sentiment scores of comments are", scores_comments[:10])
# Join with post
post_joined = post_df.merge(comment_agg_df, how='left', on='post_id')
# Dealing with empty comments - bascially, posts with no comments
post_joined['comment_number'] = post_joined['comment_number'].fillna(0)
post_joined['all_comments'] = post_joined['all_comments'].fillna("")
print("Length of all posts are",len(post_joined.index))
# Aggregate posts at individual level
post_agg = data.aggregate_posts(post_joined)

# Sentiment analysis here
people = []
for i in tqdm(range(len(post_agg))):
    person = post_agg[i]
    all_posts = person['all_posts']
    all_comments = person['all_comments']
    post_sentiment = sentiment.sentiment_analysis(all_posts,type="post")
    comments_sentiment = sentiment.sentiment_analysis(all_comments,type="comment")
    person.update(post_sentiment)
    person.update(comments_sentiment)
    people.append(person)
people_df = pd.DataFrame(people)
people_df.to_csv("C:\\Users\\Tianjie.Deng\\Dropbox\\PROF PREP\\DCB Facebook\\Facebook Data Analysis\\Data\\Analysis Results\\MISQ\\aggregated_posts.csv")


# join the data
# machine learning