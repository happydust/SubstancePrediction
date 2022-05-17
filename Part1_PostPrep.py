import pandas as pd
import os
from tqdm import tqdm
import text2emotion as te # for emotion
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer # for sentiment
import emoji
import nltk

#read the data
path = os.path.join(os.path.dirname(os.getcwd()), "Data\\GSSWData-Full\\Raw\\posts.csv")
post_df = pd.read_csv(path)
print(post_df.head)


analyser = SentimentIntensityAnalyzer()

for new_col in ['author_by_user', 'messageLength', 'negativity', 'positivity', 'neutrality', 'compound','happy','angry','sad','surprise','fear','emotional','clean_text']:
    post_df.insert(len(post_df.columns), new_col, range(len(post_df.index)))

analyser = SentimentIntensityAnalyzer()

#TODO: de-emoji
for rowidx in tqdm(post_df.index):
    row = post_df.loc[rowidx]
    message = str(row['message'])
    if(message == "nan"):
        message = ""
    #sentiment detection
    post_df['author_by_user'].loc[rowidx] = (1 if row['user_id'] == row['poster_id'] else 0)
    score = analyser.polarity_scores(message)
    # emotion detection
    emotions = te.get_emotion(message)

    message_length = len(message)
    post_df['messageLength'].loc[rowidx] = message_length

    # De-emoji
    clean_txt = emoji.demojize(u'{}'.format(message))
    # print(clean_txt)
    # clean_txt = clean_txt.replace("<NL>", " ") # new line
    # clean_txt = clean_txt.replace("<CM>", " ") #
    # patt = re.compile(":.*?:")
    # matches = re.findall(patt, clean_txt)

    # #I cannot understand this part below
    # for match in matches:
    #     non_match_strings = [" ", "<", ")", "(", ">"]
    #     if any([match.find(x) > -1 for x in non_match_strings]):
    #         pass
    #     else:
    #         m = match.replace(":", "")
    #         m = " EMOJI_"+m
    #         clean_txt = clean_txt.replace(match, m)
    #         emojis.append(m)

    # Sentiment after text cleaning
    score_new = analyser.polarity_scores(clean_txt)
    if message_length == 0:
        negativity = -1
        positivity = -1
        neutrality = -1
        compound = -1
        happy, angry, surprise, sad, fear = -1, -1, -1, -1, -1
        new_negativity, new_positivity, new_neutrality, compound = -1, -1, -1, -1
    else:
        negativity = score['neg']
        positivity = score['pos']
        neutrality = score['neu']
        compound = score['compound']
        happy = emotions['Happy']
        angry = emotions['Angry']
        surprise = emotions['Surprise']
        sad = emotions['Sad']
        fear = emotions['Fear']
    post_df['negativity'].loc[rowidx] = negativity
    post_df['positivity'].loc[rowidx] = positivity
    post_df['neutrality'].loc[rowidx] = neutrality
    post_df['compound'].loc[rowidx] = compound
    post_df['happy'].loc[rowidx] = happy
    post_df['angry'].loc[rowidx] = angry
    post_df['surprise'].loc[rowidx] = surprise
    post_df['sad'].loc[rowidx] = sad
    post_df['fear'].loc[rowidx] = fear
    post_df['clean_text'].loc[rowidx] = clean_txt


    if emotions['Happy'] == 0 and emotions['Angry'] == 0 and emotions['Sad'] == 0 and emotions['Surprise'] == 0 and \
            emotions['Fear'] == 0:
        post_df['emotional'].loc[rowidx] = 0
    else:
        post_df['emotional'].loc[rowidx] = 1

# print(post_df.head())
# Check the sentiment result file, it looks like that the sentiment is not taken care of with the emojis
post_df.to_csv("..\Data\Analysis Results\MISQ\post_processed.csv")
#post_df.to_excel("..\Data\\GSSWData-Full\\2-SentimentResult.xlsx")
