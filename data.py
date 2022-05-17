import os
import pandas as pd
import re
import emoji
import numpy as np

def read_survey(pth,outfile=None):
    if outfile is not None and os.path.isfile(outfile):
        survey_df = pd.read_csv(outfile, encoding_errors='ignore')
    else:
        path = os.path.join(os.path.dirname(os.getcwd()),"Data\GSSWData-Full\Raw", pth)
        survey_df = pd.read_csv(path)
        survey_df = survey_df.rename(
            columns={'Q1.16_8': 'user_id', "Q7.1": "alcohol", "Q7.2": "marijuana", "Q7.3": "cocaine",
                     "Q7.4": "crack",
                     "Q7.5": "heroin", "Q7.6": "meth", "Q7.7": "ecstacy", "Q7.8": "needle", "Q7.9": "prescriptionDrug",
                     "Q1.3": "Age",
                     "Q2.2_1": "genderMale", "Q2.2_2": "genderFemale", "Q2.2_3": "genderTransMale",
                     "Q2.2_4": "genderTransFemale",
                     "Q2.2_5": "genderQueer", "Q2.2_6": "genderOther", "Q2.2_7": "genderDecline",
                     "Q2.1": "Race", "Q2.3": "SexualOrientation", "Q2.4": "Education",
                     "Q2.5": "AttendingSchool", "Q2.6": "Working", "Q2.7": "EverTravelled", "Q2.8": "Traveller",
                     "Q4.1": "PerceivedHealth",
                     "Q6.1": "hadSex",
                     "Q8.5": "OnlineTime",
                     "Q8.8": "DailySNTime", "Q8.9": "WeeklySNTime",
                     "Q8.11_1": "SR_Network", "Q8.11_3": "SR_School",
                     "Q8.11_4": "SR_News", "Q8.11_5": "SR_Knowledge",
                     "Q8.11_11": "SR_Messaging", "Q8.11_7": "SR_MeetPeople", "Q8.11_8": "SR_Sex",
                     "Q8.11_9": "SR_Entertainment", "Q8.11_16": "SR_Family", "Q8.11_17": "SR_Friends",
                     "Q8.11_18": "SR_FillTime", "Q8.11_19": "SR_Politics", "Q8.11_10": "SR_Other",
                     "Q8.11_10_TEXT": "SR_OtherText",
                     "Q8.12": "OnlineEasy",
                     "Q8.13.0": "onlineSexPartner",
                     "Q8.14": "findSexOnline",
                     "Q9.8": "Lonely1", "Q9.9": "Lonely2", "Q9.10": "Lonely3",
                     "Q9.14": "Jail",
                     "Q10.3": "beAttacked", "Q10.4": "beingThreathen", "Q10.5": "OthersAttacked",
                     "Q9.5_1": "depression1", "Q9.5_2": "depression2", "Q9.5_3": "depression3",
                     "Q9.5_4": "depression4", "Q9.5_5": "depression5", "Q9.5_6": "depression6",
                     "Q9.5_7": "depression7", "Q9.5_8": "depression8", "Q9.5_9": "depression9",
                     "Q9.3_1": "anxiety1", "Q9.3_2": "anxiety2", "Q9.3_3": "anxiety3",
                     "Q9.3_4": "anxiety4", "Q9.3_5": "anxiety5", "Q9.3_6": "anxiety6",
                     "Q9.3_7": "anxiety7"})
        #Delete the ones with no user ID
        survey_df.dropna(subset=['user_id'])
        print("After deleting people without FB IDs, there are {} people left".format(len(survey_df)))
        # Deleted= the one with multiple id
        survey_df = survey_df.loc[
            [i for i in survey_df.index if (survey_df.loc[i]['user_id'] != " " and survey_df.loc[i]['user_id'] != np.nan
                                            and survey_df.loc[i]['user_id'] != '149437232298631'
                                            and survey_df.loc[i]['user_id'] != '1979000205677699')]]

        print("After deleting duplicated people, there are {} people left".format(len(survey_df)))
        if outfile is not None:
            survey_df.to_csv(outfile)
    return survey_df

def read_data(pth, outfile = None,type="post"):
    #TODO: empty texts for post and comments - do i treat them differently, or same?
    if outfile is not None and os.path.isfile(outfile):
        df = pd.read_csv(outfile,encoding_errors='ignore')
    else:
        path = os.path.join(os.path.dirname(os.getcwd()), "Data\GSSWData-Full\Raw", pth)
        df = pd.read_csv(path,encoding_errors='ignore')
        pd.set_option('display.max_columns', None)
        print("The raw data has {} rows".format(len(df)))
        #delete the rows not authored by user - only for post
        if type == "post":
            df = df[(df['user_id'] == df['poster_id'])]
            print("After removing the posts by other people, the data has {} rows".format(len(df)))

        # delete the rows with invalid reaction
        df = df[(df["reaction_like"] > -1) & (df["reaction_love"] > -1) &
              (df["reaction_wow"]>-1) & (df['reaction_haha']>-1) &
              (df['reaction_sad'] >-1) & (df['reaction_angry']>-1) & (df['reaction_thankful']>-1) &
              (df['reaction_pride']>-1)]
        print("After removing the invalid reactions, the data has {} rows".format(len(df)))
        df['message'].fillna("", inplace=True)
        clean_messages = [clean_message(df.loc[i]['message']) for i in df.index]
        df.insert(len(df.columns), "text", clean_messages)

        # calculate the length of the texts and then remove empty texts
        df["text_length"] = df.apply(lambda row: len(row.text), axis=1)
        df = df[df["text_length"] >0]
        print("After removing empty text, the data has {} rows".format(len(df)))

        if outfile is not None:
            df.to_csv(outfile)
    return df

def demojize(m):
    emojis = pd.read_csv('emojis.csv')
    for eidx in emojis.index:
        emo = emojis.loc[eidx]['emoji']
        meaning = " :" + re.sub("(\w+_|[0-9]+)", "", emojis.loc[eidx]['meaning']) + ": "
        m = m.replace(emo, meaning)
        m = emoji.demojize(m)
    return m

def clean_message(message):

    def remove_url(m):
        m= re.sub("(http|www).*?\s+"," ",m+" ").strip()
        return m

    def replace_expletive(m):
        m = re.sub("(s|f|b)\*+", " [expletive] ", m)
        return m

    def remove_punkt(m):
        m = re.sub("(\?|\.|\\|\/|$|\@|!|\+|,)", " ", m)
        return m
    def remove_xspaces(m):
        m = re.sub("\s+", " ", m)
        return m.strip()
    #message = demojize(message)
    message = remove_url(message)
    message = replace_expletive(message)
    message = remove_punkt(message)
    charactor_list = ["<CM>","<NL>", "<QT>", "'", '"']


    for c in charactor_list:
        message = message.replace(c," ")
    message = remove_xspaces(message)
    return message

def calc_mean(df, cols):
    if type(cols) == list:
        for col in cols:
            yield df[col].mean()
    else:
        return df[cols].mean()

def aggregate_comments(comment_df):
    #TODO: when do i do the emotion - before or after aggregation
    dfgrp = comment_df.groupby('parent_id')
    #post_comments = {"post_id":[],"all_comments":[],"comment_number":[]}
    post_comments = []
    for grp in dfgrp:
        post_id = grp[0]
        grpdf = grp[1]
        comment_number = grpdf["message"].count()
        mean_length = grpdf["text_length"].mean()
        all_comments = " ".join(grpdf["text"].tolist())
        #post_comments['post_id'].append(post_id)
        #post_comments['all_comments'].append(all_comments)
        #post_comments['comment_number'].append(comment_number)
        post_comment = {"post_id":post_id, "all_comments":all_comments, "comment_number":comment_number}
        post_comments.append(post_comment)
    return post_comments

def aggregate_posts(post_df):
    #now this post already has two more columns: all comments, comment number
    #TODO: check which ID
    #TODO: how to join comments if missing post id
    dfgrp = post_df.groupby('user_id')
    people = []
    for grp in dfgrp:
        person_id = grp[0]
        grpdf = grp[1]
        post_number = grpdf["message"].count()
        #make sure the text being fed is correct
        all_posts = " ".join(grpdf["text"].tolist())
        all_comments = " ".join(grpdf["all_comments"].tolist())
        comment_number = grpdf["comment_number"].sum()
        mean_length = grpdf["text_length"].mean()
        #calculate reactions
        mean_like = grpdf["reaction_like"].mean()
        mean_love = grpdf["reaction_love"].mean()
        mean_wow = grpdf["reaction_wow"].mean()
        mean_haha = grpdf["reaction_haha"].mean()
        mean_sad = grpdf["reaction_sad"].mean()
        mean_angry = grpdf["reaction_angry"].mean()
        mean_thankful = grpdf["reaction_thankful"].mean()
        mean_pride = grpdf["reaction_pride"].mean()
        person = {"person_id": person_id, "all_posts": all_posts, "post_number": post_number,"mean_post_length":mean_length,
                  "mean_like": mean_like, "mean_love": mean_love, "mean_wow":  mean_wow, "mean_haha": mean_haha,
                   "mean_sad": mean_sad, "mean_angry": mean_angry,"mean_thankful": mean_thankful,"mean_pride":mean_pride,
                  "all_comments": all_comments,"comment_number": comment_number}
        people.append(person)
    return people


