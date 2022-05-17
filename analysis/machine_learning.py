#Prediction and evaluation
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, BaggingClassifier, StackingClassifier
from sklearn.metrics import classification_report, precision_score, roc_auc_score, roc_curve,plot_roc_curve, recall_score
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, LinearSVC
#import tensorflow as tf
import numpy as np
import random
from pandas import DataFrame, read_csv
import os
#from xx import user_df

# Read in the data - user_df
#@ignore_warnings(category=ConvergenceWarning)
def evaluation(model,X_test_data,y_test,y_pred):
    accuracy = model.score(X_test_data, y_test)
    rpt = classification_report(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    #print(f"{str(clf.__name__)} Model Performance: \n\n {rpt}")
    #print(f"{str(clf.__name__)} AUC: \n\n {auc}")
    return accuracy, rpt, auc, precision, recall

# def ml(clf, X, y, rand, samp_id, test_size=0.20,*args, **kwargs):
#     # print(f'Fitting model {str(clf.__name__)}{(" - "+str(kwargs.get("base_estimator")) if "base_estimator" in kwargs else "")} on X{samp_id}...')
#     save_data_arg = kwargs.pop('save_data', True)
#     save_model_arg = kwargs.pop("save_model", True)
#
#     train_file = f"data\\train_idx.csv"
#     test_file = f"data\\test_idx.csv"
#     model_file = f"models\\{clf.__name__}(base_est={kwargs.get('base_estimator', 'N/A')}," \
#                  f"n_estimators={kwargs.get('n_estimators', 'N/A')}," \
#                  f"bootstrap={kwargs.get('bootstrap', 'N/A')}," \
#                  f"max_samples={kwargs.get('max_samples', 'N/A')}, id={samp_id}).fitted"
#
#
#     if os.path.isfile(train_file) is False or (save_data_arg is True and samp_id == 0):
#         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=rand)
#         X_train_idx, X_train_data = [x[0] for x in X_train], [x[1] for x in X_train]
#         X_test_idx, X_test_data = [x[0] for x in X_test], [x[1] for x in X_test]
#         #length of the vector
#         v_size = len(X_train_data[0])
#
#         #Print out the train and testing dataset out
#         train_df = DataFrame(data=list(zip(X_train_idx, y_train)), columns=['Xidx', "y"])
#         test_df = DataFrame(data=list(zip(X_test_idx, y_test)), columns=['Xidx', "y"])
#         train_df.to_csv(train_file)
#         test_df.to_csv(test_file)
#
#     elif os.path.isfile(train_file) and (save_data_arg is False or (save_data_arg is True and samp_id > 0)):
#         print("Loading Training and Testing Set")
#         train_df = read_csv(train_file)
#         test_df = read_csv(test_file)
#
#         X_train_idx, X_train_data = train_df['Xidx'].to_list(), [X[i][1] for i in range(len(X)) if X[i][0] in train_df['Xidx']]
#         X_test_idx, X_test_data = test_df['Xidx'].to_list(), [X[i][1] for i in range(len(X)) if X[i][0] in test_df['Xidx']]
#         y_train, y_test = train_df['y'].to_list(), test_df['y'].to_list()
#         v_size = len(X_train_data[0])
#
#     if save_model_arg is True or os.path.isfile(model_file) is False:
#         mod = clf.__call__(*args, **kwargs)
#         mod.fit(X_train_data, y_train)
#
#         pickle.dump(mod, open(model_file, mode='wb'))
#
#     elif os.path.isfile(model_file) and save_model_arg is False:
#         mod = pickle.load(open(model_file, mode='rb'))
#
#     y_pred = mod.predict(X_test_data)
#
#     pred_test_match = [(y_pred[X_test_idx.index(i)] if i in X_test_idx else -1) for i in user_df.index]
#     user_df.insert(len(user_df.columns), f"{samp_id}_predicted", pred_test_match)
#
#     accuracy, rpt, auc,precision, recall = evaluation (mod,X_test_data,y_test,y_pred)
#     #accuracy = mod.score(X_test_data, y_test)
#     #rpt = classification_report(y_test, y_pred)
#     #auc = roc_auc_score(y_test,y_pred)
#     #precision = precision_score(y_test,y_pred)
#     #recall = recall_score(y_test,y_pred)
#     #print(f"{str(clf.__name__)} Model Performance: \n\n {rpt}")
#     #print(f"{str(clf.__name__)} AUC: \n\n {auc}")
#     return v_size, auc, accuracy,precision, recall,pred_test_match


def build_classifier(X_train, y_train, X_test, y_test,men_xtest, men_ytest, women_xtest,women_ytest, young_xtest, young_ytest,
                     baseclf, n_estimators, **kwargs):
    meta_clf = BaggingClassifier(baseclf, n_estimators=n_estimators, **kwargs)
    meta_clf.fit(X_train, y_train)
    if type(baseclf).__name__ =='DecisionTreeClassifier':
        feature_importances = np.mean([
            tree.feature_importances_ for tree in meta_clf.estimators_], axis=0)
        print(feature_importances)
    else:
        feature_importances = np.mean([
            model.coef_ for model in meta_clf.estimators_], axis=0)
        print(feature_importances)
    pred = meta_clf.predict(X_test)
    actual = y_test
    men_pred = meta_clf.predict(men_xtest)
    #print('The predicted values are: ',pred)
    accuracy, rpt, auc, precision, recall = evaluation(meta_clf, X_test, actual, pred)
    print("The overall performance is", rpt)
    print("the overall auc is", auc)
    accuracy_m, rpt_m, auc_m, precision_m, recall_m = evaluation(meta_clf, men_xtest, men_ytest, men_pred)
    print("For men, the performance is", rpt_m)

def user_val_transform(x):
    if int(x) == 1:
        return 0
    else:
        return 1


def prep_data(X_feats, y_feat,test_size=0.30):
    pd.set_option('display.max_columns', None)
    #Read the user data
    author_df = pd.read_csv("C:\\Users\\Tianjie.Deng\\Dropbox\\PROF PREP\\DCB Facebook\\Facebook Data Analysis\Data\Analysis Results\MISQ\\person_topics_lemma_19.csv",encoding="utf-8")
    survey_df = pd.read_csv("C:\\Users\\Tianjie.Deng\\Dropbox\\PROF PREP\\DCB Facebook\\Facebook Data Analysis\Data\Analysis Results\MISQ\\survey_cleaned.csv")
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
    print(f"There are {len(survey_df)} respondents.")
    # Deleted the one with multiple id
    survey_df = survey_df.loc[
        [i for i in survey_df.index if (survey_df.loc[i]['user_id'] != " " and survey_df.loc[i]['user_id'] != np.nan
                                        and survey_df.loc[i]['user_id'] != '149437232298631'
                                        and survey_df.loc[i]['user_id'] != '1979000205677699')]]
    print(f"After deleting duplicated IDs, there are {(len(survey_df))} respondents.")
    user_df = author_df.merge(survey_df, how='inner', left_on="person_id", right_on="user_id")

    print(f"The length of the user df after merging is {len(user_df)}")
    user_df['marijuana_user'] = user_df['marijuana'].apply(user_val_transform)
    user_df['alcohol_user'] = user_df['alcohol'].apply(user_val_transform)
    user_df['cocaine_user'] = user_df['cocaine'].apply(user_val_transform)
    user_df['crack_user'] = user_df['crack'].apply(user_val_transform)
    user_df['heroin_user'] = user_df['heroin'].apply(user_val_transform)
    user_df['meth_user'] = user_df['meth'].apply(user_val_transform)
    user_df['ecstacy_user'] = user_df['ecstacy'].apply(user_val_transform)
    user_df['needle_user'] = user_df['needle'].apply(user_val_transform)
    user_df['prescriptionDrug_user'] = user_df['prescriptionDrug'].apply(user_val_transform)
    print("The marijuana user column looks like", user_df['marijuana_user'])
    X = user_df[X_feats].to_numpy()
    y = user_df[["genderMale", 'Age', 'genderFemale', y_feat]].to_numpy()

    # split train_test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    print(y_test)

    men_idx, female_idx = [], []
    young_idx, old_idx = [], []
    for i in range(len(y_test)):
        if y_test[i][0] == '1':
            men_idx.append(i)
        elif y_test[i][2] == '1':
            female_idx.append(i)
        if y_test[i][1] in ['1','2','3','4','5']:
            young_idx.append(i)
        elif y_test[i][1] in ['6', '7']:
            old_idx.append(i)
    # men_idx = [i for i in range(len(y_test)) if y_test[i][0] == '1']
    # print("men idex are", men_idx)
    # female_idx = [i for i in range(len(y_test)) if y_test[i][2]=='1']
    # young_idx = [i for i in y_test.index if y_test[i][1] in ['1','2','3','4','5']]
    # old_idx = [i for i in range(len(y_test)) if y_test[i][1] in ['6','7']]
    y_train = y_train[:, 3]
    y_train = y_train.astype('int')
    men_xtest = X_test[men_idx, :]
    #print("men x test is", men_xtest)
    men_ytest = y_test[men_idx, 3]
    men_ytest = men_ytest.astype('int')
    women_xtest = X_test[female_idx, :]
    women_ytest = y_test[female_idx, 3]
    women_ytest = women_ytest.astype('int')
    young_xtest = X_test[young_idx, :]
    young_ytest = y_test[young_idx, 3]
    youny_ytest = young_ytest.astype('int')
    y_test = y_test[:, 3]
    y_test = y_test.astype('int')


    return X_train, X_test, y_train, y_test, men_xtest, men_ytest, women_xtest,women_ytest, young_xtest, young_ytest

x_feature_sets = {"sentiment": ["post_sentiment", "post_number",'mean_post_length','mean_like',
                                'mean_love','mean_wow','mean_haha',
                                'mean_sad','mean_angry',
                                'post_happy','post_angry','post_surprise','post_sad','post_fear',
                                "comment_sentiment","comment_number",
                                "topic1","topic2","topic3","topic4","topic5","topic6","topic7",
                                "topic8","topic9","topic10","topic11","topic12","topic13","topic14",
                                "topic15","topic16","topic17","topic18","topic19"],
                  "female_dominated":['4', '5', '6']}

X_train, X_test, y_train, y_test, men_xtest, men_ytest, women_xtest,women_ytest, young_xtest, young_ytest = prep_data(X_feats=x_feature_sets['sentiment'], y_feat="marijuana_user",test_size=0.25)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

estimators = [('log',LogisticRegression()), ('tree', DecisionTreeClassifier()),('svc',SVC()),('NB',GaussianNB())]
#build_classifier(X_train, y_train, X_test, y_test, LogisticRegression, 100, bootstrap=True)
#build_classifier(X_train, y_train, X_test, y_test, men_xtest, men_ytest, women_xtest,women_ytest, young_xtest, young_ytest, LogisticRegression(), 10000, bootstrap=True)

build_classifier(X_train, y_train, X_test, y_test, men_xtest, men_ytest, women_xtest,women_ytest, young_xtest, young_ytest, DecisionTreeClassifier(), 100000,
                 max_samples=0.8, random_state=111, bootstrap=True, save_model=True)
# define the feature sets here

#for set_name, feat_set in enumerate(x_feature_sets):
    #X_train, y_train, X_test, y_test = prep_data(X_feats=feat_set, y_feat='target')
    #clf = build_classifier()

