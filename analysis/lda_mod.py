import csv
import matplotlib.pyplot as plt
import warnings
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
import os
import pandas as pd
from pprint import PrettyPrinter as pprint
#from run_main import prepped_lemmas_agg, post_agg
from sentiment import sentiment_analysis
from analysis import prep
import nltk
nltk.download('omw-1.4')

aggregated_posts =pd.read_csv("C:\\Users\\Tianjie.Deng\\Dropbox\\PROF PREP\\DCB Facebook\\Facebook "
                                     "Data Analysis\\Data\\Analysis Results\\MISQ\\aggregated_posts.csv")
#prepped_lemmas_agg = prep.prep_documents(aggregated_posts.tolist(), nlp=prep.nlp)


def matrices(coherence_values, coherence_measure, perplexity_values, level, method, stop, step=1, start=0):
    x = range(start, stop, step)
    f = open(f'C:\\Users\\Tianjie.Deng\\Dropbox\\PROF PREP\\DCB Facebook\\Facebook Data Analysis\\MISQCase\\models\\{level}_matrices\\matrix_({method}_{start}, {stop}, {step},{coherence_measure}).csv', 'w',newline='')
    writer = csv.writer(f)
    writer.writerow(["Number of Topics","Coherence","Perpelxity"])
    for m, cv, p in zip(x, coherence_values, perplexity_values):
        #print("Num Topics =", m, " has Coherence Value of", round(cv, 4), "has perpelixty", round(p,4))
        writer.writerow([m,cv,p])
    f.close()
    print("Matrix saved!")

def plot(coherence_values, perplexity_values, level, method, stop, start=0, step=1):
    x = range(start, stop, step)
    print("The coherence values", coherence_values)
    print("The perplexity values", perplexity_values)
    plt.plot(x, coherence_values)
    plt.xlabel("Num Topics")
    plt.ylabel("Coherence score")
    plt.legend(("coherence_values"), loc='best')
    plt.savefig(f'C:\\Users\\Tianjie.Deng\\Dropbox\\PROF PREP\\DCB Facebook\\Facebook Data Analysis\\MISQCase\\models\\{level}_plots\\Cluster vs Coherence_{method}_{start}, {stop}, {step}.png')
    plt.show()


    plt.plot(x, perplexity_values)
    plt.xlabel("Num Topics")
    plt.ylabel("Perplexity score")
    plt.legend(("perplexity_values"), loc='best')
    plt.savefig(f'C:\\Users\\Tianjie.Deng\\Dropbox\\PROF PREP\\DCB Facebook\\Facebook Data Analysis\\MISQCase\\models\\{level}_plots\\Cluster vs Perplexity_{method}_{start}, {stop}, {step}.png')
    plt.show()
    print(f"{level} plots saved!")


def fit_models(dictionary, corpus, texts, limit, level, method, coherence_measure, start=2, step=1, use_saved=True, ow=False,
               save_matrices=True, save_plots=True, get_coherence=True, get_perplexity=True,
               ):
    """
    Compute c_v coherence for various number of topics

    Parameters:
    ----------
    dictionary : Gensim dictionary
    corpus : Gensim corpus
    texts : List of input texts
    limit : Max num of topics
    start : Minimum num of topics (default = 2)
    step : Num of topics increment (default = 3)
    use_saved : Use saved model (default=True)
    ow : Overwrite saved model [if exists] (default=False)
    level : Data Level ('listinglevel' or 'questionlevel')

    Returns:
    -------
    model_list : List of LDA topic models
    coherence_values : Coherence values corresponding to the LDA model with respective number of topics
    """
    coherence_values = []
    perplexity_values =[]
    model_list = []

    for num_topics in range(start, limit, step):
        mod_name = f"lda_{method}_{num_topics}.model"

        exists = os.path.isfile(f"C:\\Users\\Tianjie.Deng\\Dropbox\\PROF PREP\\DCB Facebook\\"
                                f"Facebook Data Analysis\\MISQCase\\models\\{level}_models\\{mod_name}")
        if exists is True and use_saved is True and ow is False:
            print(f"Loading {level} Model {num_topics}...")
            model = gensim.models.ldamodel.LdaModel.load(f"C:\\Users\\Tianjie.Deng\\Dropbox\\PROF PREP\\DCB Facebook\\"
                                f"Facebook Data Analysis\\MISQCase\\models\\{level}_models\\{mod_name}")
        else:
            print(f"Exists: {exists} -- Use Saved: {use_saved} -- Overwrite: {ow}")
            print(f"Building {level} Model {num_topics}...")
            model = gensim.models.ldamodel.LdaModel(corpus=corpus, num_topics=num_topics, id2word=dictionary,
                                                    #random_state=100+(num_topics*100),
                                                    random_state=100,
                                                    update_every=1,
                                                    chunksize=102,
                                                    passes=10,
                                                    alpha='auto',
                                                    per_word_topics=True
                                                    )
            print("Saving Model...")
            model.save(f"C:\\Users\\Tianjie.Deng\\Dropbox\\PROF PREP\\DCB Facebook\\"
                                f"Facebook Data Analysis\\MISQCase\\models\\{level}_models\\{mod_name}")

        model_list.append(num_topics)
        if get_coherence:
            coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary,
                                            coherence=coherence_measure)
            coherence_values.append(coherencemodel.get_coherence())
        if get_perplexity:
            perplexity = model.log_perplexity(corpus)
            perplexity_values.append(perplexity)

    if save_matrices:
        print("Building Matrices...")
        matrices(coherence_values=coherence_values, coherence_measure= coherence_measure, perplexity_values=perplexity_values, stop=limit,
                 start=start, step=step, level=level,method=method)
    if save_plots:
        print("Building Plots...")
        plot(coherence_values=coherence_values, perplexity_values=perplexity_values,
             stop=limit, start=start, step=step, level=level,method=method)
    return model_list, coherence_values, perplexity_values

def fetch_optimal_candidates(candidate_list, level,method):
    #load the optimal model
    #topics_optimals = [3,5,7,11]# the elbows
    for num_topics_optimal in candidate_list:
        optimal_model = gensim.models.ldamodel.LdaModel.load(f"C:\\Users\\Tianjie.Deng\\Dropbox\\PROF PREP\\DCB Facebook\\"
                                f"Facebook Data Analysis\\MISQCase\\models\\{level}_models\\lda_{method}_{num_topics_optimal}.model")
        #optimal_model = model_list[7]
        optimal_model.show_topics(formatted=False)
        topics = optimal_model.print_topics(num_words=25)
        pp = pprint(4)
        pp.pprint(topics)

    #optimal_number = 7
    #lda = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=id2word_).load(f"listinglevel_models\\lda_{optimal_number}.model")

def auto_select_candidates(coherence_values, mod_nums, n=-1):
    candidates = []
    for c in range(len(coherence_values)-2):
        mod_num = mod_nums[c+1]
        c1 = coherence_values[c]
        c2 = coherence_values[c+1]
        c3 = coherence_values[c+2]
        if c1 < c2 and c2 > c3:
            candidates.append(mod_num+1)
    if n == -1:
        return candidates
    else:
        return candidates[:n]


def dominant_topic(lst, corpus,optimal_number, lda_level,method):
    print(f"loading {lda_level} topic model")
    lda = gensim.models.ldamodel.LdaModel.load(f"C:\\Users\\Tianjie.Deng\\Dropbox\\PROF PREP\\DCB Facebook\\"
                                f"Facebook Data Analysis\\MISQCase\\models\\{lda_level}_models\\lda_{method}_{optimal_number}.model")
    print("In application to find dominate_topic, corpus looks like ",corpus[:3])
    person_topic = []
    for i in range(len(corpus)):
        person_item = lst[i]
        print(person_item)
        corpitem = corpus[i]
        probability = lda[corpitem][0] # >> Example: [.3, .3, .2, .2]
        vector_dict ={k: 0.0 for k in range(0,optimal_number)}
        vector_dict.update(probability)
        person_item.update(vector_dict)
        #vector_dict.update({f"topic_{k}": prob_dict[k] for k in prob_dict})
        #all_posts = person_item["all_posts"]
        #all_comments = person_item["all_comments"]
        #post_sentiment = sentiment_analysis(all_posts)
        #comment_sentiment = sentiment_analysis(all_comments)
        #person_item.update(post_sentiment)
        #person_item.update(comment_sentiment)
        person_topic.append(person_item)
    path = f"C:\\Users\\Tianjie.Deng\\Dropbox\\PROF PREP\\DCB Facebook\\Facebook Data Analysis\\Data\\Analysis Results\\MISQ" \
           f"\\person_topics_{method}_{optimal_number}.csv"
    with open(path,newline="",mode='w',encoding="utf-8") as f:
        keys = person_topic[0].keys()
        dict_writer = csv.DictWriter(f, keys)
        dict_writer.writerows(lst)
        f.close()

def run(candidate_selection='manual', method="lemma", level="authorlevel", num_candidates=4,**kwargs):
    data_lemmas = prep.prep_documents(aggregated_posts['all_posts'].tolist(), nlp=prep.nlp,add_stops=["get"],method=method)
    #print("prepped_lemmas_agg is: ", prepped_lemmas_agg[:10])

    print(f"Build Using {level} level data using {method} method\n")
    # Create Dictionary/Vocabulary
    id2word_lda = corpora.Dictionary(data_lemmas)
    # Remove rare and common tokens
    id2word_lda.filter_extremes(no_below=0.01, no_above=0.7)
    # Create Corpus
    corpus_lda = [id2word_lda.doc2bow(text) for text in data_lemmas]
    print("The documents look like ", data_lemmas[:3])
    print("The corpus looks like ", corpus_lda[:3])

    model_list, coherence_values, perplexity_values = fit_models(dictionary=id2word_lda, corpus=corpus_lda,
                                                                 texts=data_lemmas, coherence_measure='c_v',
                                                                 start=5, limit=25, method=method,
                                                                 step=1, level=level,
                                                                 save_matrices=True, save_plots=True)
    if candidate_selection == 'auto':
        candidate_models = auto_select_candidates(coherence_values, mod_nums=model_list, n=num_candidates)
    else:
        candidate_models = input("Enter list of candidates: ")
        candidate_models = [int(c.strip()) for c in candidate_models.split(",")]

    fetch_optimal_candidates(candidate_models,level,method=method)
    select_optimal_model = input("Enter Optimal Model: ")
    print("Now applying ...")
    print("The corpus of the new documents are ", )
    aggregated_posts_lst = aggregated_posts.to_dict('records')
    dominant_topic(lst= aggregated_posts_lst, corpus = corpus_lda,
                   optimal_number=int(select_optimal_model),
                   lda_level=level,method=method)
#TODO check the raw data for run in the main, for the auction project
if __name__=="__main__":
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        #run(raw_data=qdata, lda_level_input="questionlevel", application_level_input="listinglevel")
        run()