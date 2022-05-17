#We will perform topic modeling
import matplotlib.pyplot as plt
import nltk
import sys
#pip install git+https://github.com/rwalk/gsdmm.git
from gsdmm import MovieGroupProcess as mgp # From https://github.com/rwalk/gsdmm
from nltk.corpus import stopwords
import warnings
import pickle
import pandas as pd
import csv
from nltk import ngrams, FreqDist
from collections import Counter
### LDA ##########
# Gensim for topic modeling

import gensim.corpora as corpora
from gensim.models import CoherenceModel

# spacy for lemmatization
import tqdm
import spacy
import os

# def text_tokenize(data):
#     if(str(data))=="nan":
#         tokens =[]
#     else:
#         tokens = nltk.word_tokenize(str(data).lower())
#     return tokens
#
# def build_data(texts):
#     # read the posts - lst
#     documents = []
#     for i in texts:
#         documents.append(text_tokenize(i))
#     return documents
#
# def prep_documents(posts, nlp, add_stops=[]):
#     # documents are list of tokenized texts [["w1","w2","w3],["w1","w3","w5']
#     # in this function, we will clean the input documents and return a lemmitized list of tokenized texts
#     #bigrams
#     # Build the bigram and trigram models
#     # Define functions for stopwords, bigrams, trigrams and lemmatization
#     documents = build_data(posts)
#     stops = stopwords.words('english')
#     # TODO Need to add more stopwords for this dataset
#     cust_stops = ['hi', 'thanks', 'car', 'would', 'cheers', 'please', 'vehicle', 'hey', 'thank', "s", "could",
#                   "interested", "also", "be", "s", 'dannevirke', 'pammy', "do", "does", "doing", "did"]
#     stops.extend(cust_stops)
#     stops.extend(add_stops)
#
#     def remove_stopwords(texts, stops):
#         return [[word for word in simple_preprocess(str(doc)) if word not in stops] for doc in texts]
#
#     def make_bigrams(texts):
#         bigram = gensim.models.Phrases(documents, min_count=5, threshold=100)  # higher threshold fewer phrases.
#         bigram_mod = gensim.models.phrases.Phraser(bigram)
#
#         return bigram, [bigram_mod[doc] for doc in texts]
#
#     def make_trigrams(bmod, texts):
#         trigram = gensim.models.Phrases(bmod[documents], threshold=100)
#         trigram_mod = gensim.models.phrases.Phraser(trigram)
#         return trigram, [trigram_mod[bmod[doc]] for doc in texts]
#
#     def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
#         """https://spacy.io/api/annotation"""
#         texts_out = []
#         for sent in texts:
#             doc = nlp(" ".join(sent))
#             texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags and len(token.lemma_) >= 2])
#         return texts_out
#
#     # bigram = gensim.models.Phrases(documents['data'], min_count=5, threshold=100) # higher threshold fewer phrases.
#     # trigram = gensim.models.Phrases(bigram[documents['data']], threshold=100)
#
#     # Faster way to get a sentence clubbed as a trigram/bigram
#     # bigram_mod = gensim.models.phrases.Phraser(bigram)
#     # trigram_mod = gensim.models.phrases.Phraser(trigram)
#
#     #remove stop words
#
#     data_words_nostops = remove_stopwords(documents, stops=stops)
#     # print(data_words_nostops)
#
#     bigram_mod, data_words_bigrams = make_bigrams(texts=data_words_nostops)
#     # data_words_trigrams = make_trigrams(bigram_mod, texts=data_words_bigrams)
#
#     # Initialize spacy 'en' model, keeping only tagger component (for efficiency)
#
#     # Do lemmatization keeping only noun, adj, vb, adv
#     data_lemmatized = lemmatization(data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])
#     #for frequency
#     clean_words = []
#     for d in data_words_nostops:
#         clean_words.extend(d)
#
#     print("data_lemmatized looks like ", data_lemmatized[:5])
#     return data_lemmatized

def save(mod, path, **kwargs):
    modname = kwargs.get("modname", f"GSDMM(K={mod.K}, a={mod.alpha}, b={mod.beta}, n_iters={mod.n_iters})")
    overwrite = kwargs.get("ow", False)
    params = {"K": mod.K, "alpha": mod.alpha, 'beta': mod.beta,
              "vocab_size": mod.vocab_size, "D": mod.number_docs, "cluster_doc_count":
              mod.cluster_doc_count, "cluster_word_count": mod.cluster_word_count,
              "cluster_word_distribution": mod.cluster_word_distribution}

    fname = os.path.join(path, modname)+'.pkl'
    if os.path.isfile(fname):
        if overwrite is True:
            fstream = open(fname, mode='wb')
            pickle.dump(params, file=fstream)
            fstream.flush()
            fstream.close()
            print("Saved.")
        else:
            print(f"Model {modname} Already Exists, to replace set ow=True")
    else:
        fstream = open(fname, mode='wb')
        pickle.dump(params, file=fstream)


def load(path, **kwargs):
    modname = kwargs.pop("modname", f"GSDMM(K={kwargs.get('K')}, "
                                    f"a={kwargs.get('alpha')}, "
                                    f"b={kwargs.get('beta')}, "
                                    f"n_iters={kwargs.get('n_iters')})")
    fname = os.path.join(path, modname)+'.pkl'
    mod = None
    if os.path.isfile(fname) is False:
        print(f"Model {modname} - Not Found")
    else:
        fstream = open(fname, mode='rb')
        modparams = pickle.load(fstream)
        mod = mgp.from_data(**modparams)
        fstream.close()
    return mod


def build_model(docs, get_coherence=True,
                coherence_measure='c_v', n_words=20,
                min_docs = 5, clusters=10, load_existing=None,
                overwrite=False,
                **kwargs):
    '''
    Build gsdmm model
    Parameters
    ----------
    docs : documents list
    clusters : gsdmm parameter K
    kwargs :
        alpha : gsdmm parameter alpha (default=.1)
        beta : gsdmm parameter beta (default=.3)
        n_iters : gsdmm number of iterations (default=20)
    '''
    dictionary = corpora.Dictionary(docs)
    #dictionary.filter_extremes(no_below=3, no_above=.6, keep_n=2000000)
    dictionary.filter_extremes(no_below=3, no_above=.7, keep_n=2000000)
    bow_corpus = [dictionary.doc2bow(doc) for doc in docs]

    # You can edit the modname parameter for load or save it just
    #   defaults to GSDMM(K=__, alpha=__, beta=__, n_iters=__)

    # Change Path to Reflect Save Location (Without Filename)

    gsdmm_mod = None
    if load_existing is not None:
        gsdmm_mod = load(path=load_existing,
                         docs=docs, vocab_size=len(dictionary),
                         K=clusters, otherwise_fit=True, **kwargs)

    if gsdmm_mod is None:
        gsdmm_mod = mgp(K=clusters, **kwargs)
        gsdmm_mod.fit(docs=docs, vocab_size=len(dictionary))
        if load_existing is not None:
            save(gsdmm_mod, path=load_existing, ow=overwrite)

    doc_count = gsdmm_mod.cluster_doc_count
    print("Number of Documents per Cluster :", doc_count)
    dropped_clusters = []
    top_clusters = []
    for cl in range(len(doc_count)):
        if doc_count[cl] < min_docs:
            dropped_clusters.append(cl)
        else:
            top_clusters.append(cl)
    print(f"Dropped Clusters {dropped_clusters} because # of documents < {min_docs}")

    print("Most Important Cluster by # of Docs", top_clusters)
    cl_word_dist = gsdmm_mod.cluster_word_distribution
    topics = []
    for clust_num in top_clusters:
        wordfreq = sorted(cl_word_dist[clust_num].items(), key=lambda k: k[1], reverse=True)
        if len(wordfreq) >= n_words:
            wordfreq = wordfreq[:n_words]
            print(f"\nCluster {clust_num} : {dict(wordfreq)}")
            clust_topic = [k for k, v in wordfreq]
            topics.append(clust_topic)

    if get_coherence:
        cohere_mod = CoherenceModel(topics=topics,
                                    dictionary=dictionary,
                                    corpus=bow_corpus,
                                    texts=docs,
                                    coherence=coherence_measure)
        gsdmm_coherence = cohere_mod.get_coherence()
        print("Coherence GSDMM Model: ", gsdmm_coherence)

    return gsdmm_mod

def run(data_lemmas, cluster_number, **kwargs):

    #docs1_lemma = [[t for t in doc if t != 'do'] for doc in docs1_lemma]
    print("The tokens are", data_lemmas)
    mod = build_model(docs=data_lemmas, clusters=cluster_number,
                      alpha=kwargs.get("alpha", .25), beta=kwargs.get('beta', .4), n_iters=kwargs.get('n_iters', 50),
                      min_docs=kwargs.get("min_docs", 1),
                      load_existing="models\\",
                      overwrite=True)
    data_topic = []
    #id_column = data['post_id']
    #data_column = data['data']

    for i in range(len(data_lemmas)):
        doc = data_lemmas[i]
        best_topic = mod.choose_best_label(doc)
        #TODO: take care of the vector
        record_dict = {"topic":best_topic[0], 'topic probability':best_topic[1], "vector": mod.score(doc)}
        #print(f"For Document: \n{doc}\nBest Label: {mod.choose_best_label(doc)}\nScore??: {mod.score(doc)}\n##############\n")
        data_topic.append(record_dict)
    print(data_topic[:2])
    return data_topic
    #path = f"..\\..\\data\\GSDMM_topics_{cluster_number}.csv"
    #with open(path, newline="", mode='w') as f:
        #writer = csv.writer(f)
        #writer.writerow(['ID', 'Message', 'DominateTopic', 'Dominate Topic Probability','Vectors'])
        #writer.writerows(data_topic)
        #f.close()


