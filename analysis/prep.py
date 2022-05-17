from gensim.utils import simple_preprocess
from nltk.stem.porter import *
import gensim
import spacy
from nltk import word_tokenize
from nltk.corpus import stopwords

nlp = spacy.load('en_core_web_sm')

def text_tokenize(data):
    if str(data) =="nan":
        tokens =[]
    else:
        #lower case everything
        tokens = word_tokenize(str(data).lower())
    return tokens

def build_data(texts):
    # read the posts - lst
    documents = []
    for i in texts:
        documents.append(text_tokenize(i))
    return documents

def prep_documents(posts, nlp, add_stops=[],method="stem"):
    # documents are list of tokenized texts [["w1","w2","w3],["w1","w3","w5']
    # in this function, we will clean the input documents and return a lemmitized list of tokenized texts
    #bigrams
    # Build the bigram and trigram models
    # Define functions for stopwords, bigrams, trigrams and lemmatization
    documents = build_data(posts)
    stops = stopwords.words('english')
    # TODO Need to add more stopwords for this dataset
    # TODO make all lower cases
    cust_stops = ['hi', 'thanks', 'would', 'cheers', 'please', 'hey', "could","also", "be", "s", "do", "does", "doing",
                  "did","think",'shit','fuck', 'want','lol']
    stops.extend(cust_stops)
    stops.extend(add_stops)

    def remove_stopwords(texts, stops):
        return [[word for word in simple_preprocess(str(doc)) if word not in stops] for doc in texts]

    def make_bigrams(texts):
        bigram = gensim.models.Phrases(documents, min_count=5, threshold=100)  # higher threshold fewer phrases.
        bigram_mod = gensim.models.phrases.Phraser(bigram)

        return bigram, [bigram_mod[doc] for doc in texts]

    def make_trigrams(bmod, texts):
        trigram = gensim.models.Phrases(bmod[documents], threshold=100)
        trigram_mod = gensim.models.phrases.Phraser(trigram)
        return trigram, [trigram_mod[bmod[doc]] for doc in texts]

    def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
        """https://spacy.io/api/annotation"""
        texts_out = []
        for sent in texts:
            doc = nlp(" ".join(sent))
            if method == "lemma":
                texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags and len(token.lemma_) >= 2])
            else:
                stemmer = PorterStemmer()
                texts_out.append([stemmer.stem(token.text) for token in doc if token.pos_ in allowed_postags and len(token.lemma_) >= 2])
        return texts_out

    # bigram = gensim.models.Phrases(documents['data'], min_count=5, threshold=100) # higher threshold fewer phrases.
    # trigram = gensim.models.Phrases(bigram[documents['data']], threshold=100)

    # Faster way to get a sentence clubbed as a trigram/bigram
    # bigram_mod = gensim.models.phrases.Phraser(bigram)
    # trigram_mod = gensim.models.phrases.Phraser(trigram)

    #remove stop words

    data_words_nostops = remove_stopwords(documents, stops=stops)
    # print(data_words_nostops)

    bigram_mod, data_words_bigrams = make_bigrams(texts=data_words_nostops)
    # data_words_trigrams = make_trigrams(bigram_mod, texts=data_words_bigrams)

    # Initialize spacy 'en' model, keeping only tagger component (for efficiency)

    # Do lemmatization keeping only noun, adj, vb, adv
    data_lemmatized = lemmatization(data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])
    #for frequency
    clean_words = []
    for d in data_words_nostops:
        clean_words.extend(d)

    print("data_lemmatized looks like ", data_lemmatized[:5])
    return data_lemmatized