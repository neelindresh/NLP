from summa import summarizer
from summa import keywords
import re
from nltk.corpus import stopwords
from nltk.cluster.util import cosine_distance
import pandas as pd
import numpy as np
import networkx as nx

stopwords_list=stopwords.words('english')


stopwords_list=stopwords.words('english')
def sort_dict(dict_with_val):
    return sorted(dict_with_val.items(), key=lambda x: x[1], reverse=True)


#----------FREQ SUMMARIZER START-----------#
def count_vectorizer(text_arr,weighted=True):
    freq={}
    max_freq=0
    for sent in text_arr:
        if len(sent.split())<=1:
            continue
        for words in sent.split():
            if words in stopwords_list:
                continue
            else:
                if words in freq:

                    freq[words]+=1
                else:
                    freq[words]=1
                if freq[words]>max_freq:
                    max_freq=freq[words]
    if weighted:
        freq={i:freq[i]/max_freq for i in freq}
    return freq
def _freq_score(freq,text_arr):
    score_dict={}
    for sent in text_arr:
        score=0
        words=0
        for word in sent.split():
            if word in freq:
                score+=freq[word]
                words+=1
        if words>0:
            score_dict[sent]=score/words
    return score_dict
def text_processing(text,separator=False):
    replace_char=['”',"`"]
    
    for i in replace_char:
        text=text.replace(i,"")
    text=re.sub("[^A-Za-z ]","<br>",text)
    if separator:
        return text
    text_arr=text.split("<br>")
    
    text_arr=[i.strip().lower() for i in text_arr if len(i.strip())>3]
    return text_arr
def freq_summazier(text,top=5):
    text_arr=text_processing(text)
    freq=count_vectorizer(text_arr)
    score_dict=_freq_score(freq,text_arr)
    score_dict=sort_dict(score_dict)
    #print(score_dict[:top])
    return [i[0] for i in score_dict[:top]]
#----------FREQ SUMMARIZER END-----------#



#----------TextRank START(Small paragraphs not documents)-----------#
def _vectorize(sent,max_len):
    all_words=sent.split()
    vector=[0] * max_len
    for w in all_words:
        if w in stopwords_list:
            continue
        vector[all_words.index(w)] += 1
    return vector
def _similarity_matrix(vector_space):
    matrix=[]
    for i in vector_space:
        vector_similarity=[]
        for j in vector_space:
            #print(i,j,cosine_distance(i,j))
            vector_similarity.append(1-cosine_distance(i,j))
        matrix.append(vector_similarity)
    return matrix
def page_rank(similarity_matrix):
    nx_graph = nx.from_numpy_array(np.array(similarity_matrix))
    scores = nx.pagerank(nx_graph)
    return scores
def textRank(sent):
    text_arr=text_processing(sent)
    vector_space=[]
    max_len=max([len(sent.split()) for sent in text_arr ])
    for sent in text_arr:
        vector_space.append(_vectorize(sent,max_len=max_len))
    similarity_matrix=_similarity_matrix(vector_space)
    scores=page_rank(similarity_matrix)
    sent_score={i:scores[idx] for idx,i in enumerate(text_arr)}
    score_dict=sort_dict(sent_score)

    return score_dict
#----------TextRank END-----------#
#----------RAKE START (Small paragraphs not documents)-----------#

def degree(text_arr):
    deg={}
    for sent in text_arr:
        word_array=[word for word in sent.split() if word not in stopwords_list]
        if len(word_array)<=1:
            continue
        for words in word_array:
            if words in deg:
                deg[words]+=len(word_array)
            else:
                deg[words]=len(word_array)
    return deg     
def rake_score(deg,freq):
    score={}
    for word,degree in deg.items():
        score[word]=degree/freq[word]
    return score
def rake_sent_score(text_arr,score_dict,norm=True):
    text_score={}
    for sent in text_arr:
        sent_score=0
        words=sent.split()
        for word in words:
            if word in score_dict:
                sent_score+=score_dict[word]
        if norm:
            text_score[sent]=sent_score/len(words)
        else:
            text_score[sent]=sent_score
    return text_score
def RAKE_SENT(text):
    text_arr=text_processing(text)
    freq=count_vectorizer(text_arr,weighted=False)
    deg=degree(text_arr)
    score=rake_score(deg,freq)
    score_dict=rake_sent_score(text_arr,score)
    score_dict=sort_dict(score_dict)
    
    return score_dict
def RAKE(text):
    text_arr=text_processing(text)
    freq=count_vectorizer(text_arr,weighted=False)
    deg=degree(text_arr)
    score=rake_score(deg,freq)
    score=sort_dict(score)
    return score
#----------RAKE END-----------#



def replace_escape(text):
    return text.replace("\n"," ")


def get_keywords(text,ratio=0.5):
    
    text=replace_escape(text)
    kw=keywords.keywords(text,ratio=ratio)
    
    return kw.split("\n")
