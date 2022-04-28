
import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# from sentence_transformers import SentenceTransformer
# sbert_model = SentenceTransformer('bert-base-nli-mean-tokens')

import nltk
from nltk.corpus import stopwords

nltk.download('punkt')
nltk.download('stopwords')

embed_dict = {}

with open('./content/glove.6B.200d.txt','r') as f:
  for line in f:
    values = line.split()
    word = values[0]
    vector = np.asarray(values[1:],'float32')
    embed_dict[word]=vector

stop_words = stopwords.words('english')
updated_stop_words = stop_words.copy()
updated_stop_words.extend(['br'])
updated_stop_words.remove('not')
updated_stop_words.remove('nor')
stop_words = updated_stop_words

DIM = 128
VECTOR_LENGTH = 200

out_v = open('./data/vectors.tsv', 'r', encoding='utf-8')
out_m = open('./data/metadata.tsv', 'r', encoding='utf-8')

vocab = out_m.read().split('\n')
weights = out_v.read().split('\n')
w = []
for id,i in enumerate(weights):
    try:
      w.append(np.array([float(j) for j in i.split('\t')]))
    except:
      print([j for j in i.split('\t')])

we = {}

for index, word in enumerate(vocab):
  vec = weights[index]
  we[word] = w[index]

def clean_sentence(sentence,string_out=False):
  sent = tf.strings.regex_replace(sentence,
                                  "<br />|[^\w\s]" , '')
  sent = tf.strings.lower(sent)
  
  if string_out==True:
    sentence = tf.constant(sent)
    sentence = clean_sentence(sentence)
    new_sent = tf.strings.split(sentence," ")
    new_sent = [i.numpy().decode("utf-8") for i in new_sent]
    new_sent = [i for i in new_sent if i!="" and i not in stop_words]
    return " ".join(new_sent)

  return sent


def sentence_to_vectors(sent,clean=True):
  if clean==True:
    sentence = tf.constant(sent)
    sentence = clean_sentence(sentence)
    new_sent = tf.strings.split(sentence," ")
    new_sent = [i.numpy().decode("utf-8") for i in new_sent]
    new_sent = [i for i in new_sent if i!="" and i not in stop_words]
  
  else:
    new_sent = [i for i in sent if i!="" and i not in stop_words]

  vectors = []

  for i in new_sent:
    if i in we.keys():
      vectors.append(we[i])
    else:
      vectors.append(we['[UNK]'])

  return vectors

def add_vecs(vectors,shape):
  res = np.zeros(shape)
  for i in vectors:
    res = np.add(res,i)
  return res

def encode(sent):
  l = []
  for i in sent.split(" "):
    try:
      l.append(np.array(embed_dict[i]))
    except:
      l.append(np.array([0]*VECTOR_LENGTH))
  return np.array(l)

data = pd.read_csv("./data/movie_reviews.csv")
movie_vec = {data['name'][i]:encode(data['review'][i]) for i in range(len(data))}

def get_movies(movie_desc,top=10):
  user_vec = encode(clean_sentence(movie_desc,string_out=True))
  #print(value.shape for key,value in movie_vec.items())
  #for key,val in movie_vec.items():
  #  print(val.shape)
  #  break
  scores = {key:cosine_similarity(user_vec,value).tolist() for key,value in movie_vec.items()}
  sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)

  l = []

  for name,score in sorted_scores:
    l.append({"name":name,"link":data[data['name']==name]['link']}) #],"score":score})
  
  return l[:top]
