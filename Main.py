import numpy as np
import matplotlib.pyplot as plt
import seaborn
import nltk
import os
import sklearn.feature_extraction.text as text

CORPUS_PATH = os.path.join('Corpus','austen-bronte-split')
filesNames = sorted([os.path.join(CORPUS_PATH,fn) for fn in os.listdir(CORPUS_PATH)])
vectorizer = text.CountVectorizer(input='filename', stop_words='english', min_df=1)
dtm = vectorizer.fit_transform(filesNames).toarray()
vocab = np.array(vectorizer.get_feature_names())


'''
print(vocab)
['000' '10' '10th' ..., 'zornes' 'zschokke' 'zurueck']
'''

import numpy as np
from sklearn import decomposition
num_topics = 20
num_top_words = 20
clf = decomposition.NMF(n_components=num_topics, random_state=1)
doctopic = clf.fit_transform(dtm)
topic_words = []
for topic in clf.components_:
    word_idx = np.argsort(topic)[::-1][0:num_top_words]
    topic_words.append([vocab[i] for i in word_idx])
    '''
    print(word_idx)
    print(vocab)
    [ 6733  6835 12205  8011  5209 18179 19989 13131 17419 20119 19987 19477 2845 18527 12340  3926  9278 16140 12443  2678]
    ['000' '10' '10th' ..., 'zornes' 'zschokke' 'zurueck']
    [17862 11872  5258 20119  3957 17419 18527 13799 16119 19477 19987  5233 9132  9126  1795 11807  9995 11953 11445 22433]
    ['000' '10' '10th' ..., 'zornes' 'zschokke' 'zurueck']
    '''

'''
PREGUNTA 1:
El bucle esta recorriendo los pesos y agregandolos a la matrix, despues de terminar, se hace un calculo de los pesos normalizados.
Estos pesos se calculan por el clasificador basandoce en los topicos mas importantes y la relacion entre los documentos
'''


doctopic = doctopic / np.sum(doctopic, axis=1, keepdims=True)

'''
print(doctopic)
[[ 0.          0.27875534  0.00450424 ...,  0.          0.          0.        ]
 [ 0.1006442   0.27680987  0.         ...,  0.0247592   0.          0.10284378]
 [ 0.08224933  0.22782252  0.         ...,  0.06403714  0.          0.06334557]
 ..., 
 [ 0.07865496  0.          0.         ...,  0.0497371   0.05991098  0.        ]
 [ 0.06626874  0.          0.         ...,  0.11773758  0.04059651
   0.02392662]
 [ 0.02561435  0.          0.         ...,  0.          0.          0.06847266]]
'''

novel_names = []
for fn in filenames:
    basename = os.path.basename(fn)
    name, ext = os.path.splitext(basename)
    name = name.rstrip('0123456789')
    novel_names.append(name)
novel_names = np.asarray(novel_names)
doctopic_orig = doctopic.copy()
num_groups = len(set(novel_names))
doctopic_grouped = np.zeros((num_groups, num_topics))

for i, name in enumerate(sorted(set(novel_names))):
    doctopic_grouped[i, :] = np.mean(doctopic[novel_names == name, :], axis=0)
doctopic = doctopic_grouped

for t in range(len(topic_words)):
    print("Topic {}: {}".format(t, ' '.join(topic_words[t][:15])))

novels = sorted(set(novel_names))
for i in range(len(doctopic)):
    top_topics = np.argsort(doctopic[i,:])[::-1][0:3]
    top_topics_str = ' '.join(str(t) for t in top_topics)
    print("{}: {}".format(novels[i], top_topics_str))





