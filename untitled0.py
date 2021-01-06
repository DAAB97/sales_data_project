# -*- coding: utf-8 -*-
"""
Created on Wed Dec 30 15:51:22 2020

@author: abdo
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score


documents = ["This little kitty came to play when I was eating at a restaurant.", 
             "Merley has the best squooshy kitten belly.",
             "Google Translate app is incredible.", 
             "If uou open 100 tab in google you get a smiley face.", 
             "Best cat photo I've ever taken.", 
             "Climbing ninja cat.", 
             "Impressed with google map feedback.", 
             "Key promoter extention for Google Chrome."]


vectorizer = TfidfVectorizer()
stop_words = 'english'

X = vectorizer.fit_transform(documents)

print(vectorizer.vocabulary_)

print(vectorizer.get_feature_names())

true_k = 2
model = KMeans(n_clusters=true_k, init='k-means++', max_iter=300, n_init=1)
model.fit(X)

print(model.labels_)



print("Prediction")
Y = vectorizer.transform(["chrome browser to open."])
prediction = model.predict(Y)
print(prediction)

Y = vectorizer.transform(["My cat is hungry."])
prediction = model.spredict(Y)
print(prediction)

print("Top terms per cluster:")
order_centroids = model.cluster_centers_.argsort()[:, ::-1]
terms = vectorizer.get_feature_names()
for i in range(true_k):
    print("Cluster %d: " %i)
    for ind in order_centroids[i, :10]:
        print(' %s' % terms[ind])















