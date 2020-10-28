#!/usr/bin/env python
# coding: utf-8

# In[156]:


#importing required libraries
import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import nltk
from collections import Counter
from nltk.util import ngrams
import re 
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
#get_ipython().run_line_magic('matplotlib', 'inline')


# In[157]:


#loading the file
f = open("wiki_01txt.txt", "r", encoding="utf8")


# In[158]:


#extracting the text
soup = BeautifulSoup(f, 'html.parser')
wiki_text = soup.get_text()


# In[159]:


#setting to lower case and removing punctuations
wiki_text = wiki_text.lower()
wiki_text = re.sub(r'[^\w\s]', '', wiki_text)


# In[160]:


#tokenizing
unigram = nltk.word_tokenize(wiki_text)
print(unigram)
unigram_count = Counter(unigram) 
#print(unigram_count)
print("Number of unique unigrams:")
print(len(unigram_count))


# In[161]:


#plotting frequency vs rank
uni_sorted_array=unigram_count.most_common()
y_axis=[y for (x, y) in uni_sorted_array]
print(len(y_axis))
x_axis=[i+1 for i in range(len(y_axis))]
plt.plot(x_axis, y_axis)
plt.title('Unigram Frequency vs Rank')
plt.yscale('log')
plt.xscale('log')
plt.xlabel('rank (log10)')
plt.ylabel('cf (log10)')
plt.savefig('Unigram Frequency vs Rank')
plt.show()


# In[162]:


#uni-grams required to cover 90% of the complete corpus
sum=0
count=0
num=len(list(unigram))
for i in uni_sorted_array:
    sum+=i[1]
    if sum>0.9*(num):
        break
    count+=1
print("Uni-grams required to cover 90% of the complete corpus:")
print(count)


# In[163]:


#applying for bigram
bigram = ngrams(unigram,2)
#print(bigram)
bigram_count = Counter(bigram) 
#print(bigram_count)
print("Number of unique bigrams:")
print(len(bigram_count))


# In[164]:


#plotting frequency vs rank
bi_sorted_array=bigram_count.most_common()
y_axis=[y for (x, y) in bi_sorted_array]
print(len(y_axis))
x_axis=[i+1 for i in range(len(y_axis))]
plt.plot(x_axis, y_axis)
plt.title('Bigram Frequency vs Rank')
plt.yscale('log')
plt.xscale('log')
plt.xlabel('rank (log10)')
plt.ylabel('cf (log10)')
plt.savefig('Bigram Frequency vs Rank')
plt.show()


# In[165]:


#bi-grams required to cover 80% of the complete corpus
sum=0
count=0
num=len(list(ngrams(unigram,2)))
for i in bi_sorted_array:
    sum+=i[1]
    if sum>0.8*(num):
        break
    count+=1
print("Bi-grams required to cover 80% of the complete corpus:")
print(count)


# In[166]:


#applying for trigram
trigram = ngrams(unigram,3)
#print(trigram)
trigram_count = Counter(trigram) 
#print(trigram_count)
print("Number of unique trigrams:")
print(len(trigram_count))


# In[167]:


#plotting frequency vs rank
tri_sorted_array=trigram_count.most_common()
y_axis=[y for (x, y) in tri_sorted_array]
print(len(y_axis))
x_axis=[i+1 for i in range(len(y_axis))]
plt.plot(x_axis, y_axis)
plt.title('Trigram Frequency vs Rank')
plt.yscale('log')
plt.xscale('log')
plt.xlabel('rank (log10)')
plt.ylabel('cf (log10)')
plt.savefig('Trigram Frequency vs Rank')
plt.show()


# In[168]:


#tri-grams required to cover 70% of the complete corpus
sum=0
count=0
num=len(list(ngrams(unigram,3)))
for i in tri_sorted_array:
    sum+=i[1]
    if sum>0.7*(num):
        break
    count+=1
print("Tri-grams required to cover 70% of the complete corpus:")
print(count)


# In[169]:


#stemming process
snowBallStemmer = SnowballStemmer("english")
stemWords_unigram = [snowBallStemmer.stem(word) for word in unigram]
#print(' '.join(stemWords))


# In[170]:


#stemming+unigram
stemWords_unigram_count = Counter(stemWords_unigram) 
#print(stemWords_unigram_count)
print("Number of unique unigrams (after stemming):")
print(len(stemWords_unigram_count))


# In[171]:


#plotting frequency vs rank
stem_uni_sorted_array=stemWords_unigram_count.most_common()
y_axis=[y for (x, y) in stem_uni_sorted_array]
print(len(y_axis))
x_axis=[i+1 for i in range(len(y_axis))]
plt.plot(x_axis, y_axis)
plt.title('Unigram (after stemming) Frequency vs Rank')
plt.yscale('log')
plt.xscale('log')
plt.xlabel('rank (log10)')
plt.ylabel('cf (log10)')
plt.savefig('Unigram (after stemming) Frequency vs Rank')
plt.show()


# In[172]:


#uni-grams required to cover 90% of the complete corpus after stemming
sum=0
count=0
#stem_uni_sorted_array=bigram_count.most_common()
num=len(list(stemWords_unigram))
for i in stem_uni_sorted_array:
    sum+=i[1]
    if sum>0.9*(num):
        break
    count+=1
print("Uni-grams required to cover 90% of the complete corpus after stemming:")
print(count)


# In[173]:


#stemming+bigram
stemWords_bigram = ngrams(stemWords_unigram,2)
#print(sremWords_bigram)
stemWords_bigram_count = Counter(stemWords_bigram) 
#print(stemWords_bigram_count)
print("Number of unique bigrams (after stemming):")
print(len(stemWords_bigram_count))


# In[174]:


#plotting frequency vs rank
stem_bi_sorted_array=stemWords_bigram_count.most_common()
y_axis=[y for (x, y) in stem_bi_sorted_array]
print(len(y_axis))
x_axis=[i+1 for i in range(len(y_axis))]
plt.plot(x_axis, y_axis)
plt.title('Bigram (after stemming) Frequency vs Rank')
plt.yscale('log')
plt.xscale('log')
plt.xlabel('rank (log10)')
plt.ylabel('cf (log10)')
plt.savefig('Bigram (after stemming) Frequency vs Rank')
plt.show()


# In[175]:


#bi-grams required to cover 80% of the complete corpus after stemming
sum=0
count=0
num=len(list(ngrams(stemWords_unigram,2)))
for i in stem_bi_sorted_array:
    sum+=i[1]
    if sum>0.8*(num):
        break
    count+=1
print("Bi-grams required to cover 80% of the complete corpus after stemming:")
print(count)


# In[176]:


#stemming+trigram
stemWords_trigram = ngrams(stemWords_unigram,3)
#print(stemWords_trigram)
stemWords_trigram_count = Counter(stemWords_trigram) 
#print(stemWords_trigram_count)
print("Number of unique trigrams (after stemming):")
print(len(stemWords_trigram_count))


# In[177]:


#plotting frequency vs rank
stem_tri_sorted_array=stemWords_trigram_count.most_common()
y_axis=[y for (x, y) in stem_tri_sorted_array]
print(len(y_axis))
x_axis=[i+1 for i in range(len(y_axis))]
plt.plot(x_axis, y_axis)
plt.title('Trigram (after stemming) Frequency vs Rank')
plt.yscale('log')
plt.xscale('log')
plt.xlabel('rank (log10)')
plt.ylabel('cf (log10)')
plt.savefig('Trigram (after stemming) Frequency vs Rank')
plt.show()


# In[178]:


#tri-grams required to cover the 70% of the complete corpus after stemming
sum=0
count=0
num=len(list(ngrams(stemWords_unigram,3)))
for i in stem_tri_sorted_array:
    sum+=i[1]
    if sum>0.7*(num):
        break
    count+=1
print("Tri-grams required to cover the 70% of the complete corpus after stemming:")
print(count)


# In[179]:


#lemmatizing the words
lemmatizer = WordNetLemmatizer()
lemWords_unigram_verb=[lemmatizer.lemmatize(word,pos="v") for word in unigram]
lemWords_unigram_adjective=[lemmatizer.lemmatize(word,pos="a") for word in lemWords_unigram_verb]
lemWords_unigram=[lemmatizer.lemmatize(word,pos="n") for word in lemWords_unigram_adjective]
#print(lemWords_unigram)


# In[180]:


lemWords_unigram_count = Counter(lemWords_unigram) 
#print(lemWords_unigram_count)
print("Number of unique unigrams (after lemmatization):")
print(len(lemWords_unigram_count))


# In[181]:


#plotting frequency vs rank
lem_uni_sorted_array=lemWords_unigram_count.most_common()
y_axis=[y for (x, y) in lem_uni_sorted_array]
print(len(y_axis))
x_axis=[i+1 for i in range(len(y_axis))]
plt.plot(x_axis, y_axis)
plt.title('Unigram (after lemmatization) Frequency vs Rank')
plt.yscale('log')
plt.xscale('log')
plt.xlabel('rank (log10)')
plt.ylabel('cf (log10)')
plt.savefig('Unigram (after lemmatization) Frequency vs Rank')
plt.show()


# In[182]:


#uni-grams required to cover 90% of the complete corpus after lemmatization
sum=0
count=0
num=len(list(lemWords_unigram))
for i in lem_uni_sorted_array:
    sum+=i[1]
    if sum>0.9*(num):
        break
    count+=1
print("Uni-grams required to cover the 90% of the complete corpus afer lemmatization:")
print(count)


# In[183]:


#lemmatization+bigram
lemWords_bigram = ngrams(lemWords_unigram,2)
#print(lemWords_bigram)
lemWords_bigram_count = Counter(lemWords_bigram) 
#print(lemWords_bigram_count)
print("Number of unique bigrams (after lemmatization):")
print(len(lemWords_bigram_count))


# In[184]:


#plotting frequency vs rank
lem_bi_sorted_array=lemWords_bigram_count.most_common()
y_axis=[y for (x, y) in lem_bi_sorted_array]
print(len(y_axis))
x_axis=[i+1 for i in range(len(y_axis))]
plt.plot(x_axis, y_axis)
plt.title('Bigram (after lemmatization) Frequency vs Rank')
plt.yscale('log')
plt.xscale('log')
plt.xlabel('rank (log10)')
plt.ylabel('cf (log10)')
plt.savefig('Bigram (after lemmatization) Frequency vs Rank')
plt.show()


# In[185]:


#bi-grams required to cover 80% of the complete corpus after lemmatization
sum=0
count=0
#stem_uni_sorted_array=bigram_count.most_common()
num=len(list(ngrams(lemWords_unigram,2)))
for i in lem_bi_sorted_array:
    sum+=i[1]
    if sum>0.8*(num):
        break
    count+=1
print("Bi-grams required to cover 80% of the complete corpus after lemmatization:")    
print(count)


# In[186]:


#lemmatization+trigram
lemWords_trigram = ngrams(lemWords_unigram,3)
#print(lemWords_trigram)
lemWords_trigram_count = Counter(lemWords_trigram) 
#print(lemWords_trigram_count)
print("Number of unique trigrams (after lemmatization):")
print(len(lemWords_trigram_count))


# In[187]:


#plotting frequency vs rank
lem_tri_sorted_array=lemWords_trigram_count.most_common()
y_axis=[y for (x, y) in lem_tri_sorted_array]
print(len(y_axis))
x_axis=[i+1 for i in range(len(y_axis))]
plt.plot(x_axis, y_axis)
plt.title('Trigram (after lemmatization) Frequency vs Rank')
plt.yscale('log')
plt.xscale('log')
plt.xlabel('rank (log10)')
plt.ylabel('cf (log10)')
plt.savefig('Trigram (after lemmatization) Frequency vs Rank')
plt.show()


# In[188]:


#tri-grams required to cover 70% of the complete corpus after lemmatization
sum=0
count=0
num=len(list(ngrams(lemWords_unigram,3)))
for i in lem_tri_sorted_array:
    sum+=i[1]
    if sum>0.7*(num):
        break
    count+=1
print("Tri-grams required to cover 70% of the complete corpus after lemmatization:")
print(count)


# In[189]:


# top 20 bi-gram collocations in the text corpus using Chi-square test
num = len(list(ngrams(unigram, 2)))
#print(num)
arr=[];
for i in bigram_count:
    o11=bigram_count[i]
    o12=unigram_count[i[0]]-bigram_count[i]
    o21=unigram_count[i[1]]-bigram_count[i]
    o22=num-o11-o12-o21
    chi=(num*(o11*o22-o12*o21)*(o11*o22-o12*o21))/((o11+o12)*(o11+o21)*(o12+o22)*(o21+o22))
    if (o12+o21>15): 
        arr.append((i,chi))
def takeSecond(elem):
    return elem[1]
# sort list with key
arr.sort(key=takeSecond, reverse=True)
# print list
print('Sorted list:', arr)


# In[ ]:




