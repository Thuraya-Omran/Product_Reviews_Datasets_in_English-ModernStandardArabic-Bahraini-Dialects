# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 14:50:39 2020

@author: Thuraya
"""




import csv
import pandas as pd
import re
import string
import sys
import argparse
import nltk
from stop_words import get_stop_words
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
import gensim

from gensim.models import Word2Vec
from sklearn.decomposition import PCA
from matplotlib import pyplot
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout
import tensorflow as tf
from gensim.models import Word2Vec

import encodings
from gensim.parsing.preprocessing import remove_stopwords
from gensim.parsing.preprocessing import strip_multiple_whitespaces
from gensim.parsing.preprocessing import strip_non_alphanum
from gensim.parsing.preprocessing import strip_numeric
from gensim.parsing.preprocessing import strip_punctuation
from gensim.parsing.preprocessing import strip_short #to remove words lesser than certain number of characters 
from gensim.parsing.preprocessing import strip_tags
from gensim.parsing.preprocessing import stem_text
import re
#import pyarabic
#from pyarabic import stopwords
import csv
import pandas as pd
import re
import string
import sys
import argparse
import nltk
from stop_words import get_stop_words
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
import gensim
#from gensim.parsing.preprocessing import remove_stopwords
from gensim.models import Word2Vec
from sklearn.decomposition import PCA
from matplotlib import pyplot
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout
import tensorflow as tf
from gensim.models import Word2Vec


#to add more stop words to the builtin one  

from gensim.parsing.preprocessing import STOPWORDS
english_stop_words = STOPWORDS.union(set(['i', 'this','it','the']))
#print(my_stop_words)



################   Defining list for each data set ###################

originalrev=[]
origianlbl=[]

newrevlist=[]
newlbllist=[]

newoddrevlist=[]
newoddlbllist=[]
#-----------------------------
OriginlReviewList=[]
newTextRawList=[]
newTextOddRawList=[]
OriginalandAugemted=[]
#------------------------------
Ddocs=[]
labels=[]

df=pd.read_csv('EC-5000-Bal.txt',sep='\t',names=['polarity','Dcomment'],encoding='utf-8')
#print(df) # to print the dataframe 


X3=df["Dcomment"]
y=df["polarity"]

counter1=0
counter2=0
counter3=0
import nlpaug.augmenter.word as naw
from nlpaug.util import Action


aug=naw.RandomWordAug(action=Action.SWAP,aug_min=1,aug_max=10)
#aug2=naw.RandomWordAug(action=Action.CROP,aug_min=1,aug_max=3)
aug2=naw.RandomWordAug(action=Action.SWAP,aug_min=1,aug_max=5)


for index, row in df.iterrows():
     text3=row["Dcomment"]
    #augmented_text3=aug.augment(text3)
    #text3=augmented_text3
     orglbl=row["polarity"]
     OriginlReviewList.append(text3)
     if index %2==0:# using mode to augment even reviews

        #print (' The {} th review : '.format(index))
        #print( newtext)
        newtext=aug.augment(text3)
        newtext=aug2.augment(newtext)
        newTextRawList.append(newtext)
        #converting text to lowercase 
        newtext= newtext.lower()
    #removing  non-alphabetic characters (hash , @)
        newtext=strip_non_alphanum(newtext)
    #removing digits  
        newtext=strip_numeric(newtext)
     #replace punctuation with space
        newtext=strip_punctuation(newtext)
    # removing tags
        newtext=strip_tags(newtext)
        #remove words lesser than certain number of characters such as i  
        #text=strip_short(text, minsize=3)
        #removing white spaces (tab space)
        newtext=strip_multiple_whitespaces(newtext)
        newtext = word_tokenize(newtext)
        newtext = [w for w in  newtext if not w in english_stop_words]
        newrevlist.append(newtext)
        nlbl=row["polarity"]
        newlbllist.append(nlbl)
        counter2+=1
   #************Augmenting and preprocessing of odd reviews
        
     if index %2==1:# using mode to augment odd reviews
        #print (' The {} th review : '.format(index))
        #print(text3)
        newtextodd=aug.augment(text3)
        newtextodd=aug2.augment(newtextodd)
        newTextOddRawList.append(newtextodd)
        newtextodd= newtextodd.lower()
    #removing  non-alphabetic characters (hash , @)
        newtextodd=strip_non_alphanum(newtextodd)
    #removing digits  
        newtextodd=strip_numeric(newtextodd)
     #replace punctuation with space
        newtextodd=strip_punctuation(newtextodd)
    # removing tags
        newtextodd=strip_tags(newtextodd)
        #remove words lesser than certain number of characters such as i  
        #text=strip_short(text, minsize=3)
        #removing white spaces (tab space)
        newtextodd=strip_multiple_whitespaces(newtextodd)
        newtextodd = word_tokenize(newtextodd)
        newtextodd = [w for w in  newtextodd if not w in english_stop_words]
        newoddrevlist.append(newtextodd)            
        noddlbl=row["polarity"]
        counter3+=1
        newoddlbllist.append(noddlbl)
    
    
     text3= text3.lower()
    #removing  non-alphabetic characters (hash , @)
     text3=strip_non_alphanum(text3)
    #removing digits  
     text3=strip_numeric(text3)
     #replace punctuation with space
     text3=strip_punctuation(text3)
    # removing tags
     text3=strip_tags(text3)
        #remove words lesser than certain number of characters such as i  
        #text=strip_short(text, minsize=3)
        #removing white spaces (tab space)
     text3=strip_multiple_whitespaces(text3)
     text3 = word_tokenize(text3)        
     text3 = [w for w in text3 if not w in english_stop_words] # completing code of removing english  stop word 
     counter1+=1
        
     originalrev.append(text3)
     origianlbl.append(orglbl)
        
        
counter=counter1+counter2+counter3        
print('\n')
Ddocs=originalrev+ newrevlist+ newoddrevlist
print('The The total number of reviews  after augmentation is {} : '.format(counter))
#print(Ddocs)
print('\n\n')
labels=origianlbl+newlbllist+newoddlbllist
print('The new list of labels after augmentation is {} : '.format(counter))
#print(labels)
            

#-----------------------------------------------------------------------
# to write the original and augmented raw reviews to a text file 
OriginalandAugemted=OriginlReviewList+newTextRawList+newTextOddRawList

import pandas as pd
import numpy as np

l1=labels
l2=OriginalandAugemted
mydataframe = pd.DataFrame(list(zip(l1, l2)))


#print('mydataframe ') 
#print(mydataframe)
mydataframe.to_csv('Augment Eng 5000 rev.txt', sep='\t',header=None)

print('********Creating augmented file has been done *******')


# to count the numberof positive andnegative reviews in the original file and the augmented file 

import numpy as np
neg, pos = np.bincount(df['polarity'])
total = neg + pos
print('Numbber of original Reviews :\n    Total: {}\n    Positive: {} ({:.2f}% of total)\n    Negative: {} ({:.2f}% of total)\n '.format(
    total, pos, 100 * pos / total,neg,100 * neg/total))





df=pd.read_csv('Augment Eng 5000 rev.txt',sep='\t',names=['polarity','Dcomment'],encoding='utf-8')
neg, pos = np.bincount(df['polarity'])
total = neg + pos
print('Number of Reviews after augmentation:\n    Total: {}\n    Positive: {} ({:.2f}% of total)\n    Negative: {} ({:.2f}% of total)\n '.format(
    total, pos, 100 * pos / total,neg,100 * neg/total))
#----------------------------------------------------------------------------------
