# -*- coding: utf-8 -*-
"""
Created on Fri Dec 25 13:49:26 2020

@author: Thuraya Omran
"""



import csv
import pandas as pd
import re
import string
import argparse

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


stop_words = set(stopwords.words("Arabic"))
#add words that aren't in the NLTK stopwords list
new_stopwords = ['مثل','كان','اني','اكون'
                 ,'الأخرى','لقد ','الى','عندما','و','ان'
                 ,'انها','احد','احدى','الاخرى']
new_stopwords_list = stop_words.union(new_stopwords)
#print(new_stopwords_list)


# to create dialects stopwords
def getStopWordList(stopWordListFileName):
    #read the myStopWords file and build a list
    myStopWords = []
    #myStopWords.append('AT_USER')
    #myStopWords.append('URL')

    fp = open('MSA-Stopwords.txt', 'r',encoding='utf-8')
    line = fp.readline()
    while line:
        word = line.strip()
        myStopWords.append(word)
        line = fp.readline()
    fp.close()
    return myStopWords
dialect_stopwords = getStopWordList('MSA-Stopwords.txt')

dialect_stopwords_list =stop_words.union(dialect_stopwords)


#to add words to gensim stopwords 

from gensim.parsing.preprocessing import STOPWORDS
my_stop_words = STOPWORDS.union(set(['i', 'this','it','the']))


arabic_punctuations = '''`÷×؛<>_()*&^%][ـ،/:"؟.,'{}~¦+|!”…“–ـ'''
english_punctuations = string.punctuation
punctuations_list = arabic_punctuations + english_punctuations

arabic_diacritics = re.compile("""
                             ّ    | # Tashdid
                             َ    | # Fatha
                             ً    | # Tanwin Fath
                             ُ    | # Damma
                             ٌ    | # Tanwin Damm
                             ِ    | # Kasra
                             ٍ    | # Tanwin Kasr
                             ْ    | # Sukun
                             ـ     # Tatwil/Kashida

                         """, re.VERBOSE)



parser = argparse.ArgumentParser(description='Pre-process arabic text (remove '
                                             'diacritics, punctuations, and repeating '
                                             'characters).')

def normalize_arabic(text):
    text = re.sub("[إأآا]", "ا", text)
    #text = re.sub("ي", "ى", text) # the letters are switched from that in the original file 
    #text = re.sub("ى", "ي", text) # this preprocessing is cancelled , for the purpose of stop word على
    text = re.sub("ؤ", "ء", text)
    text = re.sub("ئ", "ء", text)
    text = re.sub("ة", "ه", text)
    text = re.sub("گ", "ك", text)
    return text


def remove_diacritics(text):
    text = re.sub(arabic_diacritics, '', text)
    return text


def remove_punctuations(text):
    translator = str.maketrans('', '', punctuations_list)
    return text.translate(translator)



def remove_repeating_char(text):
    #return re.sub(r'(.)\1+', r'\1', text)
    # to keep two character of repeated characters
    return re.sub(r'(.)\1+', r'\1\1', text)





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
#-------------------------------
df=pd.read_csv('MSA-5000-Bal.txt',sep='\t',names=['polarity','Dcomment'],encoding='utf-8')


#print(df) # to print the dataframe 


X3=df["Dcomment"]
y=df["polarity"]
counter1=0
counter2=0
counter3=0
import nlpaug.augmenter.word as naw
from nlpaug.util import Action

aug=naw.RandomWordAug(action=Action.SWAP,aug_min=1,aug_max=10)

aug2=naw.RandomWordAug(action=Action.SWAP,aug_min=1,aug_max=5)
for index, row in df.iterrows():
    text3=row["Dcomment"]
    OriginlReviewList.append(text3)
    #augmented_text3=aug.augment(text3)
    #text3=augmented_text3
    orglbl=row["polarity"]
   
    if index %2==0:# using mode to augment even reviews
        newtext=aug.augment(text3)
        newtext=aug2.augment(newtext)
        newTextRawList.append(newtext)
        
        if __name__ == '__main__':
            args = parser.parse_args()
            ###text = args.infile.read()
            newtext = remove_punctuations(newtext)
            newtext = remove_diacritics(newtext)
            newtext = remove_repeating_char(newtext)
            newtext = normalize_arabic(newtext)
            # to remove english words 
            newtext=re.sub(r'\s*[A-Za-z]+\b', '' , newtext)
            newtext=newtext.rstrip()
            
             # to remove numbers 
            #text3=re.sub(r'[0-9]+', '',text3)
            #another way replace [0-9] by \d
            newtext= re.sub(r"\d+", "",newtext)
            
            
            newtext= word_tokenize(newtext)
           # ^^^^^^^to remove  MSA  stopword^^^^^^
            newtext = [w for w in newtext if not w in dialect_stopwords_list]
        
            newrevlist.append(newtext)
            nlbl=row["polarity"]
            counter2+=1
            newlbllist.append(nlbl)
            
         #************Augmenting and preprocessing of odd reviews
    if index %2==1:# using mode to augment odd reviews
        newtextodd=aug.augment(text3)
        newtextodd=aug2.augment(newtextodd)
        newTextOddRawList.append(newtextodd)
        if __name__ == '__main__':
            args = parser.parse_args()
            ###text = args.infile.read()
            newtextodd = remove_punctuations(newtextodd)
            newtextodd = remove_diacritics(newtextodd)
            newtextodd = remove_repeating_char(newtextodd)
            newtextodd = normalize_arabic(newtextodd)
            # to remove english words 
            newtextodd=re.sub(r'\s*[A-Za-z]+\b', '' , newtextodd)
            newtextodd=newtextodd.rstrip()
            
             # to remove numbers 
            #text3=re.sub(r'[0-9]+', '',text3)
            #another way replace [0-9] by \d
            newtextodd= re.sub(r"\d+", "",newtextodd)
            
            
            newtextodd= word_tokenize(newtextodd)
           # ^^^^^^^to remove  MSA stopword^^^^^^
            newtextodd = [w for w in newtextodd if not w in dialect_stopwords_list]
        
            newoddrevlist.append(newtextodd)
            noddlbl=row["polarity"]
            counter3+=1
            newoddlbllist.append(noddlbl)
                
    ######################*************** preprocessing of original MSA  reviews *************####################### 
    if __name__ == '__main__':
        args = parser.parse_args()
        ###text = args.infile.read()
        text3 = remove_punctuations(text3)
        text3 = remove_diacritics(text3)
        text3 = remove_repeating_char(text3)
        text3 = normalize_arabic(text3)
        # to remove english words 
        text3=re.sub(r'\s*[A-Za-z]+\b', '' , text3)
        text3=text3.rstrip()
        
         # to remove numbers 
        #text3=re.sub(r'[0-9]+', '',text3)
        #another way replace [0-9] by \d
        text3= re.sub(r"\d+", "",text3)
        
        
        text3= word_tokenize(text3)
       # ^^^^^^^to remove  dialects stopword^^^^^^
        text3 = [w for w in text3 if not w in dialect_stopwords_list] # completing code of removing arabic stop word 
        counter1+=1
        
        originalrev.append(text3)
        origianlbl.append(orglbl)
        
        
counter=counter1+counter2+counter3



print('\n\n') 
print('counter 1 (original)value is {}'.format(counter1))
print('counter 2 (even index)value is {}'.format(counter2))
print('counter 3 (odd index)value is {}'.format(counter3))           
print('\n')
Ddocs=originalrev+ newrevlist+ newoddrevlist
print('The The total number of reviews  after augmentation is: {}  '.format(counter))
#print(Ddocs)
print('\n\n')
labels=origianlbl+newlbllist+newoddlbllist
print('The new list of labels after augmentation is :{}  '.format(counter))
#print(labels)
        
#-----------------------------------------------------------------------
# to write the original and augmented raw reviews to a text file 
OriginalandAugemted=OriginlReviewList+newTextRawList+newTextOddRawList

import pandas as pd
import numpy as np

l1=labels
l2=OriginalandAugemted
mydataframe = pd.DataFrame(list(zip(l1, l2)))


mydataframe.to_csv('Augment 5000 MSA reviews.txt', sep='\t',header=None)

print('********Creating augmented file has been done *******')


# to count the number of positive and negative reviews in the original file and the augmented file 

import numpy as np
neg, pos = np.bincount(df['polarity'])
total = neg + pos
print('Numbber of original Reviews:\n    Total: {}\n    Positive: {} ({:.2f}% of total)\n    Negative: {} ({:.2f}% of total)\n '.format(
    total, pos, 100 * pos / total,neg,100 * neg/total))



df=pd.read_csv('Augment 5000 MSA reviews.txt',sep='\t',names=['polarity','Dcomment'],encoding='utf-8')
neg, pos = np.bincount(df['polarity'])
total = neg + pos
print('Number of Reviews after augmentation:\n    Total: {}\n    Positive: {} ({:.2f}% of total)\n    Negative: {} ({:.2f}% of total)\n '.format(
    total, pos, 100 * pos / total,neg,100 * neg/total))
#----------------------------------------------------------------------------------

