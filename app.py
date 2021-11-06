# Flask utils
from flask import Flask, request, render_template
import pandas as pd
import numpy as np
import string
import re
#import stweet as st
import pickle5 as pickle
from camel_tools.utils.dediac import dediac_ar

# Visualization imports
import seaborn as sns
import matplotlib.pyplot as plt


# NLP imports
import nltk 
from nltk import word_tokenize
from nltk.stem.isri import ISRIStemmer
from keras.models import load_model
###############################################


app = Flask(__name__) #Initialize the flask App

model = models.load_model('bert_mini2.sav')
# loading
with open('tokenizer.pickle', 'rb') as handle:
    word_tokenizer = pickle.load(handle)


######################


def model_predict(tweet): 
    stop_words_1 = set(nltk.corpus.stopwords.words("arabic"))
    stop_words_2 =['من','في','على','و','فى','يا','عن','مع', 'ان', 'هو','علي','ما','اللي','كل','بعد'
                 ,'ده','اليوم','ان','يوم','انا','الى','كان','ايه','اللى','الى','دي','بين','انت'
                 ,'انا','حتى','لما','فيه','هذا','واحد','احنا','اي','كده','إن','او','أو',
                 'عليه','ف','دى','مين','الي','كانت','امام','زي','يكون','خلال','ع','كنت',
                 'هي','فيها' 'عند','التي','الذي','قال','هذه','قد','انه','ريتويت','بعض','اول','ايه','الان','اي',
                 'منذ','عليها','له','ال','تم','ب','دة','عليك','اى','كلها','اللتى','هى','دا','انك','وهو','ومن',
                 'منك','نحن','زى','انت','انهم','معانا','حتي','وانا','عنه','الي','ونحن','وانت','منكم','وان',
                 'معاهم','معايا','وأنا','عنها','انه','اني','معك','اننا','فيهم','د', 'انتا','عنك','وهى',
                 'معا','ان','انتي','وانت','وان','ومع','وعن','معاكم','معاكو','معاها','وعليه','وانتم','وانتي','¿','|']
    stop_words = set(stop_words_1).union(set(stop_words_2))
    
    # keep arabic charachters and arabic digits and remove the rest
    tweet = re.sub(r'[^\u0600-\u06ff\u0750-\u077f\ufb50-\ufbc1\ufbd3-\ufd3f\ufd50-\ufd8f\ufd50-\ufd8f\ufe70-\ufefc\uFDF0-\uFDFD]+', ' ', str(tweet))
    
    # replace (#this_word) with (this word)
    tweet = re.sub(r'#([^\s]+)', r'\1', tweet)
    tweet = tweet.replace("_", " ")
    
    # remove repeated letters
    tweet = re.sub(r'(.)\1+', r'\1', tweet)
    
    # remove english and arabic punctuations
    arabic_punctuations = '''`÷×؛<>_()*&^%][ـ،/:"؟.,'{}~¦+|!”…“–ـ'''
    english_punctuations = string.punctuation
    punctuations_list = arabic_punctuations + english_punctuations
    translator = str.maketrans('', '', punctuations_list)
    tweet = tweet.translate(translator)
    
    # normalize arabic letters
    tweet = re.sub("[إأآا]", "ا", tweet)
    tweet = re.sub("ى", "ي", tweet)
    tweet = re.sub("ؤ", "ء", tweet)
    tweet = re.sub("ئ", "ء", tweet)
    tweet = re.sub("ة", "ه", tweet)
    tweet = re.sub("گ", "ك", tweet)
    
    # remove all Arabic stop words and white spaces
    original_words = []
    words = word_tokenize(tweet)
    for word in words:
        if word not in stop_words:
            original_words.append(word)
    tweet = " ".join(original_words)
    
    #remove Arabic numbers
    tweet = re.sub("[\u0660-\u0669]+", " ", tweet)
    
    #remove dediac
    tweet = dediac_ar(tweet)
    
    # Stemming for tweets
    ar_st = ISRIStemmer()
    stemmmed_tokens = []
    tokens = word_tokenize(tweet)
    specific_words = ['توكلنا', 'ابشر', 'اعتمرنا', 'صحتي', 'ناجز', 'حافز', 'جداره' ,'تباعد', 'ساهر', 'نور', 'ريف', 'تمهير']
    for token in tokens:
        if token not in specific_words:
            stemmmed_tokens.append(ar_st.stem(token))
        else:
            stemmmed_tokens.append(token)
    tweet = " ".join(stemmmed_tokens)
    
    # part2 - predict 

    preds = model.predict(tweet)
    
    return preds
    
   

###########

@app.route('/', methods=['Get'])
def index():
    #home page
    return render_template('home.html')

###########

@app.route('/predict',methods=['POST','GET'])
def predict():
    if request.method == 'POST':
        tweet = request.form['tweet']
        data = [tweet]
        my_pred = model_predict(data)
        return render_template('result.html',prediction = my_pred)
############

@app.route('/home', methods=['Get'])
def home():
    #home page
    return render_template('home.html')

if __name__ == "__main__":
    app.run(debug=True)




