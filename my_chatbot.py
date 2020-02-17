import io
import random
import string
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

import nltk
from nltk.stem import WordNetLemmatizer
nltk.download('popular' , quiet=True)

with open('chatbot.txt','r',errors='ignore',encoding='utf8') as fin:
    raw_data=fin.read().lower()


sent_tokens=nltk.sent_tokenize(raw_data)
word_tokens=nltk.word_tokenize(raw_data)

lemmer=WordNetLemmatizer()
def lemTokens(tokens):
    return[lemmer.lemmatize(token) for token in tokens]
punc_dict=dict((ord(punct),None) for punct in string.punctuation)

def lemNormalize(text):
    return lemTokens(nltk.word_tokenize(text.lower().translate(punc_dict)))

GREETING_INPUTS = ("hello", "hi", "greetings", "sup", "what's up","hey","yo","wtcha doin?","wassup")
Gali_input=("bhosadike","bsdk","mc","madarchod","mkl","mkc","chutiye","chutiya","teri maa ki","teri maa ka bhosda","sale","kutte","lund","lawde","lavde")
Gali_output=["kripya aisi bhasha ka prayog na karen","sale madarchod","bhosadike","gand mara le","sale lund","muh me le le","maa chuda le","behen k lode","ae lawde chal nikal pehli fursat me nikal"]
GREETING_RESPONSES = ["hi", "hey", "*nods*", "hi there", "hello", "I am glad! You are talking to me","hiiii","hey there!"]

def greeting(sentence):
    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)
        elif word.lower() in Gali_input:
            return random.choice(Gali_output)

def response(user_response):
    robo_response=''
    sent_tokens.append(user_response)
    TfidfVec = TfidfVectorizer(tokenizer=lemNormalize, stop_words='english')
    tfidf = TfidfVec.fit_transform(sent_tokens)
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx=vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    if(req_tfidf==0):
        robo_response=robo_response+"I am sorry! I don't understand you"
        return robo_response
    else:
        robo_response = robo_response+sent_tokens[idx]
        return robo_response


flag=True
print("RO_s_BOT: I am Rohini's Chat Bot. Rohini is busy right now so you may talk to me instead if you wish. If you want to end this conversation, say a Bye!")
while(flag==True):
        user_response = input()
        user_response=user_response.lower()
        if(user_response!='bye'):
            if(user_response=='thanks' or user_response=='thank you' or user_response=='thanx' or user_response=='thnx' or user_response=='ty'):
                flag=False
                print(random.choice["RO_s_BOT: Ah! I'm blushing :)","RO_s_BOT: You are Welcome!:)","RO_s_BOT: Ab mai itni v koi khas nahi ;-)","RO_s_BOT: Bas bas :P"])
            else:
                if(greeting(user_response)!=None):
                    print("RO_s_BOT: "+greeting(user_response))
                else:
                    print("RO_s_BOT: ",end="")
                    print(response(user_response))
                    sent_tokens.remove(user_response)
        else:
            flag=False
            print("RO_s_BOT: Bye! take care :)")