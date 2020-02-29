from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
df=pd.read_csv("hm_train.csv")
df1=pd.read_csv("hm_test.csv")
y=df.iloc[:,4].values
#print(df['cleaned_hm'])
#cleaning the text
corpus=[]
for i in range(0,60321):
    cleaned_hm=re.sub('[^a-zA-Z]', ' ', df['cleaned_hm'][i])
    cleaned_hm=cleaned_hm.lower()
    cleaned_hm=cleaned_hm.split()
    ps=PorterStemmer()
    cleaned_hm=[ps.stem(word) for word in cleaned_hm if not word in set(stopwords.words('english'))]
    cleaned_hm=' '.join(cleaned_hm)
    corpus.append(cleaned_hm)
corpus1=[]
for i in range(0,40213):
    cleaned_hm1=re.sub('[^a-zA-Z]', ' ', df1['cleaned_hm'][i])
    cleaned_hm1=cleaned_hm1.lower()
    cleaned_hm1=cleaned_hm1.split()
    ps=PorterStemmer()
    cleaned_hm1=[ps.stem(word1) for word1 in cleaned_hm1 if not word1 in set(stopwords.words('english'))]
    cleaned_hm1=' '.join(cleaned_hm1)
    corpus1.append(cleaned_hm1)

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=11335)
x=cv.fit_transform(corpus).toarray()
#x_test=cv.fit_transform(corpus1).toarray()
labelencoder_y=LabelEncoder()
y = labelencoder_y.fit_transform(y)
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size = 0.30, random_state=0)
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB().fit(xtrain, ytrain)
gnb_predictions = gnb.predict('a b c')
print(gnb_predictions)
accuracy = gnb.score(xtest, ytest)
print(accuracy)
out=['achievement','affection' ,'bonding', 'enjoy_the_moment' ,'exercise', 'leisure', 'nature' ]
list=[]
for i in gnb_predictions:
    '''
    list.append(i)
    temp=list[i]
    list.append(out[temp])
    '''
    list.append(out[i])
    print(out[i])
plt.plot(df['cleaned_hm'],df['reflection_period'])