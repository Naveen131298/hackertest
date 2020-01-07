import pandas as pd
data=pd.read_csv('hm_train.csv')
data_test=pd.read_csv('hm_test.csv')
hm_test=data_test['cleaned_hm']
x=data['cleaned_hm']
#,'hmid','reflection_period','num_sentence']]

y=data['predicted_category']
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
count=CountVectorizer()

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.4,random_state=1)


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

clf=LinearSVC(random_state=0)
tf=Pipeline([('tfidf',TfidfVectorizer()),('clf',LinearSVC(random_state=0))])
print(x_train.shape,y_train.shape)
tf.fit(x_train,y_train)
pre=tf.predict(['I spent the weekend in Chicago with my friends'])
#data_test['predicted_category']=pred
print(pre)


#from sklearn.metrics import confusion_matrix,classification_report
# print(pred)
#data_test_csv={'hmid':data_test['hmid'],'predicted_category':data_test['predicted_category']}
#df=pd.DataFrame.from_dict(data_test_csv)
#df.to_csv('df.csv')