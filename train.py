from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
df=pd.DataFrame()
df=pd.read_csv("hm_train.csv")
print(df)
print(np.unique(df['predicted_category']))
x=df.iloc[:,:-1].values
df1=pd.DataFrame()
df1=pd.read_csv('hm_test.csv')
x_test=df1.iloc[:,:].values
y=df.iloc[:,4].values

#encode categorial data
labelencoder_x=LabelEncoder()
x[:,2] = labelencoder_x.fit_transform(x[:,2])
x[:,1] = labelencoder_x.fit_transform(x[:,1])
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

labelencoder_x=LabelEncoder()
x_test[:,2] = labelencoder_x.fit_transform(x_test[:,2])
x_test[:,1] = labelencoder_x.fit_transform(x_test[:,1])
#print(y)
xtrain, xtest, ytrain, ytest = train_test_split(x,y,test_size = 0.25 )
out=['achievement','affection' ,'bonding', 'enjoy_the_moment' ,'exercise', 'leisure', 'nature' ]
'''
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=7).fit(xtrain, ytrain)

# accuracy on X_test
accuracy = knn.score(xtest, ytest)
print(accuracy)

# creating a confusion matrix
knn_predictions = knn.predict(x_test)
'''
'''
from sklearn.tree import DecisionTreeClassifier
dtree_model = DecisionTreeClassifier(max_depth = 7).fit(xtrain, ytrain)
dtree_predictions = dtree_model.predict(x_test)
list=[]
'''
from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB().fit(xtrain, ytrain)
gnb_predictions = gnb.predict(x_test)

# accuracy on X_test
accuracy = gnb.score(xtest, ytest)
print(accuracy)
list=[]
for i in gnb_predictions:
    '''
    list.append(i)
    temp=list[i]
    list.append(out[temp])
    '''
    list.append(out[i])
    print(out[i])