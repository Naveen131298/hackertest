import pandas as pd
from sklearn.preprocessing import LabelEncoder
data=pd.read_csv('train.csv')
data_test=pd.read_csv('test.csv')


y=data.loc[:,'Result']
x=data.iloc[:,1:-1]
labelencoder_y=LabelEncoder()
y=labelencoder_y.fit_transform(y)


from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size = 0.30, random_state=0)
from sklearn.svm import SVC
svc=SVC()
svc.fit(xtrain,ytrain)
submission=pd.DataFrame(data_test['ID'],columns=['ID'])
data_test.drop(['ID'],axis=1,inplace=True)
predict=(svc.predict(data_test))
collection=['h0','h1','h2','h3','h4','h5']
lis=[]
for i in predict:
    lis.append(collection[i])
# print(data_test)
data_test=pd.DataFrame(lis,columns=['Results'])
data_test.to_csv('submission.csv',index=False)
submission['Results'] = lis
submission.to_csv('submission.csv',index = False)
print(submission)

