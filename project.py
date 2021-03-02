import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier

print("============Suppervised learning project by all classifiers================")
df = pd.read_csv('titanic.csv')
print(df)
count = df["Cabin"].isna().sum()
print(count)
a = df.info()
print(a)
#Not solve by label encoder
#Making dummy values for Embarked column data to understand it by computer
ports = pd.get_dummies(df.Embarked, prefix='Embarked')
print(ports.head)
#concatenate the ports data with orginal dataset
df = df.join(ports)
print("==============New Data===========")
print(df)
#Update the original dataset
df.drop(['Embarked'],axis = 1,inplace = True)
print(df)
enc = preprocessing.LabelEncoder()
df.Sex = enc.fit_transform(df['Sex'])
print(df['Sex'])
x = df.Survived.copy()
y = df.drop(['Survived'], axis=1)
print(x)
print(y)
df.drop(['Cabin','Ticket','Name','PassengerId'], axis= 1, inplace = True)
print(df)
print(df.info)
print(df.isnull().values.any())
df.Age.fillna(df.Age.mean(), inplace=True)
print(df.isnull().values.any())
print(df['Age'])
print(df)
x = df.iloc[:,2:13]
y=df.iloc[:,:1]
print(x)
print(y)
x_train,x_test,y_train,y_test = train_test_split(df,y,test_size=0.3)
print("=============X-Train==============")
print(x_train)
print("=============Y-Train==============")
print(y_train)
print("==================X-Test============")
print(x_test)
print("=================Y-Test=============")
print(y_test)
#sur = df['Survived']
#plt.pie(sur)
#age = df['Age']
#sb.distplot(age)
#plt.hist(age)
#fare = df['Fare']
#plt.hist(fare)
#survive = df['Survived'].value_counts()
#print(survive)
#sb.countplot(survive)
#plot price paid of each class
# plt.scatter(df['Fare'],df['Pclass'],label='Paid Passenger')
# plt.ylabel('Class')
# plt.xlabel('Fare')
# plt.title('Price of Each Class')
# plt.legend()
# plt.show()  
#gender = df.Sex.value_counts()
#plt.pie(gender, labels=['Males','Females'])
#plt.title('Sex')   
#fare plot by age
# age = df['Age']
# fare = df['Fare']
# plt.bar(age,fare)
# plt.title('Fare distribution by age')
# plt.xlabel('Age')
# plt.ylabel('Fare')
#sex = df['Sex'].value_counts()
#bar = sb.countplot(sex)
print("==================Logistic Regression===================")
lr = LogisticRegression()
logreg = lr.fit(x_train,y_train)
prediction_lr = logreg.predict(x_test)
print("================Prediction of Model=================")
print(prediction_lr)
print("===========Actual Answers==========")
print(y_test)
print("==============Accuracy============")
print("===================Training Accuracy===========")
ta = lr.score(x_train,y_train)
trainingAcc = ta*100
print(trainingAcc)
print("=====================Testing Accuracy================")
ta1 = accuracy_score(y_test,prediction_lr)
testAcc = ta1*100
print(testAcc)

print("=================K-Nearest Neighbor==================")
kn = KNeighborsClassifier(n_neighbors = 3)
knn = kn.fit(x_train,y_train)
prediction_knn = knn.predict(x_test)
print("===========Prediction Of Model===========")
print(prediction_knn)
print("==============Actual Answers==============")
print(y_test)
print("=================Accuracy================")
print("================Training Accuracy============")
ta = kn.score(x_train,y_train)
trainingAcc = ta*100
print(trainingAcc)
print("=============Testing Accuracy===============")
ta1 = accuracy_score(y_test,prediction_knn)
testAcc = ta1*100
print(testAcc)

print("======================Naive Bayes============")
n = GaussianNB()
nb = n.fit(x_train,y_train)
prediction_nb = nb.predict(x_test)
print("===================Prediction of Model=================")
print(prediction_nb)
print("================Actual Answers=================")
print(y_test)
print("=====================Accuracy=================")
print("=================Training Accuracy==================")
ta = n.score(x_train,y_train)
trainingAcc = ta*100
print(trainingAcc)
print("======================Testing Accuracy===========")
ta1 = accuracy_score(y_test,prediction_nb)
testAcc = ta1*100
print(testAcc) 

print("=====================Random Forest=============")
r = RandomForestClassifier(n_estimators=50)
rf = r.fit(x_train,y_train)
prediction_rf = rf.predict(x_test)
print("===================Prediction Of Model==================")
print(prediction_rf)
print("================Actual Answers===============")
print(y_test)
print("================Accuracy=================")
print("==============Training Accuracy================")
ta = r.score(x_train,y_train)
trainingAcc = ta*100
print(trainingAcc)
print("===============Testing Accuracy===================")
ta1 = accuracy_score(y_test,prediction_rf)
testAcc = ta1*100
print(testAcc)

print("===================Support Vector Machine================")
sv = svm.SVC()
svm = sv.fit(x_train,y_train)
prediction_svm = svm.predict(x_test)
print("============Prediction Of Model===============")
print(prediction_svm)
print("=============Actual Answers================")
print(y_test)
print("====================Accuracy=================")
print("=============Training Accuracy================")
ta = sv.score(x_train,y_train)
trainingAcc = ta*100
print(trainingAcc)
print("================Testing Accuracy===================")
ta1 = accuracy_score(y_test,prediction_svm)
testAcc = ta1*100
print(testAcc)

print("=========================Decision Tree===============")
d = DecisionTreeClassifier(random_state=7)
dt = d.fit(x_train,y_train)
prediction_dt = dt.predict(x_test)
print("===============Prediction Of Model===================")
print(prediction_dt)
print("=================Actual Answers=================")
print(y_test)
print("=================Accuracy====================")
print("===================Training Accuracy====================")
ta = d.score(x_train,y_train)
trainingAcc = ta*100
print(trainingAcc)
print("==================Testing Accuracy================")
ta1 = accuracy_score(y_test,prediction_dt)
testAcc = ta1*100
print(testAcc)
print("==========================")
df2 = pd.DataFrame(y_test).reset_index()
b = df2.iloc[:,1:]
print(b)
classifiers = {'prediction_lr': prediction_lr,
               'prediction_knn': prediction_knn,
               'prediction_svm': prediction_svm,
               'prediction_nb': prediction_nb,
               'prediction_rf': prediction_rf,
               'prediction_dt': prediction_dt}

dff = pd.DataFrame(classifiers)


print(a)
dff["Y_Test Values"]=b
dff.to_csv('Comparison.csv')
