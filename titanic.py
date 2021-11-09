
#importing important lib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data_train=pd.read_csv("train.csv")# taking training data from csv file
data_test=pd.read_csv("Test.csv")# taking test data
data_train.shape#finding the shape of the taraning data

data_test.shape#finding the shape of the test data

data_train['Survived'].value_counts() #This gives the counts of death and lived rate in the csv 0= dead and 1 is alive

sns.countplot(data_train['Survived'],label="Count")#shows the bar chart for the above count using seaborn

data_train.groupby('Sex')[['Survived']].mean()#this is used to find the mean value for the survuvred rate of male and female

data_train.pivot_table('Survived', index='Sex', columns='Pclass') #gives a ratio for male and female survived with respect to the class

data_train.pivot_table('Survived', index='Sex', columns='Pclass').plot()

sns.barplot(x='Pclass', y='Survived', data=data_train)

age = pd.cut(data_train['Age'], [0, 18, 80])#restring the final data of the age or grouping it to 0-18 and 18-80
data_train.pivot_table('Survived', ['Sex', age], 'Pclass')

data_train.isna().sum()#checking the blanks in the data

for val in data_train:
   print(data_train[val].value_counts())
   print()

data_train = data_train.drop(['Cabin','Ticket', 'Name'], axis=1)#cabin, name, ticket are irrelevent so the table is dropped they are non integer

data_train = data_train.dropna(subset =['Embarked', 'Age'])#droping the blank row from the data

data_train.shape#finding the shape of the data after non relevant data are removed

print(data_train['Sex'].unique())#finding uniqueness in the sex
print(data_train['Embarked'].unique())

data_train.dtypes #finding the data types of the training data

from sklearn.preprocessing import LabelEncoder#preparing the data to renamed or relabeled
labelencoder = LabelEncoder()
data_train.iloc[:,3]= labelencoder.fit_transform(data_train.iloc[:,3].values)#giving the position of sex

data_train.iloc[:,8]= labelencoder.fit_transform(data_train.iloc[:,8].values)#giving the position of Embarked

print(data_train['Sex'].unique())
print(data_train['Embarked'].unique())

X = data_train.iloc[:, 2:9].values #giving values to x axis
Y = data_train.iloc[:, 1].values #giving values to y axis

from sklearn.model_selection import train_test_split#spliting data for training and testing
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

from sklearn.preprocessing import StandardScaler#preprossing the data
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

def models(X_train,Y_train):
  
  #Using Logistic Regression Algorithm to the Training Set
  from sklearn.linear_model import LogisticRegression
  log = LogisticRegression(random_state = 0)
  log.fit(X_train, Y_train)
  
  #Using KNeighborsClassifier Method of neighbors class to use Nearest Neighbor algorithm
  from sklearn.neighbors import KNeighborsClassifier
  knn = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
  knn.fit(X_train, Y_train)

  #Using SVC method of svm class to use Support Vector Machine Algorithm
  from sklearn.svm import SVC
  svc_lin = SVC(kernel = 'linear', random_state = 0)
  svc_lin.fit(X_train, Y_train)

  #Using SVC method of svm class to use Kernel SVM Algorithm
  from sklearn.svm import SVC
  svc_rbf = SVC(kernel = 'rbf', random_state = 0)
  svc_rbf.fit(X_train, Y_train)

  #Using GaussianNB method of naïve_bayes class to use Naïve Bayes Algorithm
  from sklearn.naive_bayes import GaussianNB
  gauss = GaussianNB()
  gauss.fit(X_train, Y_train)

  #Using DecisionTreeClassifier of tree class to use Decision Tree Algorithm
  from sklearn.tree import DecisionTreeClassifier
  tree = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
  tree.fit(X_train, Y_train)

  #Using RandomForestClassifier method of ensemble class to use Random Forest Classification algorithm
  from sklearn.ensemble import RandomForestClassifier
  forest = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
  forest.fit(X_train, Y_train)
  
  #print model accuracy on the training data.
  print('[0]Logistic Regression Training Accuracy:', log.score(X_train, Y_train))
  print('[1]K Nearest Neighbor Training Accuracy:', knn.score(X_train, Y_train))
  print('[2]Support Vector Machine (Linear Classifier) Training Accuracy:', svc_lin.score(X_train, Y_train))
  print('[3]Support Vector Machine (RBF Classifier) Training Accuracy:', svc_rbf.score(X_train, Y_train))
  print('[4]Gaussian Naive Bayes Training Accuracy:', gauss.score(X_train, Y_train))
  print('[5]Decision Tree Classifier Training Accuracy:', tree.score(X_train, Y_train))
  print('[6]Random Forest Classifier Training Accuracy:', forest.score(X_train, Y_train))
  
  return log, knn, svc_lin, svc_rbf, gauss, tree, forest

model = models(X_train,Y_train)

from sklearn.metrics import confusion_matrix 
for i in range(len(model)):
   cm = confusion_matrix(Y_test, model[i].predict(X_test)) 
   #extracting TN, FP, FN, TP
   TN, FP, FN, TP = confusion_matrix(Y_test, model[i].predict(X_test)).ravel()
   print(cm)
   print('Model[{}] Testing Accuracy = "{} !"'.format(i,  (TP + TN) / (TP + TN + FN + FP)))
   print()

forest = model[6]#importance iof random forest
importances = pd.DataFrame({'feature':data_train.iloc[:, 2:9].columns,'importance':np.round(forest.feature_importances_,3)})
importances = importances.sort_values('importance',ascending=False).set_index('feature')
importances.plot.bar()

data_test.isna().sum()

data_test.dtypes

data_test = data_test.drop(['Cabin','Ticket', 'Name'], axis=1)
data_test= data_test.dropna(subset =['Embarked', 'Age','Fare'])
data_test.dtypes

from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
data_test.iloc[:,2]= labelencoder.fit_transform(data_test.iloc[:,2].values)

data_test.iloc[:,7]= labelencoder.fit_transform(data_test.iloc[:,7].values)

print(data_test['Sex'].unique())
print(data_test['Embarked'].unique())

data_test.head(10)

last = data_test.iloc[:, 1:8].values 
myid=892

my=int(input("Enter the model number "))
pred = model[my].predict(last)
print(pred)

list1 = pred.tolist()
print(list1)

print("PassengerID \t   Survivual")
naman=892
for i in list1:
  print(naman,"\t\t\t",i)
  naman=naman+1

