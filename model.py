# Random Forest Classification

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from time import time
# Importing the dataset
dataset = pd.read_csv('diabetes3.csv')
X = dataset.iloc[:, [0,1,2,3,4,5,6,7]].values
y = dataset.iloc[:, 8].values
start=time()

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
#X_train = sc.fit_transform(X_train)
#X_test = sc.transform(X_test)

# Fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

from sklearn.metrics import confusion_matrix,mean_squared_error
print("the mean square error",mean_squared_error(y_test, y_pred))
TP=cm[0][0]
FP=cm[0][1]
FN=cm[1][0]
TN=cm[1][1]
acc=(TP+TN)/(TP+TN+FP+FN)
recall=TP/(TP+FN)
precision=TP/(TP+FP)
F_measure=2*precision*recall/(precision+recall)
#precision,recall,threshold=precision_recall_curve(y_test, y_pred)
print("the accuracy is:",acc*100,"%")
print("the precision is:",precision*100,"%")
print("the recall is:",recall*100,"%")
print("the F-measure:",F_measure*100,"%")
end=time()
print("the time is "+str(end-start))
print(cm)
#save the model
import pickle 
pickle.dump(classifier,open("model.pkl","wb"))
# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
print(model.predict([[2, 138, 62,53,120,33.6,.12,47]]))

