
import csv as csv
import numpy as np
import pandas as pd
import pylab as P

import sklearn.cross_validation as cv

import datacleaning as dc
import os


def preprocessData(fileName, dropMissingValues):
    print os.getcwd()
    print(fileName)
    df = pd.read_csv(fileName, header=0)
    
    print("Records read in.. %d" % len(df))
    
    df = dc.recodeVariables(df)
    df = dc.fixMissingAgeValues(df)
    
    df['AgeIsNull'] = pd.isnull(df.Age).astype(int)
    
    df['FamilySize'] = df['SibSp'] + df['Parch']

    df['Age*Class'] = df.AgeFill * df.Pclass



    df = df.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'Embarked', 'Fare'], axis = 1)
    df = df.drop(['Age'], axis =1)
    
    if (dropMissingValues == True):
        df = df.dropna()

    final_data = df
    return final_data

def writeResultToFile(outputFileName, outputDS, predictions):
    #prediction_file = open('../../out/SVMmodel.csv', "wb")
    prediction_file = open(outputFileName, "wb")
    prediction_file_object = csv.writer(prediction_file)
    
    prediction_file_object.writerow(["PassengerId", "Survived"])


    i = 0    
    for row in outputDS.values:
        passenger_id = str(int(row[0]))
        result = predictions[i]
        prediction_file_object.writerow([passenger_id, "%d" % result]) 
        i = i+1
    
    prediction_file.close()
    print("Finished writing ouput")


# For .read_csv always use header=0 when you know row 0 is the header row
train_data = '../../data/train.csv'
test_data = '../../data/test.csv'

test_data_panda = preprocessData(test_data, False) 
train_data_panda = preprocessData(train_data, True) 

test_data_panda_orig = test_data_panda

train_data_panda = train_data_panda.drop(['PassengerId'], axis = 1)
test_data_panda =  test_data_panda.drop(['PassengerId'], axis = 1)

train_data = train_data_panda.values
test_data = test_data_panda.values

print("\nTraining Data set built.  There are %d entries" % len(train_data))
print("\nTest Data set built.  There are %d entries" % len(test_data))
print("\norig Test Data set built.  There are %d entries" % len(test_data_panda_orig))


training_set = train_data[0::,1::]
target_set = train_data[0::,0]
X_train, X_test, y_train, y_test = cv.train_test_split(training_set, target_set, test_size=0.4, random_state=0)

#Import from the random forest package
from sklearn import svm

clf = svm.SVC()
clf = clf.fit(X_train, y_train)
output = clf.predict(test_data)

#create the random forest object which will include all the parameters for fit

#Fit the training data to the Survived labels and create the decision trees


#Take the same decision trees and run it on the test data
#output = forest.predict(test_data)

print("Predictions made")
#writeResultToFile('../../out/SVMmodel.csv', test_data_panda_orig, output)


print(clf.score(X_test, y_test))

from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(n_estimators = 1000)
#forest = forest.fit(X_train, y_train)
print("Random forest ensemble built")
#print(forest.score(X_test, y_test))

scores = cv.cross_val_score(forest, training_set, target_set, cv=10, scoring='f1')
#print(scores)
print("Random Forest Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

from sklearn.linear_model import LogisticRegression
logReg = LogisticRegression()
scores = cv.cross_val_score(logReg, training_set, target_set, cv=10, scoring='f1')
#print(scores)
print("Logistic Regression Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))