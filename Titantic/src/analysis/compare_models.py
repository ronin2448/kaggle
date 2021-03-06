
import csv as csv
import numpy as np
import pandas as pd
import pylab as P

import sklearn.cross_validation as cv

import data_cleaning as dc

#Import from the random forest package
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.grid_search import GridSearchCV


def preprocess_data(fileName, dropMissingValues):
    df = pd.read_csv(fileName, header=0)
    
    print("Records read in.. %d" % len(df))
    
    df = dc.recode_variables(df)
    df = dc.replace_missing_age_values(df)
    
    df['AgeIsNull'] = pd.isnull(df.Age).astype(int)
    
    df['FamilySize'] = df['SibSp'] + df['Parch']

    df['Age*Class'] = df.AgeFill * df.Pclass



    df = df.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'Embarked', 'Fare'], axis = 1)
    df = df.drop(['Age'], axis =1)
    
    if dropMissingValues:
        df = df.dropna()

    final_data = df
    return final_data


if __name__ == '__main__':

    # For .read_csv always use header=0 when you know row 0 is the header row
    train_data = '../../data/train.csv'
    test_data = '../../data/test.csv'
    
    test_data_panda = preprocess_data(test_data, False) 
    train_data_panda = preprocess_data(train_data, True) 
    
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
    
    clf = svm.SVC()
    clf = clf.fit(X_train, y_train)
    output = clf.predict(test_data)
    
    
    
    
    print(clf.score(X_test, y_test))
    
    forest = RandomForestClassifier(n_estimators = 1000)
    
    #forest = forest.fit(X_train, y_train)
    print("Random forest ensemble built")
    #print(forest.score(X_test, y_test))
    
    scores = cv.cross_val_score(forest, training_set, target_set, cv=10, scoring='f1')
    
    
    print("Random Forest Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    
    logReg_1 = LogisticRegression()
    logReg_2 = LogisticRegression()
    tuned_parameters = [ {'C': [0.5, 1, 5, 10, 50, 100, 500, 1000] }]
    clf = GridSearchCV(logReg_1,  tuned_parameters, cv=10, scoring="f1")
    clf.fit(X_train, y_train)
    print(clf.best_estimator_)
    scores = cv.cross_val_score(logReg_2, training_set, target_set, cv=10, scoring='f1')
    #print(scores)
    print(logReg_2.C)
    print("Logistic Regression Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))