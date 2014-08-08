
import csv as csv
import numpy as np
import pandas as pd
import pylab as P

import datacleaning as dc
import sklearn.cross_validation as cv


#Import Classifiers for this model
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression



"""  thi module does blah """
def preprocess_data(fileName, dropMissingValues):
    print(fileName)
    df = pd.read_csv(fileName, header=0)
    
    print("Records read in.. %d" % len(df))
    
    df['AgeIsNull'] = pd.isnull(df.Age).astype(int)
    
    df = dc.recode_variables(df)
    df = dc.add_title_variable(df)
    df = dc.replace_missing_age_values(df)
    df = dc.replace_missing_age_values_using_titles(df)
    
    
    df['FamilySize'] = df['SibSp'] + df['Parch']
    df['Age*Class'] = df.AgeFill * df.Pclass


    df = df.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'Fare', 'Title'], axis = 1)
    df = df.drop(['Age'], axis =1)
    
    if dropMissingValues:
        df = df.dropna()

    final_data = df
    return final_data

def write_results(outputFileName, outputDS, predictions):

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

if __name__ == '__main__':    
    # For .read_csv always use header=0 when you know row 0 is the header row
    train_data = '../../data/train.csv'
    test_data = '../../data/test.csv'
    
    # Preprocess Data by converting the files into data frames
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
    
    forest = RandomForestClassifier(n_estimators = 1000)
    
    #Test Random Forest Ensemble
    scores = cv.cross_val_score(forest, training_set, target_set, cv=10, scoring='f1')
    print("Random Forest Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    
    
    #logReg = LogisticRegression(C=0.003)
    logReg = LogisticRegression(C=100, class_weight=None, dual=False, fit_intercept=True,
          intercept_scaling=1, penalty='l2', random_state=None, tol=0.0001)

    logReg.fit(training_set, target_set)
    #print(logReg.coef_)
    scores = cv.cross_val_score(logReg, training_set, target_set, cv=10, scoring='f1', verbose=1)
    print(logReg.coef_)
    print("Logistic Regression Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    
    output = logReg.predict(test_data)
    print("Predictions made")
    write_results('../../out/RegLogRegModelWithTitles.csv', test_data_panda_orig, output)
