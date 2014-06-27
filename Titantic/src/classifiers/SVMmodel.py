
# coding: utf-8

# In[1]:

import csv as csv
import numpy as np
import pandas as pd
import pylab as P


def preprocessData(fileName, dropMissingValues):
    df = pd.read_csv(fileName, header=0)
    
    print("Records read in.. %d" % len(df))

    # Map gender column Female = > 0 and Male => 1
    df['Gender'] = df['Sex'].map( {'female':0 ,'male':1} ).astype(int)
    median_ages = np.zeros((2,3))
    for i in range(0,2):
        for j in range(0,3):
            median_ages[i,j] = df[ (df['Gender'] == i) & (df['Pclass']==j+1)]['Age'].dropna().median()
            
    df['AgeFill'] = df['Age']
    
    for i in range(0,2):
        for j in range(0,3) :
            df.loc[ (df.Age.isnull()) & (df.Gender == i) & (df.Pclass == j+1), 'AgeFill'] = median_ages[i,j]

    df['AgeIsNull'] = pd.isnull(df.Age).astype(int)
    
    df['FamilySize'] = df['SibSp'] + df['Parch']

    df['Age*Class'] = df.AgeFill * df.Pclass



    df = df.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'Embarked', 'Fare'], axis = 1)
    df = df.drop(['Age'], axis =1)
    
    if (dropMissingValues == True):
        df = df.dropna()

    final_data = df
    return final_data


# For .read_csv always use header=0 when you know row 0 is the header row
train_data = 'C:/Users/Domingo/Documents/code_workspace/python_practice/Titantic_Machine_Learning/data/train.csv'
test_data = 'C:/Users/Domingo/Documents/code_workspace/python_practice/Titantic_Machine_Learning/data/test.csv'

test_data_panda = preprocessData(test_data, False) 
train_data_panda = preprocessData(train_data, True) 

test_data_panda_orig = test_data_panda

train_data_panda = train_data_panda.drop(['PassengerId'], axis = 1)
test_data_panda =  test_data_panda.drop(['PassengerId'], axis = 1)

print test_data_panda_orig[ test_data_panda_orig['PassengerId'] == 1044]

train_data = train_data_panda.values
test_data = test_data_panda.values

print("\nTraining Data set built.  There are %d entries" % len(train_data))
print("\nTest Data set built.  There are %d entries" % len(test_data))
print("\norig Test Data set built.  There are %d entries" % len(test_data_panda_orig))

test_file = open('C:/Users/Domingo/Documents/code_workspace/python_practice/Kaggle_Python/Titantic/data/test.csv', 'rb')
test_file_object = csv.reader(test_file)
header = test_file_object.next()


#Import from the random forest package
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm

clf = svm.SVC()
clf = clf.fit(train_data[0::,1::], train_data[0::,0])
output = clf.predict(test_data)

#create the random forest object which will include all the parameters for fit
#forest = RandomForestClassifier(n_estimators = 100)

#Fit the training data to the Survived labels and create the decision trees
#forest = forest.fit(train_data[0::,1::], train_data[0::,0])

print("Random forest ensemble built")

#Take the same decision trees and run it on the test data
#output = forest.predict(test_data)

print("Predictions made")


prediction_file = open('C:/Users/Domingo/Documents/code_workspace/python_practice/Kaggle_Python/Titantic/out/SVMmodel.csv', "wb")
prediction_file_object = csv.writer(prediction_file)
    
prediction_file_object.writerow(["PassengerId", "Survived"])


i = 0    
for row in test_data_panda_orig.values:
    passenger_id = str(int(row[0]))
    result = output[i]
    prediction_file_object.writerow([passenger_id, "%d" % result]) 
    i = i+1
    
prediction_file.close()
print("Finished writing ouput")

