'''
Created on Jun 23, 2014

@author: Domingo
'''

import csv as csv
import numpy as np

if __name__ == '__main__':
    
    test_file = open('../data/test.csv', 'rb')
    test_file_object = csv.reader(test_file)
    header = test_file_object.next()
    
    prediction_file = open('../data/genderbasedmodel.csv', "wb")
    prediction_file_object = csv.writer(prediction_file)
    
    prediction_file_object.writerow(["PassengerId", "Survived"])
    
    for row in test_file_object:
        if (row[3] == 'female') :
            prediction_file_object.writerow([row[0], '1']) # predict 1
        else:
            prediction_file_object.writerow([row[0], '0']) # predict 0
        
    test_file.close()
    prediction_file.close()
             
    pass