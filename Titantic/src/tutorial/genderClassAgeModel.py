'''
Created on Jun 23, 2014

@author: Domingo
'''

import csv as csv
import numpy as np

if __name__ == '__main__':
    
    csv_file_object = csv.reader( open('../data/train.csv', 'rb') )
    print("File read")
    
    header = csv_file_object.next()   #next() skips the first line which is the header
    
    data = []  # create a list object called data
    
    for row in csv_file_object:
        data.append(row)
    
    data = np.array(data) # convert from a list to an array.  Be aware each item is currently a string in this format

    # do analysis by binning class and ages...
    fare_ceiling = 40
    
    #the modify the data in the Fare column to = 39, if greater or equal to ceiliing
    data[ data[0::,9].astype(np.float) >= fare_ceiling, 9  ] = fare_ceiling - 1.0    
    
    fare_bracket_size = 10
    number_of_price_brackets = fare_ceiling / fare_bracket_size
    
    #But it is better practice to calculate from data
    # Take the length of an array of unique values in column index 2
    number_of_classes = len(np.unique(data[0::,2]))

    
    # Initialize the survival table with all zeros
    survival_table = np.zeros((2, number_of_classes, number_of_price_brackets))
    
    
    for i in xrange(number_of_classes):  #foreach class
        for j in xrange(number_of_price_brackets): # freach price bin
            
            women_only_stats = data[ 
                                    ( data[0::,4] == "female") # is a female
                                    & (data[0::,2].astype(np.float) == i+1)  # was in the ith class
                                       & (data[0:,9].astype(np.float) >= j*fare_bracket_size) # was greater than this bin 
                                       & (data[0:,9].astype(np.float) < (j+1)*fare_bracket_size) # was less than the next bin in the 2nd col
                                       , 1]

            men_only_stats = data[ 
                                    ( data[0::,4] != "female") # is a male
                                    & (data[0::,2].astype(np.float) == i+1)  # was in the ith class
                                       & (data[0:,9].astype(np.float) >= j*fare_bracket_size) # was greater than this bin 
                                       & (data[0:,9].astype(np.float) < (j+1)*fare_bracket_size) # was less than the next bin in the 2nd col
                                       , 1]
            
            survival_table[0,i,j] = np.mean(women_only_stats.astype(np.float))
            survival_table[1,i,j] = np.mean(men_only_stats.astype(np.float))
            survival_table[ survival_table != survival_table] = 0
    
    print(survival_table)
    
    survival_table[ survival_table < 0.5 ] = 0
    survival_table[ survival_table >= 0.5] = 1

        
    test_file = open('../data/test.csv', 'rb')
    test_file_object = csv.reader(test_file)
    header = test_file_object.next()
    
    prediction_file = open('../data/genderclassmodel.csv', "wb")
    prediction_file_object = csv.writer(prediction_file)
    
    prediction_file_object.writerow(["PassengerId", "Survived"])
    
    for row in test_file_object:
        
        for j in xrange(number_of_price_brackets):
            
            try:
                
                row[8] = float(row[8])
            except:
                bin_fare = 3 - float(row[1])
                break
            if (row[8] > fare_ceiling):
                
                bin_fare = number_of_price_brackets - 1
                break
            if (row[8] >= j * fare_bracket_size) and (row[8] < (j+1)*fare_bracket_size) :
                bin_fare = 3
                break
            
            
        if (row[3] == 'female') :
            result = int(survival_table[0, float(row[1])-1, bin_fare ])
            prediction_file_object.writerow([row[0], "%d" % result]) # predict 1
        else:
            result = int(survival_table[1, float(row[1])-1, bin_fare ])
            prediction_file_object.writerow([row[0], "%d" % result]) # predict 0
        
    
    
    
    test_file.close()
    prediction_file.close()
    
    print("Finished Predicting")

    pass