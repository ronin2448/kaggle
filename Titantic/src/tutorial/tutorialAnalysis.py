'''
Created on Jun 22, 2014

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
    print("Converted data into a Numpy array")
    
    print(data)
    
    #The size () function counts how many elements are in  the array
    # and sum() sums them
    
    number_passengers = np.size( data[0::, 1].astype(np.float) )
    number_survived = np.sum( data[0::,1].astype(np.float) )
    proportion_survivors = number_survived / number_passengers
    
    print(number_passengers)
    print(number_survived)
    print(proportion_survivors)
    
    women_only_stats = data[0::,4] == "female" # This finds where all  the elements in the gender column that equals female
    men_only_stats = data[0::,4] != "female"   # This finds where all the elements do not equal female (i.e. male)
    
    #using the index from above we select the females and males seperately
    women_onboard = data[women_only_stats,1].astype(np.float)
    men_onboard = data[men_only_stats,1].astype(np.float)
    
    # Then we find the proportions of them that survived
    proportion_women_survived = np.sum(women_onboard) / np.size(women_onboard)
    proportion_men_survived = np.sum(men_onboard) / np.size(men_onboard)
    
    #print results!
    print 'Proportion of women who survived is %s' % proportion_women_survived
    print 'Proportion of men who survived is %s' % proportion_men_survived
    
    # do analysis by binning class and ages...
    fare_ceiling = 40
    
    #the modify the data in the Fare column to = 39, if greater or equal to ceiliing
    data[ data[0::,9].astype(np.float) >= fare_ceiling, 9  ] = fare_ceiling - 1.0    
    
    fare_bracket_size = 10
    number_of_price_brackets = fare_ceiling / fare_bracket_size
    
    #I know there were at least 3 classes on board
    number_of_classes = 3
    
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
                                            
    print(survival_table)

    pass