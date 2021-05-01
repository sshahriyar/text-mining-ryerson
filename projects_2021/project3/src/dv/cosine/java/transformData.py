#!/usr/bin/env python3

# NOTE : Code include top row so that it runs on me machine.  This may cause issues on other machines

import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import sys
from matplotlib.table import Table

if __name__ == "__main__":

    '''
    Data File Structure :
    id	
    dateAdded	
    dateUpdated	name	
    asins	
    brand	
    categories	
    primaryCategories	
    imageURLs	
    keys	
    manufacturer	
    manufacturerNumber	
    reviews.date	
    reviews.dateSeen	
    reviews.didPurchase	
    reviews.doRecommend	
    reviews.id	
    reviews.numHelpful	
    reviews.rating	
    reviews.sourceURLs	
    reviews.text	
    reviews.title	
    reviews.username	
    sourceURLs

    File Length : 
    dataset is 28333 
    dataset2 is 34661
    dataset3 is 5000
    where original file is 100,000
    '''
    dataset1 = pd.read_csv("AmazonCustomerReviewsData/archive/Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products.csv")
    dataset2 = pd.read_csv("AmazonCustomerReviewsData/archive/Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products_May19.csv")
    dataset3 = pd.read_csv("AmazonCustomerReviewsData/archive/1429_1.csv")

    a_file = open("alldata-id_p1gram.txt", "w")

    print(dataset1.head())
    reviews = np.array(dataset1['reviews.text'])
    i = 0
    print(i,reviews.shape)
    for i,review in enumerate(reviews):
        #print(review)
        a_file.writelines("_*"+str(i)+" "+str(review)+"\n")

    print(dataset2.head())
    reviews = np.array(dataset2['reviews.text'])
    print(i,reviews.shape)
    for i,review in enumerate(reviews):
        #print(review)
        a_file.writelines("_*"+str(i)+" "+str(review)+"\n")

    print(dataset3.head())
    reviews = np.array(dataset3['reviews.text'])
    print(i,reviews.shape)
    for i,review in enumerate(reviews):
        #print(review)
        a_file.writelines("_*"+str(i)+" "+str(review)+"\n")
     
    a_file.close()