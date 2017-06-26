# Name of file: textMining.py
#
# Implementation: This program creates a classifier based on
#                 Naive Bayes Algorithm to classify news articles
#                 into the topics they news articles belong to.
#
# An undirected graph is passed in as a text file (first command line argument).
#
# Training data set: 20_newsgroups, containing sub-folders of news articles,
#                   on different topics.
#
# Author(s): Abhishek Jaitley
#            Mansa Pabbaraju
#            Palash Gandhi
#
# Packages required: sklearn, os, re
# Reference for scikit:
# Scikit-learn: Machine Learning in Python, Pedregosa et al., 
#                                         JMLR 12, pp. 2825-2830, 2011. 
# Reference for stopWords.txt: 
#     RIT mycourses/CSCI.720.03 - Big Data Analytics /Resources /StopWords

#! /usr/local/bin/python3.5
#import necessary packages
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn import metrics
import os
import re
import sys
'''
Function Name: populateDirList()
Input parameters: newsGroupDirectoryPath, 
                  string containing directory path
Working: This functions reads all the folders in t
          he provided directory path
Return value: dirList - list of directories present 
              input directory
'''

def populateDirList(newsGroupDirectoryPath):
    dirList = []
    first=True
    for x in os.walk(newsGroupDirectoryPath) :
        if first:
            first =False
            continue
        else :
            path= x[0].split(newsGroupDirectoryPath)
            dirList.append(path[1])
    return dirList


'''
Function Name: getTextData()
Input parameters: filepath, string containing file path
Working: This function reads in the text data from
         the file location specified
Return value: data - entire text read from file in
         single sring format
'''
values = ['Newsgroups:','Xref:', 'Path:', 'From:', 'Subject:','Message-ID:','Sender:', 'Organization:',
          'References:','Date:', 'Lines:']
def getTextData(FilePath) :
    with open(FilePath,"r",encoding="utf-8",errors='ignore') as f:
        text=''
        for line in f:
            re.sub("\d+", "", line)
            words=line.split(' ')
            #Removing the newsgroup value
            firstWord = words[0]
            #if firstWord in values:
            if firstWord in values:
                continue
            else :
                text+=line
                text+=' '
    return text


'''
Function Name: createModel()
Input parameters: None
Working: This function does the following:
         1. Reads all the files present
            in all sub-folders of teh data set,
            makes a 60-20-20 split for the training,
            testing and validation data set
         2. Cleans the data by removing stopwords
         3. Builds a TFIDF for entire Training data corpus
         4. Builds a Naive Bayes algorithm based model
            from sklearn library available in python
         5. Runs the model built on training and validation
            data set
Return value: None
'''
def createModel():

    #initialize variables
    TrainingCorpus = []
    TrainingDataClasses = []
    TestData=[]
    TestDataClasses = []
    ValidateData = []
    ValidateDataClasses = []
    #get the base directory for the 20_newsgroups folder
    baseDir = sys.argv[1:]
    baseDir = baseDir[0]

    #store all newsgroup topics in a list
    dirList = populateDirList(baseDir)

    # loop over every sub-folder in the directory
    for i in range(0,len(dirList)) :
        currDir = baseDir+dirList[i]

        # loop over every file in every sub-folder
        fileCounter = 0
        total_documents = len(os.listdir(currDir))
        num_training = round(0.6*total_documents)
        remaining = total_documents - num_training
        num_testing = round(0.5* remaining)
        testing_index_end = num_training + num_testing

        for filename in os.listdir(currDir):
            # store the topic name for this file, taken from sub-folder name
            newsGrp = dirList[i]
            text = getTextData(currDir+'/'+filename)
            # Training data
            if(fileCounter < num_training) :

                 TrainingCorpus.append(text)
                 TrainingDataClasses.append(newsGrp)

            # Testing data
            elif fileCounter >=num_training and  fileCounter < testing_index_end:
                 TestData.append(text)
                 TestDataClasses.append(newsGrp)

            # Validation data
            elif(fileCounter >= testing_index_end) :

                 ValidateData.append(text)
                 ValidateDataClasses.append(newsGrp)

            fileCounter += 1
            # reached end of data set
            if fileCounter >= total_documents:
                break

    # Take path for stop-words file
    stopwords_text = getTextData(os.getcwd() + '/stopWords.txt')
    # create a list of the stopwords
    stopwords_list = stopwords_text.rsplit(' ')

    '''
    Build a Term Frequency-Inverse Document Frequency matrix
    using the CountVectorizer() and TfidfTransformer
    function tool available in python sklearn package
    '''
    # Create a vectorizer, which creates a bag of
    # words, removing all the custom stopwords
    bagOfWords = CountVectorizer(stopwords_list)
    # Transform training data files into materix form
    Traindtm = bagOfWords.fit_transform(TrainingCorpus)
    tfidfTrans = TfidfTransformer()
    # Create TDIDF matrix
    trainTf = tfidfTrans.fit_transform(Traindtm)
    # Naive Bayes algorithm
    naivebayes = MultinomialNB()
    nbClassifier = naivebayes.fit(trainTf, TrainingDataClasses)

    #run on Test data
    runModel(1,TestData,TestDataClasses,bagOfWords,tfidfTrans,nbClassifier)

    #run on Validate data
    runModel(2,ValidateData,ValidateDataClasses,bagOfWords,tfidfTrans,nbClassifier)

'''
Function Name: runModel()
Input parameters:
         1. type - 1 for Test data, 2 for Validation data
         2. inputdata - Test/Validation data
         3. inputclasses - Final classifier values for
           corresponding input data, for model-evaluation
         4. bagOfWords - containing word counts
         5. tfidfTrans - TDIDF object to calculate TDIDF for input data
         6. nbClassifier - Naive Bayes classfier model, on which model is run
Working: This function does the following:
         1. Creates a TDIDF on the input data the model is to be run on
         2. Runs the classifier model built on the input data
         3. Using the metrics package available in python,
             calculates the accuracy rate of the model
         4. Builds a Naive Bayes algorithm based model
            from sklearn library available in python
         5. Creates the confusion matrix on te model built and displays
            the same
Return value: None
'''
def runModel(type,inputdata,inputclasses,bagOfWords,tfidfTrans,nbClassifier):
    predicted = []
    for i in range(len(inputdata))  :
        testvectr = []
        testvectr.append(inputdata[i])
        testdtm= bagOfWords.transform(testvectr)
        testingTfidf = tfidfTrans.transform(testdtm)
        predictedVal = nbClassifier.predict(testingTfidf)
        predicted.append(predictedVal)

    #Calculation of Accuracy and Confusion Matrix
    accuracy=metrics.accuracy_score(inputclasses, predicted)
    # testing data
    if type == 1:
        print('Accuracy of Test Dataset is: ',accuracy*100)
    # validation datA
    else:
        print('Accuracy of Validation Dataset is: ',accuracy*100)

    #Confusion Matrix
    confusionMatrix = metrics.confusion_matrix(inputclasses, predicted)
    print('Confusion matrix of model is:', confusionMatrix)

def main():
    createModel()

main()