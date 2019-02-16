#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 24 13:24:12 2018

@author: zhangjinlong
"""
import helper
from sklearn import svm
###把 helper 输出的test 转化为dictionary
####把test 文件转化为dictionary，每个单词对应其出现的次数
##key 是输入class的所有单词，key 对应的值是单词出现的次数
def WordFreDict(text):
        wordDict = {}        
        for row in text:
            for word in row:
                
                if word not in wordDict:
                    wordDict[word] = 1
                else:
                    wordDict[word] = wordDict[word] + 1
        vocalList = list(wordDict.keys())
        return wordDict

### test vector 有5718个特征，特征的排列与FeatureList 相同
def get_test_vector(class_test,FeatureList):
    rows = len(class_test)
    colums = len(FeatureList)
    test_vector=[[0 for _ in range(colums)] for _ in range(rows)]
    
    for i in range(rows):
        for j in range(len(class_test[i])):
            if class_test[i][j] in FeatureList:
                feature_index = FeatureList.index(class_test[i][j])
                test_vector[i][feature_index]+=1
        #sum_doc=sum(test_vector[i])
        #if sum_doc == 0:
         #   print(class_test[i])
    return test_vector

def fool_classifier(test_data):
    
    ##提取出class 1 和 class 0，获取training data
    strategy=helper.strategy()
    class1 = strategy.class1
    class0 = strategy.class0
    total_train = class1 + class0
        
    ##change training data  to dictionary
    total_train_dict=WordFreDict(total_train)
    
    ##get the feature_list
    feature_list = list(total_train_dict.keys())
    
    ##get the count vector for class 1/class 0 list as feature_list
    class1_vector = get_test_vector(class1,feature_list)
    class0_vector = get_test_vector(class0,feature_list)
    
    ## get the label y for class1/class0
    y_train1 = [1 for _ in range(len(class1_vector))]
    y_train0 = [0 for _ in range(len(class0_vector))]
    
    ##get the final x_train/y_train
    x_train=class1_vector+class0_vector
    y_train=y_train1+y_train0
    
   ##create svm
    parameters={'gamma':2, 'C': 4, 'kernel':'linear', 'degree':1.0, 'coef0':0.001}
  
   
    svm =strategy.train_svm(parameters, np.asarray(x_train), np.asarray(y_train))
    
    ##store the test data 
    test_data_set=[]
    with open(test_data,'r') as test:
        for lines in test:
            test_data_set.append(list(set(lines.strip().split(' '))))
    print(test_data_set[0])
            
    ##get the feature_list weight dict
    feature_weight_dict={}
    weights=svm.coef_[0]
    

    for i in range(len(feature_list)):
        feature_weight_dict[feature_list[i]]=weights[i]
       
    
    
    ## create the modified_data.txt
    with open("modified_data.txt","w") as f:
        ##list the weight for each doc
        for line in test_data_set:
            temp_weight=[]
            doc=line.copy()
            for word in doc:
                if word in feature_weight_dict:
                    temp_weight.append((feature_weight_dict[word],word))
                else:
                    temp_weight.append((0,word))
            remove=sorted(temp_weight,reverse=True)[:20]
            
            for i in range(20):
                
                doc.remove(remove[i][1])
            f.write(' '.join(doc))
            f.write('\n')
    modified_data = './modified_data.txt' 
    assert strategy.check_data(test_data, modified_data)
    return strategy        
            
            
test_data = "test_data.txt"
fool_classifier(test_data)                
        
        
    
    
    
   
        
        
        
    

        
    
    

'''
    ##取出test_data
    with open('test_data.txt','r') as test_file:
         i=0
         for line in test_file:
             if i==0:
                print(line)
                print(line.strip())
                print(line.strip().split(' '))
                i=1
             else:
                 break
         test_data_list=[line.strip().split(' ') for line in test_file]
    ##获取test data vector
    test_vector=get_test_vector(test_data_list,feature_list)
    
   #vetor_test_data1=vectorizer.transform(class_0_dictionary)
    prediction = svm.predict(np.asarray(test_vector))


    
   
   ##calculate the accuracy
    n_1=prediction.tolist().count(1)
    n_0=prediction.tolist().count(0)
    accuracy = n_1/(n_1+n_0)
    #weights = svm.coef_[0].toarray()
   #print(weights)
    print(accuracy)
   #print(weights)
'''     
    

    
    
     
    