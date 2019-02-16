#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 27 14:48:30 2018

@author: zhangjinlong
"""

from sklearn import svm
from sklearn.model_selection import GridSearchCV
import helper
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from random import randint

def fool_classifier(test_data):
    ##get the class1 and class0
    strategy = helper.strategy()
    class0=strategy.class0
    class1=strategy.class1    
    total_train=class0+class1
    
    ##create the total_train tfidf vector
    vectorizer = TfidfVectorizer(max_features=5720, use_idf= True, norm='l2',analyzer= 'word',token_pattern ='[^\s]+' )
    total_train_vector = vectorizer.fit_transform([' '.join(line) for line in total_train ])
    
    ##svm training
#    parameters ={'gamma':'auto','C':1.0,'kernel':'linear','degree':0,'coef0':0 }
    #parameters ={'C':[1,3,5,10,15,20],'kernel':['linear']}
    parameters ={'C':[1],'kernel':['linear']}
    x_train=total_train_vector.toarray()
    y_train=[0 for _ in range(len(class0))] + [1 for _ in range(len(class1))]
#    svm=strategy.train_svm(parameters, x_train, y_train)
    svc = svm.SVC()
    clf = GridSearchCV(svc, parameters)
    clf.fit(x_train, y_train)
    
#    print(clf.cv_results_)
#    print(clf.best_estimator_)
    
    best = clf.best_estimator_
    #svc = svm.SVC(best)
    parameters ={'gamma':'auto','C':best.C,'kernel':'linear','degree':0,'coef0':0 }
    model=strategy.train_svm(parameters, x_train, y_train)
    
    ##get feature list
    feature_list=vectorizer.get_feature_names()
    
    ##get the weight dictionary
#    weight = list(svm.coef_)[0]
    weight = list(model.coef_)[0]
    word_weight = dict.fromkeys(feature_list, 0)
    word_index=0
    for word in feature_list:
        word_weight[word]=weight[word_index]
        word_index+=1

    #print(weight0[:10]) 

    ##get the (feature,weight) list
    #feature_weight_list=[]
    #for word,weight in word_weight.items():
    #    feature_weight_list.append((weight,word))
    #weight_class1=sorted(feature_weight_list, reverse=True)
    #weight_class0 = sorted(feature_weight_list, reverse=False)
    #print(weight_class1[:20])
    #print(weight_class0[:20])
    
    ##get the remove ratio
#    len_0 = len(class0)
#    len_1 = len(class1)
#    if len_0 > len_1:
#        ratio=len_0/(len_0+len_1)
#        num_remove=math.ceil((20*ratio))
#        num_add=20-num_remove
#    if len_0 <= len_1:
#        num_remove=20
#        num_add=0
    #num_remove= 20
    #num_add= 0

    ##store the test data 
    test_data_set=[]
    with open(test_data,'r') as test:
        for lines in test:
            test_data_set.append(list(set(lines.strip().split(' '))))
    ## create the modified_data.txt
    with open("modified_data.txt","w") as f:
        ##list the weight for each doc
        weight0 = sorted(word_weight.items(),key=lambda item:item[1])
        #print(weight0[:20])
        n=0
        for line in test_data_set:
            temp_weight=[]
            doc=line.copy()
            ## get the weight lsit of the doc 
            for word in doc:
                if word in word_weight:
                    temp_weight.append((word_weight[word],word))
                else:
                    temp_weight.append((0,word))
                    
            ## get the add list and the remove list
            add=[]
            remove=[]
            
            ##sort the doc weight descending(weight-word) and class 0 weight(word-weight) 
            doc_weight=sorted(temp_weight,reverse=True)
            if n==0:
               print(doc_weight[:20])
               n+=1
            weight0 = sorted(word_weight.items(),key=lambda item:item[1])

            
            ##create the add and remove list
            modified_times=20
            index_doc=0
            index_weight0=0
            while modified_times>0:
                #print(modified_times)
                if doc_weight[index_doc][0]>0:
                    if abs(doc_weight[index_doc][0])>=abs(weight0[index_weight0][1]):
                        remove.append(doc_weight[index_doc][1])
                        index_doc+=1
                    else:
                    #if abs(doc_weight[index_doc][0])<abs(weight0[index_weight0][1]):
                        weight=weight0[index_weight0][1]
                        word = weight0[index_weight0][0]
                        pair=(weight,word)
                        while pair in doc_weight:
                            #print(1)
                            index_weight0+=1
                            weight=weight0[index_weight0][1]
                            word = weight0[index_weight0][0]
                            pair=(weight,word)
                        add.append(weight0[index_weight0][0])
                        index_weight0+=1
                else:
                #if doc_weight[index_doc][0]<=0:
                    weight=weight0[index_weight0][1]
                    word = weight0[index_weight0][0]
                    pair=(weight,word)
                    while pair in doc_weight:
                        #print(1)
                        index_weight0+=1
                        weight=weight0[index_weight0][1]
                        word = weight0[index_weight0][0]
                        pair=(weight,word)
                    add.append(pair[1])
                    index_weight0+=1
                modified_times-=1
            if n==1:
               
               n+=1
               print(len(add))
               print(len(remove))
            
                             
            
            
#            count=0
#            for word_weigh_pair in weight0:
#                if word_weigh_pair[0] not in doc:
#                    count+=1
#                    add.append(word_weigh_pair[0])
#                if count == num_add:                    
#                    break
##            print(add)
#                
##            print(sorted(temp_weight,reverse=True)[21:30])       
#            remove=sorted(temp_weight,reverse=True)[:num_remove]
            
            
            for i in range(len(remove)):                
                doc.remove(remove[i])
            for j in range(len(add)):
                if add[j] in doc:
                    print('wrong')
                doc.append(add[j])
            f.write(' '.join(doc))
            f.write('\n')
    modified_data = './modified_data.txt' 
    assert strategy.check_data(test_data, modified_data)

    
    return strategy              
    


fool_classifier('test_data.txt')

'''    
    with open('modified_data.txt','r') as test_file:
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