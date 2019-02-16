import helper
#from sklearn.feature_extraction import FeatureHasher
from sklearn.feature_extraction.text import CountVectorizer
#from sklearn.feature_extraction.text import TfidfTransformer
#from sklearn.metrics import classification_report
from sklearn import cross_validation
#import numpy as np
#from sklearn.feature_extraction.text import TfidfVectorizer
#from sklearn import svm, grid_search, cross_validation
#from sklearn.metrics import classification_report

def hasNumbers(inputString):
     return any(char.isdigit() for char in inputString)
def token_features(token, part_of_speech):
    if token.isdigit():
        yield "numeric"
    else:
        yield "token={}".format(token.lower())
        yield "token,pos={},{}".format(token, part_of_speech)
    if token[0].isupper():
        yield "uppercase_initial"
    if token.isupper():
        yield "all_uppercase"
    yield "pos={}".format(part_of_speech)


def fool_classifier(test_data):
    # Read the test data file, i.e., 'test_data.txt' from Present Working Directory...
    corpus = []
    test_corpue = []
    glo_feature = {}
    x_len = 0
    y_len = 0
    with open("class-0.txt", 'r') as f:
        for line in f.readlines():
            x_len += 1
            corpus.append(line)
            
    with open("class-1.txt", 'r') as f:
        for line in f.readlines():
            y_len += 1
            corpus.append(line)
            
    with open("test_data.txt", 'r') as f:
        for line in f.readlines():
            test_corpue.append(line)

        
#    print(x_len,y_len)
    y = [0]*x_len + [1]*y_len
    # print(corpus[:10])
    ####方法1：vectorizer 0.51
    # raw_X = (token_features(tok, pos_tagger(tok)) for tok in corpus)
    # hasher = FeatureHasher(input_type='string')
    # X_hasher = hasher.transform(raw_X)
    vectorizer = CountVectorizer(analyzer="word",token_pattern="\S+")
    X = vectorizer.fit_transform(corpus).toarray()  
    feature = vectorizer.get_feature_names()
#    print(feature)
#    print(len(feature))
    
    
    
#    vectorizer_0 = CountVectorizer(analyzer="word",token_pattern="\S+")
#    X_0 = vectorizer_0.fit_transform(x_corpus_0).toarray()  
#    feature_0 = vectorizer_0.get_feature_names()
#    y_0 = [0]*x_len
#    print(len(feature_0))
#    print(len(y_0))
#    
#    vectorizer_1 = CountVectorizer(analyzer="word",token_pattern="\S+")
#    X_1 = vectorizer_1.fit_transform(y_corpus_1).toarray()  
#    feature_1 = vectorizer_1.get_feature_names()
#    y_1 = [1]*y_len
#    print(len(feature_1))
#    print(len(y_1))
    # X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.38, random_state=62)
  

    ##方法2：TfidfVectorizer 0.14
    # vectorizer_tfiddf = TfidfVectorizer(binary = False, decode_error = 'ignore', stop_words = 'english') 
    # vectorizer_tfiddf = TfidfVectorizer()
    # X_vectorizer_tfiddf = vectorizer_tfiddf.fit_transform(corpus).toarray()
    # feature_tfidf = vectorizer_tfiddf.get_feature_names()
    # print(len(feature_tfidf))
    # X_train, X_test, y_train, y_test = cross_validation.train_test_split(X_vectorizer_tfiddf, y, test_size=0.0, random_state=62)
  
    #####方法3 TfidfTransformer 0.425
    # transformer = TfidfTransformer()
    # X_tfidf = transformer.fit_transform(X).toarray()
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=78, random_state=76)
    # X_train_2, X_test_2, y_train_2, y_test_2 = cross_validation.train_test_split(X, y, test_size=0.91, random_state=47)


#78 76
#101 76
#52 76
#50 31
#51 76

    ###训练开始
    strategy_instance = helper.strategy()
    parameters = {
        "gamma": "auto",
        "C": 1000,
        "kernel": "linear",
        "degree": 3,
        "coef0": 10.0 
    }
#    out_test_list=[]
    out = strategy_instance.train_svm(parameters,X_train,y_train)
#    print(out)
#    print(out.score(X_test, y_test))
    
    w = out.coef_[0].tolist()
    for i in range(len(feature)):
        glo_feature[feature[i]] = w[i]
    
    
    with open("modified_data.txt","w") as f:
        for i in test_corpue:
            i = i.split(' ')
            i.remove('\n')
            sentence = list(set(i))
            sample_list = []
            for j in sentence:
                if j in glo_feature:
                    sample_list.append((glo_feature[j],j))                             
            delete_word = sorted(sample_list,reverse=True)[:20]
            for k in range(20):
                sentence.remove(delete_word[k][1])
            f.write(' '.join(sentence))
            f.write('\n')

            
#        print(sample_list.sort)
#    top_positive_coefficients = np.argsort(w)[-20:]
#    top_negative_coefficients = np.argsort(w)[:20]

    
#    deleted = []
#    with open("modified_data.txt", 'r') as f:
#        for line in f.readlines():
#            deleted.append(line)
    
#    X_test_0 = vectorizer.transform(corpus[:360]).toarray()
#    X_test_1 = vectorizer.transform(corpus[360:]).toarray()
    
    ###测试结果
#    X_test_data = vectorizer.transform(deleted).toarray()
#    # X_test_data_3 = transformer.transform(X_test_data).toarray()
#    for i in X_test_data:
#        out_test = out.predict([i])
#        out_test_list.append(out_test[0])
#    print(out_test_list)
#    print(sum(out_test_list)/len(out_test_list))
    modified_data = './modified_data.txt'            
    assert strategy_instance.check_data(test_data, modified_data)
    
    # # # NOTE: You are required to return the instance of this class.
    return strategy_instance


test_data = "test_data.txt"
fool_classifier(test_data)