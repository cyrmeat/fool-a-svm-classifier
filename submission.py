import helper
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer    

def get_dict(data):
    l = []
    for line in data:
        d = {}
        for w in line:
            if w not in d:
                d[w] = 1
            else:
                d[w] += 1
        l.append(d)
    return l
   
def fool_classifier(test_data): ## Please do not change the function defination...
    ## Read the test data file, i.e., 'test_data.txt' from Present Working Directory...
    with open(test_data,'r') as test_file:
        # for DictVectorizer
        _test_data=[line.strip().split(' ') for line in test_file]
        # for CountVectorizer
        #test_data=[line.strip() for line in test_file]
    
    ## You are supposed to use pre-defined class: 'strategy()' in the file `helper.py` for model training (if any),
    #  and modifications limit checking
    strategy_instance=helper.strategy() 
    kernels = ['linear','polynomial', 'rbf', 'sigmoid']
    parameters={'gamma':'auto', 'C': 100.0, 'kernel':'sigmoid', 'degree':10.0, 'coef0':0.2}
   
    # for DictVectorizer
    y_0 = len(strategy_instance.class0)
    y_1 = len(strategy_instance.class1)
    text = get_dict(strategy_instance.class0) + get_dict(strategy_instance.class1)
    vectorizer = DictVectorizer()
    x_train = vectorizer.fit_transform(text)
    nnn = vectorizer.get_feature_names()
    print(len(nnn))
    y_train = [0]*y_0
    y_train.extend([1]*y_1)
    
    model = strategy_instance.train_svm(parameters, x_train, y_train)
    
    test_dict = get_dict(_test_data)
    test = vectorizer.transform(test_dict)
    prediction = model.predict(test)
    
    n_1 = 0
    n_0 = 0
    for i in prediction:
        if i == 1:
            n_1 += 1
        elif i == 0:
            n_0 += 1
        else:
            print(i)
    
    '''
    # for CountVectorizer
    vectorizer = CountVectorizer(lowercase=False)
    y_0 = len(strategy_instance.class0)
    y_1 = len(strategy_instance.class1)

    text = strategy_instance.class0 + strategy_instance.class1
    
    x_train = vectorizer.fit_transform(text)
    nnn = vectorizer.get_feature_names()
    print(len(nnn))
    y_train = [0]*y_0
    y_train.extend([1]*y_1)
    
    model = strategy_instance.train_svm(parameters, x_train, y_train)
    
    test = vectorizer.transform(test_data)
    #test = vectorizer.transform(strategy_instance.class1)
    prediction = model.predict(test)
    n_1 = 0
    n_0 = 0
    for i in prediction:
        if i == 1:
            n_1 += 1
        elif i == 0:
            n_0 += 1
        else:
            print(i)
    '''
    print("Accuracy:")
    print(n_1/(n_1+n_0))
    print(n_1)
    print(n_0)
       

    ##..................................#
    #
    #
    #
    ## Your implementation goes here....#
    #
    #
    #
    ##..................................#
    
    
    ## Write out the modified file, i.e., 'modified_data.txt' in Present Working Directory...
    
    
    ## You can check that the modified text is within the modification limits.
    modified_data='./modified_data.txt'
    ## assert to check the Modification Limits...(Modify EXACTLY 20- DISTINCT TOKENS)
    #assert strategy_instance.check_data(test_data, modified_data)
    return strategy_instance ## NOTE: You are required to return the instance of this class.

fool_classifier('test_data.txt')