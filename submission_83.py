import helper
import numpy as np
def fool_classifier(test_data): ## Please do not change the function defination...
    ## Read the test data file, i.e., 'test_data.txt' from Present Working Directory...
    
    
    ## You are supposed to use pre-defined class: 'strategy()' in the file `helper.py` for model training (if any),
    #  and modifications limit checking
    

    example=helper.strategy() 


    #a=re.sub(r"$\d+\W+|\b\d+\b|\W+\d+$", "", line)
    #r='[’!"#$%&\'\d+\W+()*+,-./:;<=>?@[\\]^_`{|}~]+'

    #words =re.sub(r,' ',line)
    #words =re.split(' ',words)
    #print(words)


    feature_counter = 0
    calss0_counter = 0
    calss1_counter = 0
    dictionary = dict()

    x = list()
    y = list()
    
    for lines in example.class0:
        x.append([0] * 5720)
        y.append(0)
        for token in lines:
            #if re.match("\w+",token) is None:
                #continue
            if token in dictionary.keys():
                x[calss0_counter][dictionary[token]] += 1
            else:
                dictionary[token] = feature_counter
                x[calss0_counter][dictionary[token]] = 1
                feature_counter += 1
        calss0_counter += 1
                
    for lines in example.class1:
        x.append([0] * 5720)
        y.append(1)
        for token in lines:
            #if re.match("\w+",token) is None:
                #continue
            if token in dictionary.keys():
                x[calss0_counter+calss1_counter][dictionary[token]] += 1
            else:
                dictionary[token] = feature_counter
                x[calss0_counter+calss1_counter][dictionary[token]] = 1
                feature_counter += 1
        calss1_counter += 1
    
    ## dictionary[token]记录每一个单词的index         
    #print(words0)

    x_train_tot = np.array(x)
    y_train = np.array(y)


    parameters = {'gamma':"auto","C":20,"kernel":'linear','degree':1,'coef0':500}
    clf = example.train_svm(parameters, x_train_tot, y_train)

    tokens = []
    with open(test_data,'r') as test:
        for lines in test:
            tokens.append(list(set(lines.strip().split(' '))))

    modified_data = './modified_data.txt'
    
    weights = clf.coef_[0].copy()
    
    
    with open(modified_data,'w') as modified_data_txt:
        for lines in tokens:
            weight = list()
            for token in lines:
                if token in dictionary.keys():
                    weight.append(weights[dictionary[token]])
                else:
                    weight.append(0)

            weight.sort(reverse=True)
            print(weight)
            threshold = weight[21]
            modified_counter = 0
            modified_string = ""

            i = 0
            for token in lines:
                if token not in dictionary.keys() or weights[dictionary[token]] < threshold:
                    modified_string += token
                    modified_string += " "
                else:
                    modified_counter += 1
                    if modified_counter >= 20:
                        i+=1
                        break
                i += 1

            for token in lines[i:]:
                modified_string += token
                modified_string += " "

            while modified_counter<20:
                modified_counter+=1
            
            print(modified_string,file = modified_data_txt)
    


    assert example.check_data(test_data, modified_data)
    return example
    #return clf