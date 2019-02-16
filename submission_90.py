from sklearn import svm
import helper
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from random import randint
def extract(test_data1):
    strategy = helper.strategy()
    #class0 = strategy.class0
   # class1 = strategy.class1
    class_0 = strategy.class0
    class_1 = strategy.class1
    #---------------------------------------------------------------
    # class_0_a = []
    # class_1_a = []
    # count_0 = 360
    # count_1 = 180
    # while count_0:
    #     count_line = randint(4,8)
    #     t = []
    #     while count_line:
    #         line_index = randint(0,359)
    #         index_data = randint(0, len(class0[line_index]) - 20)
    #         for index in range(index_data, index_data + 20):
    #             t.append(class0[line_index][index])
    #         count_line -= 1
    #
    #     class_0_a.append(t)
    #     count_0 -= 1
    # while count_1:
    #     count_line = randint(4,8)
    #     t = []
    #     while count_line:
    #         line_index = randint(0,179)
    #         index_data = randint(0, len(class1[line_index]) - 20)
    #         for index in range(index_data,index_data + 20):
    #             t.append(class1[line_index][index])
    #         count_line -=1
    #
    #     class_1_a.append(t)
    #     count_1 -= 1
    #
    # class_0 = class0 + class_0_a
    # class_1 = class1 + class_1_a
    # print(len(class_0))
    # print(len(class_1))
    #--------------------------------------------------------------------------





    with open(test_data1, 'r') as test_file:
        test_data = [line.strip().split(' ') for line in test_file]

    vectorizer = TfidfVectorizer(max_features=5720, use_idf= True, norm='l2',analyzer= 'word',token_pattern ='[^\s]+' )
    #class_0 = class_0[int(len(class_0)/2):]
   # data = vectorizer.fit_transform([' '.join(line) for line in class_0 + class_1 + test_data])
    data = vectorizer.fit_transform([' '.join(line) for line in class_0 + class_1 ])
 #   data1 = vectorizer.fit_transform([' '.join(line) for line in class_1])
    print(data.toarray())
    t_data =vectorizer.fit_transform([' '.join(line) for line in test_data])


    parameters ={'gamma':'auto','C':1,'kernel':'linear','degree':0,'coef0':0 }

    x_train = data.toarray()[:len(class_0+class_1)]
    y_train = [0 for _ in range(len(class_0))] + [1 for _ in range(len(class_1))]
    pre = strategy.train_svm(parameters, x_train, y_train)

    #######################################################
    #print(pre.predict(data.toarray()[len(class_0+class_1):]))
   # total = pre.predict(data.toarray()[len(class_0+class_1):])
   # one = [_ for _ in total if _==1]
   # print(len(one)/len(total))
    ###############################################################
    #final_predict = pre.predict(data.toarray()[len(class_0 + class_1):])
   # final_predict = pre.predict(t_data.toarray()[:])
    list_name=vectorizer.get_feature_names()
    return pre,list_name,test_data,class_1,class_0




def fool(pre,list_name,test_data,class_1,class_0):
    strategy = helper.strategy()

    weight_ = list(pre.coef_)[0]
    #print(pre.coef_)
    word_weight = dict.fromkeys(list_name, 0)
    n_w = 0
    for word in list_name:
        word_weight[word] = weight_[n_w]
        n_w += 1
    weight_list = []
    for word, val in word_weight.items():
        weight_list.append([val, word])
    weight_class1 = sorted(weight_list, reverse=True)
    weight_class0 = sorted(weight_list, reverse=False)
    print(weight_class1)
    print(weight_class0)
    len_class_1 = len(class_1)
    len_class_0 = len(class_0)
    rotio = 0
    de_r = 0
    if len_class_1 < len_class_0:
        r = float(len_class_0/len_class_1)
        if  r == 1:
            rotio = 0.5
        if r > 1 and r< 1.5:
            rotio = 0.7
        if r > 1.5 and r < 2:
            rotio = 0.8
        if r ==  2:
            rotio = 0.85
        if r > 2 and r < 3 :
            rotio = 0.9
        if r > 3:
            rotio = 0.95
        de_r = int(20 * rotio)
    if len_class_1 > len_class_0:
        r = float(len_class_1/len_class_0)
        if r == 1:
            rotio = 0.5
        if r > 1 and r< 1.5:
            rotio = 0.3
        if r > 1.5 and r < 2:
            rotio = 0.2
        if r ==  2:
            rotio = 0.15
        if r > 2 and r < 3 :
            rotio = 0.1
        if r > 3:
            rotio = 0.05
        de_r = int(20 * rotio)


    with open('log.txt','w') as l_f:
        with open('modified_data.txt', 'w') as t_f:
            for t_line in test_data:
                count = de_r
                add_change = []
                delete_change = []

                add_index = 0
                t_delete = []
                t_add = []
                delete = []
                add = []
        #########################################################################
        #---------------delete


                for index in range(len(weight_class1)):
                    if count > 0:
                        if weight_class1[index][1] in t_line:
                            count -= 1
                            delete_change.append(weight_class1[index][1])

                for t_data in t_line:
                    if t_data in delete_change:
                        delete.append(t_data)
                        continue

                    else:
                        t_delete.append(t_data)
                        t_f.write(t_data + ' ')

                #####################################################################################
                # ----------add
                count = 20 - de_r

                for index in range(len(weight_class0)):
                    if count > 0:
                        if weight_class0[index][1] not in t_line:
                            count -= 1
                            add_change.append(weight_class0[index][1])
                            add_index = index
                for a_data in add_change:
                    t_add.append(a_data)
                    add.append(a_data)
                    t_f.write(a_data + ' ')
                total = t_add + t_delete

        ####################################################################################################
        #######----------check add

                T = set(total)
                L = set(t_line)
                if len(set(T - L) | set(L - T)) < 20:
                    for index in range(add_index,len(weight_class0)):
                        T = set(total)
                        if len(set(T - L) | set(L - T)) > 20:
                            break
                        else:
                            total.append(weight_class0[index][1])
                            T = set(total)
                            if len(set(T - L) | set(L - T)) <= 20:
                                add.append(weight_class0[index][1])
                                t_f.write(weight_class0[index][1] + ' ')








               # print('total',total_change)
              #  print('c0',class_0_list)
              #  print('c1',class_1_list)
              #  print(len(set(T - L)))




                t_f.write('\n')
                l_f.write('-----------' + 'delete:' + '\n' + str(set(delete)) + '\n')
                l_f.write('-----------' + 'add:' + '\n' + str(set(add)) + '\n')
                l_f.write('\n')







def fool_classifier(test_data):  ## Please do not change the function defination...
    ## Read the test data file, i.e., 'test_data.txt' from Present Working Directory...


    ## You are supposed to use pre-defined class: 'strategy()' in the file `helper.py` for model training (if any),
    #  and modifications limit checking
    strategy_instance = helper.strategy()
    parameters = {}

    pre, list_name, test_data1,class_1,class_0 = extract(test_data)
    fool(pre, list_name, test_data1,class_1,class_0)

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
   # modified_data = './modified_data.txt'

  #  assert strategy_instance.check_data(test_data, modified_data)
  #  return strategy_instance  ## NOTE: You are required to return the instance of this class.

fool_classifier('test_data.txt')