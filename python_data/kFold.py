import pandas as pd
import numpy as np

from BayesNet import *

def kFold(data,k=10,structure=[0,0,0,0,0,0],verbose=True):
    '''
    Carries out 10-fold CV
    '''
    
    x=np.repeat(list(range(k)),repeats=(len(data)/k))
    data['fold'] = pd.Series(x)
    
    foldSize = len(data)/k
    accuracyList = []
    
    for fold in range(k):
        

        train = data[data['fold']!=fold]
        test = data[data['fold']==fold]

        train.drop('fold',axis = 1, inplace=True)
        test.drop('fold',axis = 1, inplace=True)

        net = BayesNet(4,structure)
        net.initGraph()
        net.compCPT(train)
        errors = 0

        #Testing
        for i in range(len(test)):

            y = test.iloc[i:(i+1)]
            out = net.predict(y)

            if out != test.iloc[i]['Class']:
                errors+=1
        acc = float(foldSize - errors)/foldSize
        accuracyList.append(acc)
        
        if verbose==True:
            print("Fold :%d Accuracy : %f"%(fold,acc))
    
    if verbose ==True:
        print("Overall CV accuracy : %f"%(np.mean(accuracyList)))
    
    return(np.mean(accuracyList))