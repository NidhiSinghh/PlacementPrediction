import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score


from django.shortcuts import render;
def home(request):
    return render(request,'home.html')

def predict(request):
    return render(request,'predict.html')    

def result(request):
    #import the dataset
    # dataset = pd.read_csv('/home/nidhik/Desktop/Placement_Data_Full_Class.csv - Sheet1.csv')
    dataset = pd.read_csv('/home/nidhik/ml_project/Placement_Data_Full_Class.csv - Sheet1.csv')
    # dropping the serial no and salary col
    dataset = dataset.drop('sl_no', axis=1)
    dataset = dataset.drop('salary', axis=1)
    # catgorising col for further labelling
    dataset["gender"] = dataset["gender"].astype('category')
    dataset["ssc_b"] = dataset["ssc_b"].astype('category')
    dataset["hsc_b"] = dataset["hsc_b"].astype('category')
    dataset["degree_t"] = dataset["degree_t"].astype('category')
    dataset["workex"] = dataset["workex"].astype('category')
    dataset["specialisation"] = dataset["specialisation"].astype('category')
    dataset["status"] = dataset["status"].astype('category')
    dataset["hsc_s"] = dataset["hsc_s"].astype('category')
    
    # labelling the columns
    dataset["gender"] = dataset["gender"].cat.codes
    dataset["ssc_b"] = dataset["ssc_b"].cat.codes
    dataset["hsc_b"] = dataset["hsc_b"].cat.codes
    dataset["degree_t"] = dataset["degree_t"].cat.codes
    dataset["workex"] = dataset["workex"].cat.codes
    dataset["specialisation"] = dataset["specialisation"].cat.codes
    dataset["status"] = dataset["status"].cat.codes
    dataset["hsc_s"] = dataset["hsc_s"].cat.codes
    X= dataset.drop("status",axis=1)
    Y=dataset["status"]
    X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2)
    model = LogisticRegression()
    model.fit(X_train,Y_train)
    var2=float(request.GET['n2'])
    var4=float(request.GET['n4'])
    var7=float(request.GET['n7'])
    var10=float(request.GET['n10'])
    var12=float(request.GET['n12'])
    var1=request.GET['n1']
    var3=request.GET['n3']   
    var5=request.GET['n5']
    var6=request.GET['n6']
    var8=request.GET['n8']
    var9=request.GET['n9']
    var11=request.GET['n11']
        # var1=float(request.GET['n1'])
     # var3=float(request.GET['n3'])
        # var5=float(request.GET['n5'])
    # var6=float(request.GET['n6'])
        # var8=float(request.GET['n8'])
    # var9=float(request.GET['n9'])
     # var11=float(request.GET['n11'])
    # var2=request.GET['n2']
     # var4=request.GET['n4']
    # var7=request.GET['n7']
    # var10=request.GET['n10']
    # var12=request.GET['n12']
    
    
    var1=0 if var1=='Female' else 1
    var3=1 if var3=='Others' else 1
    var5=1 if var5=='Others' else 1
    if var6=='Commerce':
        var6=1
    elif var6=='Arts':
        var6=0
    else:
        var6=2    
    if var8=='Sci&Tech':
        var8=2
    elif var8=='Comm&Mgmt':
        var8=0
    else:
        var8=1 
    var9=1 if var9=='Yes' else 0
    var11=1 if var11=='Mkt&HR' else 0


    # pred=model.predict(np.array([var1,var2,var3,var4,var5,var6,var7,var9,var10,var11,var12])).reshape(1,-1))
    pred=model.predict(np.array([var1,var2,var3,var4,var5,var6,var7,var8,var9,var10,var11,var12]).reshape(1,-1))
 
    #pred=round(pred[0])
    if pred==1:
        placement="You might be placed"
    else:
        placement="You may not be placed"
    
    
    # placement="You are"+str(pred)
    return render(request,"predict.html",{"result2":placement})