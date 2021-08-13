#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 17:28:02 2021

@author: dylansmith
"""
import pandas as pd
import numpy as np
import sys
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.layers import LSTM
from matplotlib import pyplot
import tensorflow.keras as keras
from tensorflow.keras import initializers
import random as ra
import tensorflow as tf

from lstm_functions import *
compute_measure=compute_measure; d_add=d_add; d_multiply=d_multiply; d_inner=d_inner; d_combine=d_combine; StratifiedKFold=StratifiedKFold
train_test_split=train_test_split; StandardScaler=StandardScaler; MinMaxScaler=MinMaxScaler; Percentile=Percentile

import warnings
warnings. simplefilter(action='ignore', category=Warning)

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def rl(x):
    return range(len(x))


######toggle -- use base level variables (0 for not)
toggle=0
######toggle -- use flat LSTM (0 for natural LSTM)
flat_toggle=0

np.set_printoptions(suppress=True)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
pd.options.display.float_format = '{:.2f}'.format


sources=['./cleaned_data_3_1_ex_lstm.csv','./Record_start_end_derived.csv']

df= pd.read_csv(sources[0],low_memory = False)
df.index=df['new random ID']; df=df.drop(['new random ID','0.Age'],axis=1)

def intersect(l,l2,outer=None):
    l=list(l);l2=list(l2);I=[]
    for i in rl(l):
        if l[i] in l2:
            I.append(l[i])
    for i in rl(I):l.remove(I[i]);l2.remove(I[i])
    if outer==0:return l
    elif outer==1:return l2
    return I

def transform(df,Len=12):
    #remove static features and create 3D dataset
    temp={}; T=["0","1","2","3","4","5","6"]; l=str(Len); ll=len(l)+2
    for i in rl(df):
        S=df.iloc[i,:]
        SI=S.index
        tem=pd.DataFrame(index=T); st=[]
        for si in rl(S):
            test=SI[si]
            if test[0] in T and "--" not in test:
                st.append(test[2:])
            elif test[-1] in T:
                if l in test[:2]: st.append(test[ll:-2])
        st=list(set(st)); ST={}
        for t in st: ST[t]=[np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan]
        for si in rl(S):
            test=SI[si]
            if test[0] in T and "--" not in test:
                ST[test[2:]][int(test[0])-1]=S[test] #not -1
            elif test[-1] in T:
                if l in test[:2]: ST[test[ll:-2]][int(test[-1])]=S[test]
        for s in ST.keys(): tem[s]=ST[s]
        #tem['Label(t-1)']=tem['Label'].shift(1)
        temp[df.index[i]]=tem.reindex(sorted(tem.columns), axis=1)
    return temp

def r3(x):
    return round(x,3)

#####IMPORTED FROM MACHINELEARNINGMASTERY.COM and EDITED
def series_to_supervised(data, n_in=1, n_out=1, dropnan=False):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = pd.DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = pd.concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	s=str(list(data.columns).index("Label")+1)
	for c in agg.columns:
		if c[-2:]=="t)" and "var"+s+"(t)" not in c: agg=agg.drop(c,axis=1)
	return agg
###############################################
    
def s(*args):
    STR=""
    for arg in args:
        STR+="_"+str(arg)
    return STR

def d2(x,steps=1):
    if len(x.shape)==3:x=x.reshape((x.shape[0], x.shape[2]*x.shape[1]))
    return x

def d3(x,steps=1):
    if len(x.shape)<3:x=x.reshape((x.shape[0], steps, int(x.shape[1]/steps)))
    return x

def merge(x,y):
    #merge array x with vector y
    return np.concatenate([x,y.reshape([len(y),1])],axis=1)


def kerLSTM(train_X,train_y,pars={},fit=True,pred=False,steps=1):
    #pars are learning rate, optimizer, kreg, neurons, epochs
    #reshape to 3D
    train_X=d3(train_X,steps)
    #if pars==None: print('lr','opt','kreg','neurons','epochs'); return None
    
    base_values = [.01, 'adam', .0001, 75, 50]; c=-1
    for par in ['lr','opt','kreg','neurons','epochs']:
        c+=1
        if par not in pars:pars[par]=base_values[c]
    if pars['opt']=='adam': Optimizer=keras.optimizers.Adam(learning_rate=pars['lr'])
    elif pars['opt']=='sgd': Optimizer=keras.optimizers.SGD(learning_rate=pars['lr'])
    elif pars['opt']=='rms': Optimizer=keras.optimizers.RMSprop(learning_rate=pars['lr'])
    model = Sequential()
    kReg=keras.regularizers.l1_l2(l1=pars['kreg'], l2=pars['kreg'])
    if type(pars['neurons'])!=int:
        for n in range(len(pars['neurons'])):
            if n!=len(pars['neurons'])-1:ret=True; c+=1
            else: ret=False
            N=pars['neurons'][n]
            model.add(LSTM(N, kernel_regularizer=kReg, return_sequences=ret, input_shape=(train_X.shape[1], train_X.shape[2]))) 
    else:
        N=pars['neurons']
        model.add(LSTM(N, kernel_regularizer=kReg, input_shape=(train_X.shape[1], train_X.shape[2])))
    model.add(Dense(2, kernel_regularizer=kReg, activation='softmax'))
    model.compile(loss='sparse_categorical_crossentropy', optimizer=Optimizer) #activation soft max --> classification
    # fit network
    if fit==True:model.fit(train_X, train_y, epochs=pars['epochs'], batch_size=72, verbose=0, shuffle=False)
    if pred==True:return model,pars,model.predict_on_batch(train_X,steps)
    return model, pars



def kerGrid(x,y,parDict,metric='diagI',folds=9,stat=9,verbose=1,threshold=.5,steps=1):
    ##x and y are training features and target value matrices
    ##takes input dictionary of lists of potential values for parameters
    stats=['f1', 'f0', 'acc1', 'acc0', 'acc', 'auc', 'ppr', 'npr', 'diagI']
    parDict=d_inner(parDict) #convert any singletons to lists
    parList=[]; parComb=[[]]; names=[]; results=[]; records=[]
    for par in parDict:parList.append(parDict[par]); names.append(par)
    for n in range(len(names)):
        temp=[]
        for i in range(len(parComb)):
            for j in range(len(parList[n])):
                temp.append(parComb[i]+[parList[n][j]])
        parComb=temp
    for pl in parComb:
        keras.backend.clear_session()
        if verbose!=0:print('starting search for combination:',pl)
        pars=dict(zip(names,pl))
        ## test stats of model through intra parameter
        skf=StratifiedKFold(n_splits=folds,shuffle=True, random_state=42); valid_record=0
        if type(threshold)==list: valid_record=dict(zip(threshold,[0,]*len(threshold)))
        w_train,w_test,z_train,z_test,long_pred=[],[],[],[],{}
        for train_index, test_index in skf.split(x, y):
            w_train.append([]),w_test.append([]),z_train.append([]),z_test.append([])
            w_train[-1], w_test[-1] = x[train_index], x[test_index]
            z_train[-1], z_test[-1] = y[train_index], y[test_index]
        for i in range(folds):
            model,tpars=kerLSTM(w_train[i],z_train[i],pars,steps=steps)
            pred=model.predict_on_batch(d3(w_test[i],steps))
            if i==0:long_pred['pred']=pred; long_pred['test']=z_test[i]
            else:
                long_pred['pred']=np.concatenate([long_pred['pred'],pred])
                long_pred['test']=np.concatenate([long_pred['test'],z_test[i]])
        if type(threshold)==list:
            for thresh in threshold:
                valid_rec=compute_measure(long_pred['test'],long_pred['pred'],threshold=thresh)
                if thresh==.5:results.append(valid_rec[stats.index(metric)])
                valid_record[thresh]=valid_rec[:stat]
            if .5 not in threshold: results.append(valid_rec[stats.index(metric)])
        else:
            valid_rec=compute_measure(long_pred['test'],long_pred['pred'],threshold=threshold)
            results.append(valid_rec[stats.index(metric)])
            valid_record=valid_rec[:stat]
        records.append(valid_record)
        if type(valid_record)==dict:print(valid_record[.5])
        else:print(valid_record)
    M=max(results); ind=results.index(M); record=records[ind]; result=M; pars=dict(zip(names,parComb[ind]))
    print('optimal combination:',pars)
    return pars,record,result

       
def kerClassify(data,y_value=None,scaler=None,pars={}, split=None,
            search=None,metric='diagI', stop=True , folds=10, pct=5,threshold=.5,stat=9,steps=1):
    ## search should be None or a dictionary with list values
    ## if search!=None, pars will be ignored
    ## split should pass in an iterable of Pandas DataFrames [Train,Test]
    ## stop=True is used to avoid running the model once parameters are found
    ## split should pass in an iterable of Pandas DataFrames [Train,Test], used for preset splits

    try: y_value=data.columns[-1]
    except: data=pd.DataFrame(data); y_value=data.columns[-1]
    y=data[y_value].values
    x=data.drop([y_value],axis=1).values
    
    if split==None:
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=1/folds)
    else: 
        x_train=split[0].drop([y_value],axis=1).values; x_test=split[1].drop([y_value],axis=1).values
        y_train=split[0][y_value].values; y_test=split[1][y_value].values
    
    if stop==True and search!=None: x_train, x_test, y_train, y_test = x,x,y,y; folds=folds+1 # ignore outer loop splitting
    if scaler=='ss': scaler=StandardScaler() ; scaler.fit(x_train);  x_train,x_test = scaler.transform(x_train),scaler.transform(x_test)##STANDARD SCALER
    elif scaler=='mm': scaler=MinMaxScaler() ; scaler.fit(x_train);  x_train,x_test = scaler.transform(x_train),scaler.transform(x_test)##MIN-MAX SCALER  
    elif scaler=='pct': x_train,x_test = Percentile(x_train,None,pct),Percentile(x_train,x_test,pct) #PERCENTILE SCALER
    
    model=kerLSTM
    mname='Dense Keras LSTM'
        
    if search==None:
        model,pars=model(x_train,y_train,pars,steps=steps)
    else:
        ## hyper parameters and test stats of model through intra parameters
        pars,valid_record,result=kerGrid(x_train,y_train,d_combine(search,d_inner(pars)),folds=folds-1,metric=metric,threshold=threshold,steps=steps)
        model,pars=model(x_train,y_train,pars,steps=steps)
    ### training accuracy
    if stop==True:
        pred = model.predict_on_batch(d3(x_train,steps))
        if type(threshold)==list:
            train_record=dict(zip(threshold,[0,]*len(threshold)))
            for thresh in threshold:
                train_record[thresh]=compute_measure(y_train,pred,threshold=thresh)[:stat]
        else: train_record=compute_measure(y_train,pred,threshold=threshold)[:stat]
    
        if search!=None:
            return model,pars,train_record,valid_record,pred     #Stop Here For Validation Sets
    
    pred = model.predict_on_batch(d3(x_test,steps))
    if type(threshold)==list:
        test_record=dict(zip(threshold,[0,]*len(threshold)))
        for thresh in threshold:
            test_record[thresh]=compute_measure(y_test,pred,threshold=thresh)[:stat]
    else: test_record=compute_measure(y_test,pred,threshold=threshold)[:stat]
    #MSE = mean_squared_error(y_test,y_pred)
    #print("The",mname,"ROC AUC Score on test set: {:.4f}".format(MSE))
    #train=[x_train,y_train]
    #test=[x_test,y_test,y_pred,y_prob]
    
    return model, pars, test_record #MSE,train,test,pars,model




####################################################################################################################################################################
####################################################################################################################################################################
####################################################################################################################################################################
####################################################################################################################################################################
####################################################################################################################################################################
####################################################################################################################################################################
####################################################################################################################################################################
####################################################################################################################################################################
####################################################################################################################################################################
####################################################################################################################################################################
####################################################################################################################################################################
####################################################################################################################################################################


#Start of Test




Threshold=[.3,.35,.4,.45,.5,.55,.6]
stats=['f1', 'f0', 'acc1', 'acc0', 'acc', 'auc', 'ppr', 'npr', 'diagI']
Base_Pars={'lr':.001,'epochs':50} ####
Search={'opt':['adam'],'kreg':[.0001,.0004],'neurons':[40,75,(75,75)]} #######
#Search={'opt':['adam,'sgd'],'kreg':[.00001,.0001,.0005],'neurons':[40,75,100,200,(75,75)]}
n_lags=[6,5,4,3,2,1]#4,3,2,1]
train_ratios=[.95,.9,.8]; target_lengths=[12,9,6,3]; metric_='diagI'

c=0
for i in rl(n_lags):
    for j in rl(target_lengths):
        c+=1
        print("python lstm_test_search.py {} 0 {} 8 > log{}.txt &".format(i,j,c))

arg=sys.argv
i=1; lags_,len_,tl_,metric_=-2,-2,-2,-2
while i<len(arg):
    if i==1: lags_=int(arg[1]) #index or -1 : lagged years
    if i==2: len_=int(arg[2]) #index or -1 : train ratios
    if i==3: tl_=int(arg[3]) #index or -1 : target months
    if i==4: metric_=arg[4] #index or name: evaluation metric
    i+=1

if lags_==-2:  lags_=   int(input("type, index or -1 : lagged years (n_lags=6,5,4,3,2,1]): "))
if len_==-2:   len_=    int(input("type, index or -1 : train ratios (train_ratios=[.95,.9,.8]): "))
if tl_==-2:    tl_=     int(input("type, index or -1 : target months (target_lengths=[12,9,6,3]): "))
if metric_==-2:metric_=input("type, index/name: evaluation metric ('f1', 'f0', 'acc1', 'acc0', 'acc', 'auc', 'ppr', 'npr', 'diagI'): ")

if lags_!=-1:n_lags=n_lags[lags_:lags_+1]
if len_!=-1: train_ratios=train_ratios[len_:len_+1]
if tl_!=-1:  target_lengths=target_lengths[tl_:tl_+1]
try:   metric_=int(metric_); metric=stats[metric_]
except:metric=metric_; metric_=stats.index(metric)


print(n_lags); print(train_ratios); print(target_lengths); print(metric)

##SETUP BASE VARIABLE TABLE

######AGE
DF=df; temp=pd.DataFrame(index=DF.index)
Record=pd.read_csv(sources[-1])
Record.index=Record['new random ID'];Record=Record[Record.index.isin(DF.index)]
record=period_plus(Record['Date Text_2'].apply(lambda x: x.split("/")),-6)
#record is date used to compute 0.Age (X -6_periods)
temp["0"]=period_plus(DF['DOB'].apply(lambda x: x.split("/")).apply(lambda x:[x[0],x[1],d_ext(x[2])]),0)
temp["1"]=period_plus(DF['DOB'].apply(lambda x: x.split("/")).apply(lambda x:[x[0],x[1],d_ext(x[2])]),-1)
temp["r"]=record
temp["0"]=(temp['0']+temp['r']).apply(lambda x: (x[:3],x[3:])).apply(lambda x:age_comp(x[0],x[1]))
temp["1"]=(temp['1']+temp['r']).apply(lambda x: (x[:3],x[3:])).apply(lambda x:age_comp(x[0],x[1]))-temp['0']
temp["1"]=temp["1"].apply(lambda x:max(.5,x))

c=0
for i in range(DF.shape[1]):
    try: int(DF.columns[i][0]); c=i; break
    except: pass

Static=DF.copy().iloc[:,:c].rename(columns={'DOB':'Age'})
Table=pd.DataFrame()
for per in range(1,7):
    Static["Age"]=list((temp['0']+temp['1'].apply(lambda x: x+(per-1)*.5)).apply(lambda x: int(x//1)))
    Static.index=list(pd.Series(temp.index).apply(lambda x:str(x)+"--"+str(per)))
    Table=pd.concat([Table,Static],axis=0)

##Add Static Variables is toggle==1
if toggle==1:
    basecols=list(Table.columns)
    cols=[]
    for i in range(1,7):
        for b in basecols:
            cols.append(str(i)+"."+b)
    Temp=pd.DataFrame(index=df.index,columns=cols)
    for i in range(1,7):
        cols_temp=cols[0+len(basecols)*(i-1):len(basecols)*(i)]
        Temp.loc[:,cols_temp]=np.array(Table.iloc[:df.shape[0],:])
    for I in Table.index:
        ind,i=I.split("--")
        Temp.loc[int(ind),i+".Age"]=int(Table.loc[I,"Age"])
    df=pd.concat([df,Temp],axis=1)

N=0; kreg=-1
#LEN=int(input("length of prediction period(default 12 months) 3/6/9/12 :"))
for LEN in target_lengths: #,12:
    new_data=transform(df,LEN)
    key=[]
    for i in range(len(new_data[38].columns)):key.append("var"+str(i+1))
    Key=dict(zip(key,list(new_data[38].columns)))
    print(Key)
    K=list(new_data.keys())
    stat_num=len(stats)
    multiply=round(1/(LEN/3)*(len(Key)/53),3) #increase for high feat count//reduce for high tar Len
    Search['kreg']=[.00012*multiply,.0006*multiply]
    print(Search)
    
    Training_Record={}; Testing_Record={}; Validation_Record={}; Par_Record={}
    
    for n_lagged in n_lags:
        #kr_power=1 #power of increase in kreg due to higher # of lags
        if flat_toggle==0:steps=n_lagged; kr_power=n_lagged #1/1.5/2
        elif flat_toggle==1:steps=1; kr_power=n_lagged**.6 ########################
        m2=n_lagged**kr_power
        Search['kreg']=[np.round(.00012*multiply*m2,8),np.round(.0006*multiply*m2,8)]
        print(f"\n\nStarting n_lag: {n_lagged}, with shape: {steps}, with kr_power: {kr_power}, with KREG: {Search['kreg']}\n\n")
        Training_Record[n_lagged]={}; Testing_Record[n_lagged]={};Validation_Record[n_lagged]={}; Par_Record[n_lagged]={}
        #n_lagged=1
        
        NewData={}
        for k in K:
            new=series_to_supervised(new_data[k],n_lagged)
            NewData[k]=new
        scaler = MinMaxScaler(feature_range=(0, 1)) 
        
        for n_train_ratio in train_ratios:
            Rmse=0; Training_Rec,Testing_Rec,Validation_Rec=dict(zip(Threshold,[0,]*len(Threshold))),dict(zip(Threshold,[0,]*len(Threshold))),dict(zip(Threshold,[0,]*len(Threshold)))
            fold_number=round(1/(1-n_train_ratio)); Choice=d_combine(Base_Pars,Search)
            for i in Choice.keys(): Choice[i]=[] #set up dictionary of parameters
            print("\n>>Testing for training % of data at",n_train_ratio)
            KT=[]; Kt=0
            for fold_index in range(fold_number):
                keras.backend.clear_session() #reset keras memory
                
                TESTING_REC,TRAINING_REC,VALIDATION_REC=dict(zip(Threshold,[0,]*len(Threshold))),dict(zip(Threshold,[0,]*len(Threshold))),dict(zip(Threshold,[0,]*len(Threshold)))
                fold_index+=1
                print("\n>init: fold",fold_index)
                #n_train_ratio=.5
                K1=np.array(pd.DataFrame(intersect(K,KT,outer=0)).sample(int(len(K)*n_train_ratio),random_state=42)).T.tolist()[0] #train
                K2=intersect(K,K1,0) #test
                KT+=K2
                Train=pd.DataFrame([]); Test=pd.DataFrame([])
                #res=0;reav=0
                
                for k in K: 
                    add=NewData[k].copy(); add.index=list(pd.Series(add.index).apply(lambda x: str(k)+"--"+x+"_0"))
                    if k in K1: Train=pd.concat([Train,add],axis=0)
                    #temp=NewData[k].dropna(0).iloc[:,-1]
                    #if np.max(temp)==1: res+=1;reav+=np.mean(temp)
                    if k in K2: Test=pd.concat([Test,add],axis=0)
                #reav=reav/res
                Test=Test.dropna(0)
                Train=Train.dropna(0)
                Kt+=len(Test)
                print(list(Train.iloc[:,-1]).count(1),list(Train.iloc[:,-1]).count(0))
                ratio_0=list(Train.iloc[:,-1]).count(1)/list(Train.iloc[:,-1]).count(0)
                #mult=1/ratio_0; resampled=int(round(mult/1))-1
                resampling=list(Train.iloc[:,-1]).count(0)-list(Train.iloc[:,-1]).count(1)
                
                sampling=Train[Train.iloc[:,-1]==1]
                #resample training
                for i in range(resampling):
                    r=ra.randint(0,len(sampling)-1)
                    add=sampling.iloc[r:r+1,:]
                    while add.index[0] in Train.index:
                        base,ind=add.index[0].split("_")
                        new_ind=base+"_"+str(int(ind)+1)
                        add.index=[new_ind]
                    Train=pd.concat([Train,add],axis=0)
                ratio_1=list(Train.iloc[:,-1]).count(1)/list(Train.iloc[:,-1]).count(0)
                print("resampling: \nold ratio-{}\nnew ratio-{}".format(ratio_0,ratio_1))
                #dropna
                #standard scaler
                # split into train and test sets

                train=Train.values
                test=Test.values
                
                # split into input and outputs
                train_X, train_y = train[:, :-1], train[:, -1]
                test_X, test_y = test[:, :-1], test[:, -1]
                scaler.fit(train_X)
                
                train_X=scaler.transform(train_X)
                test_X=scaler.transform(test_X)
                train=merge(train_X,train_y)
                test=merge(test_X,test_y)
                # reshape input to be 3D [samples, timesteps, features]
                #train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
                #test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
                print("train_X shape:",train_X.shape,"train_y shape:", train_y.shape,"test_X shape:", test_X.shape,"test_y shape:", test_y.shape)
                
                
                C=kerClassify(train,'Label',pars=Base_Pars, split=None,search=Search,metric=metric, folds=fold_number-1, threshold=Threshold,stat=stat_num,steps=steps)
                model,PARS,TRAINING_REC,VALIDATION_REC,yp=C
                #model,PARS,TRAINING_REC=C
                
                #
                rmse = np.sqrt(mean_squared_error(train_y, yp[:,-1]))
                print('Train RMSE: %.3f' % rmse)
                yp2=np.round(yp)[:,-1]
                train_acc=TRAINING_REC[.5][2:6]
                print("predict:{} ŷ=0 instances, {} ŷ=1 instances\n actual:{} y=0 instances, {} y=1 instances".format(list(yp2).count(0),
                      list(yp2).count(1),list(train_y).count(0),list(train_y).count(1)))
                a,b,c,d=train_acc
                print('Train A1,A0,A,AUC at TH=.5 : {}, {}, {}, {}'.format(r3(a),r3(b),r3(c),r3(d)))
    
                yhat = model.predict_on_batch(d3(test_X,steps))
                #
                rmse = np.sqrt(mean_squared_error(test_y, yhat[:,-1]))
                print('Test RMSE: %.3f' % rmse)
                yhat2=np.round(yhat)[:,-1]
                for thresh in Threshold:
                    TESTING_REC[thresh]=compute_measure(test_y,yhat,thresh)[:stat_num]
                test_acc=TESTING_REC[.5][2:6]
                print("predict:{} ŷ=0 instances, {} ŷ=1 instances\n actual:{} y=0 instances, {} y=1 instances".format(list(yhat2).count(0),
                      list(yhat2).count(1),list(test_y).count(0),list(test_y).count(1)))
                a,b,c,d=test_acc
                print('Test A1,A0,A,AUC at TH=.5 : {}, {}, {}, {}'.format(r3(a),r3(b),r3(c),r3(d)))
                Rmse+=rmse
                Testing_Rec=d_add(Testing_Rec,TESTING_REC)
                Training_Rec=d_add(Training_Rec,TRAINING_REC)
                Validation_Rec=d_add(Validation_Rec,VALIDATION_REC)
                Choice=d_inner(Choice,PARS)
                #Validation_Rec=d_add(Validation_Rec,d_multiply(VALIDATION_REC,1/fold_number))
            
            Choice=d_top(Choice)
            fold_number=fold_index+1
            print(">>>Results: {} Lagged Variables, {} Folds:\n\n".format(n_lagged,fold_number))
            Training_Rec=d_multiply(Training_Rec,1/fold_number)
            Testing_Rec=d_multiply(Testing_Rec,1/fold_number)
            Validation_Rec=d_multiply(Validation_Rec,1/fold_number)
            Rmse=Rmse/fold_number
            print('Test RMSE: %.3f' % Rmse)
            train_acc=Training_Rec[.5][2:6]; test_acc=Testing_Rec[.5][2:6]
            a,b,c,d=train_acc; e,f,g,h=test_acc
            print('Train A1,A0,A,AUC at TH=.5 : {}, {}, {}, {}'.format(r3(a),r3(b),r3(c),r3(d)))
            print('Test A1,A0,A,AUC at TH=.5 : {}, {}, {}, {}'.format(r3(e),r3(f),r3(g),r3(h)))
            
            ter=1-n_train_ratio
            Training_Record[n_lagged][ter]=Training_Rec; Testing_Record[n_lagged][ter]=Testing_Rec
            Validation_Record[n_lagged][ter]=Validation_Rec; Par_Record[n_lagged][ter]=Choice
            ###resample only class 1 time-instances
            ###implement other measures besides accuracy
    
    for lag in n_lags:      
        print("\n\n")
        print(">>>Results: {} Lagged Variables:\n\n".format(lag))
        for t in train_ratios:
            ter=1-t
            print("\n>Results: {} Folds:".format(round(1/ter)))
            Training_Rec=Training_Record[lag][ter]; Testing_Rec=Testing_Record[lag][ter]
            
            train_acc=Training_Rec[.5][2:6]; test_acc=Testing_Rec[.5][2:6]
            a,b,c,d=train_acc; e,f,g,h=test_acc
            print('Train A1,A0,A,AUC at TH=.5 : {}, {}, {}, {}'.format(r3(a),r3(b),r3(c),r3(d)))
            print('Test A1,A0,A,AUC at TH=.5  : {}, {}, {}, {}'.format(r3(e),r3(f),r3(g),r3(h)))    
    T=0;TT=0; P=[]
    for lag in n_lags:      
        print("\n\n")
        print(">>>Results: {} Lagged Variables:\n\n".format(lag))
        for t in train_ratios:
            ter=1-t
            string=str(d_combine(Par_Record[lag][ter],{}))+"from Search :"+str(Search)
            Training_Rec=Training_Record[lag][ter]; Testing_Rec=Testing_Record[lag][ter]
            if type(T)==int:
                T=pd.DataFrame(Testing_Rec,index=stats).T
                TT=pd.DataFrame(Training_Rec,index=stats).T
                P.append(string)
            else:
                T=pd.concat([pd.DataFrame(Testing_Rec,index=stats).T,T],axis=0)
                TT=pd.concat([pd.DataFrame(Training_Rec,index=stats).T,TT],axis=0)
                P=[string]+P
    TT=pd.concat([TT,T],axis=1)  ###reverse order
    N=np.zeros([TT.shape[0]+len(n_lags),TT.shape[1]])
    N[:-len(n_lags),:]=TT; N=pd.DataFrame(N)
    N.iloc[-len(n_lags):,0]=P; TT=N
    import datetime as dt
    prefix="output/"
    today=str(dt.datetime.today()).split(" ")[0][5:]+".csv"
    exp=TT.to_csv("project_test/output/lstm_results_opt_"+s(flat_toggle,toggle,LEN,lags_,metric,today))