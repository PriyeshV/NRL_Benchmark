import time
import scipy.io
import numpy as np
import scipy as sp
import multiprocessing as mp
from argparse import ArgumentParser
import networkx as nx
from sklearn.model_selection import GridSearchCV

import sklearn.linear_model as skl
import sklearn.metrics
from six import iteritems
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn.utils import shuffle as skshuffle

from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score
from argparse import ArgumentParser
from sklearn.metrics import roc_auc_score, make_scorer
from sklearn.metrics import average_precision_score, precision_recall_curve, auc

np.random.seed(10)
def aupr(y,prob_preds):
    precision, recall, thresholds = precision_recall_curve(y, prob_preds)
    aupr = auc(recall, precision)
    return aupr




def compute_roc_aupr_score(train_pos_arr,train_neg_arr,test_pos_arr,test_neg_arr,embedding0,embedding1,n_jobs=28,dot_product=False,edge_representation=None,grid_search=True,percent_edges = -1):
    # Sample fewer percent of edges for very large datasets (Datasets with > 5M edges)
    if(percent_edges != -1):
        ratio = percent_edges/100.0
        train_edge_size = int((ratio)*train_pos_arr.shape[0])
        test_edge_size = int((ratio)*test_pos_arr.shape[0])
        np.random.shuffle(train_pos_arr)
        np.random.shuffle(train_neg_arr)
        np.random.shuffle(test_pos_arr)
        np.random.shuffle(test_neg_arr)
        train_pos_arr = train_pos_arr[:train_edge_size]
        train_neg_arr = train_neg_arr[:train_edge_size]
        test_pos_arr = test_pos_arr[:test_edge_size]
        test_neg_arr = test_neg_arr[:test_edge_size]
    
    train_size = train_neg_arr.shape[0]+train_pos_arr.shape[0]  
    test_size = test_neg_arr.shape[0]+test_pos_arr.shape[0] 


    all_edges = np.concatenate([train_neg_arr,train_pos_arr,test_neg_arr,test_pos_arr],axis=0)

    train_y = [0] * train_neg_arr.shape[0] + [1] * train_pos_arr.shape[0] 
    test_y = [0] * test_neg_arr.shape[0] + [1] * test_pos_arr.shape[0]

    
    if(dot_product):
        all_scores = np.sum(np.multiply(embedding0[all_edges[:,0]],embedding1[all_edges[:,1]]),axis=1)
        train_prob_preds = all_scores[:train_size]
        test_prob_preds = all_scores[train_size:]
        best_c = None
        best_penalty = None

    else:    # Create different edge features

        if(edge_representation=='hadamard'):
          representation = np.multiply(embedding0[all_edges[:,0]],embedding1[all_edges[:,1]])

        elif(edge_representation=='concat'):
          representation = np.concatenate([embedding0[all_edges[:,0]],embedding1[all_edges[:,1]]],axis=1)

        elif(edge_representation=='l2'):
          representation = np.square(embedding0[all_edges[:,0]]-embedding1[all_edges[:,1]])

        train_X = representation[:train_size]
        test_X = representation[train_size:]
        del all_edges
        model=LogisticRegression()

        if grid_search:
            
            parameters = {"penalty":["l2"],"C":[0.01,0.1,1]}      
                   

            model = GridSearchCV(model, param_grid=parameters, cv=2, scoring='roc_auc', n_jobs=n_jobs, verbose=0,pre_dispatch='n_jobs')
        model.fit(train_X, train_y)
        

        train_prob_preds = model.predict_proba(train_X)[:,1]
        test_prob_preds = model.predict_proba(test_X)[:,1]
        del model
        
    train_auc=roc_auc_score(train_y, train_prob_preds)
    test_auc=roc_auc_score(test_y, test_prob_preds)
    train_aupr = aupr(train_y,train_prob_preds)
    test_aupr = aupr(test_y,test_prob_preds)
    

    return train_auc,test_auc,train_aupr,test_aupr

