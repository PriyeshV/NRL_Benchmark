'''
Implementation of https://arxiv.org/pdf/1811.12159.pdf - Supervised heuristics prediction

Systematic Biases in Link Prediction: comparing heuristic and graph embedding based methods
Sinha, Aakash and Cazabet, R{\'e}my and Vaudaine, R{\'e}mi
'''

'''
Example usage:

Assuming the splitted files are in the following organization:
|--> pubmed_80_20
    |->fold_0.mat
    |->fold_1.mat
    |->fold_2.mat
    |->fold_3.mat
    |->fold_4.mat

python link_prediction_heuristics.py --input_graph pubmed_80_20/ --dataset pubmed --num_folds 5

For larger datasets (> 5M edges) like Flickr, Youtube - We sample 10% of edges using '--percent edges' flag.

python link_prediction_heuristics.py --input_graph flickr_80_20/ --dataset flickr --num_folds 5 --percent_edges 10   

'''

import time
import scipy.io
import numpy as np
import scipy as sp
import multiprocessing as mp
from argparse import ArgumentParser
import networkx as nx


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
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from sklearn.metrics import average_precision_score
import math
from sklearn.model_selection import GridSearchCV
import os
from scipy import stats
import pandas

parser = ArgumentParser()
parser.add_argument('--input_graph', type=str, help='The directory containing different folds')
parser.add_argument('--dataset', type=str, help="Name of the dataset")
parser.add_argument("--l2_normalize", action="store_true", help='If used, l2 - normalization of embeddings used ')
parser.add_argument("--no_gridsearch", action="store_true", help='If used, gridsearch is not done')
parser.add_argument('--output_file_name', type=str,help='Name of output file', default="results_link_prediction_heuristics.csv")
parser.add_argument('--num_folds', type=int, default=5,help='Number of folds to be considered')
parser.add_argument('--percent_edges', type=int, default=-1,help='Sample edges for large datasets (> 5M edges). Example --percent_edges 10 samples 10\% of edges. Default is -1 for small dataset')
args = parser.parse_args()



def compute_aupr(y,prob_preds):
    precision, recall, thresholds = precision_recall_curve(y, prob_preds)
    aupr = auc(recall, precision)
    return aupr


def writetofile(results):
    columns=["Algorithm","dataset",'avg_roc','std_roc','avg_aupr','std_aupr']
    results_df = pandas.DataFrame(columns=columns)
    results_df = results_df.append(results, ignore_index=True) 
    with open(args.output_file_name, "a") as myfile:
        results_df.to_csv(myfile,index=False)




def compute_auc_aupr_score(train_graph,train_pos_arr,train_neg_arr,test_pos_arr,test_neg_arr,percent_edges=-1):
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
    representation = []
    for pair in all_edges:
        i,j = pair[0],pair[1]
        # Representation for each node pair consisting of 5 different heuristics features
        representation.append(heuristic_features(train_graph,i,j))
       
    representation = np.array(representation)
    if(args.l2_normalize):
        representation = sklearn.preprocessing.normalize(representation)
       


    train_y = [0] * train_neg_arr.shape[0] + [1] * train_pos_arr.shape[0] 
    test_y = [0] * test_neg_arr.shape[0] + [1] * test_pos_arr.shape[0]
    train_X = representation[:train_size]
    test_X = representation[train_size:]
    
    model=LogisticRegression()
    if not args.no_gridsearch:
        parameters = {"penalty":["l2"],"C":[0.01,0.1,1]}     
               

        model = GridSearchCV(model, param_grid=parameters, cv=2, scoring='roc_auc', n_jobs=28,pre_dispatch='n_jobs',verbose=0)
    model.fit(train_X, train_y)
    
    
    
    
    train_prob_preds = model.predict_proba(train_X)[:,1]
    test_prob_preds = model.predict_proba(test_X)[:,1]
    train_auc=roc_auc_score(train_y, train_prob_preds)
    test_auc=roc_auc_score(test_y, test_prob_preds)
    
    train_aupr = compute_aupr(train_y,train_prob_preds)
    test_aupr = compute_aupr(test_y,test_prob_preds)
    

    return test_auc,test_aupr  



def heuristic_features(graph,i,j):
    # A curated list of heuristics for each edge pair - AA, CN, PA, JA, RA
    common_neighbors = len(set(graph.neighbors(i)).intersection(set(graph.neighbors(j))))
    jaccard = len(set(graph.neighbors(i)).intersection(set(graph.neighbors(j))))/float(len(set(graph.neighbors(i)).union(set(graph.neighbors(j)))))

    adamic_adar = sum([1.0/math.log(graph.degree(v)+1) for v in set(graph.neighbors(i)).intersection(set(graph.neighbors(j)))])
    preferential_attachment = graph.degree(i) * graph.degree(j)
    resource_allocation_index = sum([1.0/(graph.degree(v)) for v in set(graph.neighbors(i)).intersection(set(graph.neighbors(j)))])
    
    
    
    features = np.array([common_neighbors,jaccard,adamic_adar,preferential_attachment,resource_allocation_index])
        
    return features


auroc_list = []
aupr_list = []

# Iterate over each fold and report average
for fold_id in range(args.num_folds):
    data_dict = sp.io.loadmat(os.path.join(args.input_graph,"fold_"+str(fold_id)+".mat") )
    train_adj = data_dict['train_network']
    test_adj = data_dict['test_network']
    adj = data_dict['network']

    # Since the heuristics - CN, AA, RA, PA, JA are only defined for undirected datasets, we consider all graphs as undirected for the heuristics method.
    G_train = nx.from_scipy_sparse_matrix(train_adj).to_undirected()
    
    
        

    pos_tr = data_dict['positive_train_edges'] 
    pos_te = data_dict['positive_test_edges']
    neg_tr = data_dict['negative_train_edges'] 
    directed = data_dict['directed']
    
    neg_te = data_dict['negative_test_edges']  
    
    
    
    auroc,aupr = compute_auc_aupr_score(G_train,pos_tr,neg_tr,pos_te,neg_te,args.percent_edges)
        
    auroc_list.append(auroc) 
    aupr_list.append(aupr)
    

avg_roc = np.mean(auroc_list) 

avg_aupr = np.mean(aupr_list)
std_roc = stats.sem(auroc_list) 

std_aupr = stats.sem(aupr_list)

dict_results={}
dict_results['Algorithm'] = 'heuristics'
dict_results['dataset'] = args.dataset
dict_results['avg_roc'] = avg_roc
dict_results['std_roc'] = std_roc

dict_results['avg_aupr'] = avg_aupr
dict_results['std_aupr'] = std_aupr
print(dict_results)

writetofile(dict_results)
