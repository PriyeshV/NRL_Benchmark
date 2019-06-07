import numpy
import numpy as np
import scipy as sp
import sklearn.linear_model as skl
import sklearn.metrics
from link_prediction_helper import *
from six import iteritems
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn.utils import shuffle as skshuffle
import scipy.io
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score
from argparse import ArgumentParser
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from gensim.models import Word2Vec , KeyedVectors
from sklearn.preprocessing import normalize
from scipy import stats
import json
import os
import pandas
import warnings
from sklearn.exceptions import DataConversionWarning
from sklearn.exceptions import UndefinedMetricWarning
'''


input graph is a .mat file, with following components:
|-> 'network': Full graph(csc format)
|-> 'train_network': Train graph(csc format)
|-> 'test_network': Test graph(csc format)
|-> 'positive_train_edges': The positive training samples for external classifier and AUC computation
|-> 'positive_test_edges': The positive test samples for external classifier and AUC computation
|-> 'negative_train_edges': The negative train samples for external classifier and AUC computation
|-> 'directed': Indicates of graph is directed or not
|-> 'negative_test_edges'


Example Usage for  ppi 5 folds
To generate the required split, see create_datasets_v1_0.py

Assuming the splitted files are in the following organization:
|--> ppi_80_20
    |->fold_0.mat
    |->fold_1.mat
    |->fold_2.mat
    |->fold_3.mat
    |->fold_4.mat

Assuming the 1st embeddings are saved in the following format:
|--> ppi_80_20_embeddings_u
    |->ppi_embedding_0.npy
    |->ppi_embedding_1.npy
    |->ppi_embedding_2.npy
    |->ppi_embedding_3.npy
    |->ppi_embedding_4.npy

Assuming the 2nd embeddings are saved in the following format:
|--> ppi_80_20_embeddings_v
    |->ppi_embedding_0.npy
    |->ppi_embedding_1.npy
    |->ppi_embedding_2.npy
    |->ppi_embedding_3.npy
    |->ppi_embedding_4.npy

Following is an example usage with 2 embedding:
python link_prediction.py --input_graph_dir ppi_80_20 --file_name fold --dataset ppi --output_file_name results.csv --input_embedding0_dir ppi_80_20_embedding_u --embedding0_file_name ppi_embedding \
--input_embedding1_dir ppi_80_20_v  --embedding1_file_name ppi_embedding --num_folds 5 --emb_size 128 --algorithm deepwalk --embedding_params '{"num_walks" :40, "walk_length" : 80,"window" : 10}' 

Following is an example usage with 1 embedding:
python link_prediction.py --input_graph_dir ppi_80_20 --file_name fold --dataset ppi --output_file_name results.csv --input_embedding0_dir ppi_80_20_embedding_u --embedding0_file_name ppi_embedding \
--share_embeddings --num_folds 5 --emb_size 128 --algorithm deepwalk --embedding_params '{"num_walks" :40, "walk_length" : 80,"window" : 10}'   
'''

def arg_as_list(s):                                                         
    v = ast.literal_eval(s)                                                 
    if type(v) is not list:                                                 
        raise argparse.ArgumentTypeError("Argument \"%s\" is not a list" % (s))
    return v

warnings.filterwarnings(action='ignore', category=UserWarning)
warnings.filterwarnings(action='ignore', category=DataConversionWarning)
warnings.filterwarnings(action='ignore', category=UndefinedMetricWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)



def argument_parse():
    parser = ArgumentParser()
    parser.add_argument('--input_graph_dir', type=str,help='The directory containing different folds')
    parser.add_argument('--file_name', type=str,help='The name of fold graph. Name of the file will be appended with the fold_id',default="fold")

    parser.add_argument("--dataset", required=True, help='Name of your dataset, used for output', default = "")
    
    parser.add_argument('--output_file_name', type=str,help='Name of output result file', default="results_link_prediction.csv")

    parser.add_argument('--input_embedding0_dir', type=str, default="example_graphs")
    parser.add_argument('--embedding0_file_name', type=str,help='The name of files in embedding0. Name of the file will be appended with the fold_id',default="embedding0")
    parser.add_argument('--input_embedding1_dir', type=str, required=False,default="example_graphs")
    parser.add_argument('--embedding1_file_name', type=str,help='The name of files in embedding1. Name of the file will be appended with the fold_id',default="embedding1")
    parser.add_argument("--share_embeddings", action="store_true", help='If used, only 1 embedding will be used for evaluation') 

    parser.add_argument("--word2vec_format", action="store_true", help='If used, genisim is used to load the embeddings')

    parser.add_argument('--num_folds', type=int, default=5,help='Number of folds to be considered')
    parser.add_argument('--n_jobs',type=int,default=28,help='Number of jobs to run in parallel for Grid Search LR')
    parser.add_argument('--percent_edges', type=int, default=-1,help='Sample edges for large datasets like Flickr. Default is -1 for all other dataset. For Flickr, use 10')
    parser.add_argument('--emb_size', type=int, default=128,help='Dimension of Embedding')
    parser.add_argument("--algorithm", required=True, help='What model is evaluated')
    parser.add_argument("--no_gridsearch", action="store_true", help='If used, gridsearch is not done')
    parser.add_argument("--dot_product", action="store_true", help='If used, dotproduct is used for link prediction')

    parser.add_argument("--embedding_params", type=json.loads, help='"embedding_params": Dictionary of parameters used for embedding generation (used to print/save results), type:dict')   
    parser.add_argument("--edge_features", default=['hadamard', 'l2', 'concat'], type=arg_as_list,  help='List of edge representations to consider (default is [\'hadamard\', \'l2\', \'concat\']')
    args = parser.parse_args()
    return args




def load_embeddings(embeddings_file, word2vec_format, num_nodes,l2_normalize):
    model = None
    features_matrix = None
    
    if word2vec_format:
        if ".bin" in embeddings_file:
            model = KeyedVectors.load_word2vec_format(embeddings_file, binary=True) 
        else:
            model = KeyedVectors.load_word2vec_format(embeddings_file, binary=False)
        # Map nodes to their features 
        features_matrix = numpy.asarray([model[str(node)] for node in range(num_nodes)])

    else:
        embeddings_file += '.npy'
        features_matrix = numpy.load(embeddings_file)
    
    if(l2_normalize):
        features_matrix = normalize(features_matrix, norm='l2', axis=1)
    
    return features_matrix




def writetofile_fn(results,edge_features,embedding_params,dataset,algorithm,num_folds,emb_size,share_embeddings,l2_normalize,output_file_name,dot_product):
    columns=["Algorithm", "Dataset", "Train %", "Num of Folds", "Embedding Size", "share_embeddings","l2_normalize"]

    if(not dot_product):
        for i in edge_features:
            columns += [str.upper(i)+'_ROC_AVG',str.upper(i)+'_ROC_ERR',str.upper(i)+'_AUPR_AVG',str.upper(i)+'_AUPR_ERR']
    else:
        mtd = 'dot_product'
        columns +=  [str.upper(mtd)+'_ROC_AVG',str.upper(mtd)+'_ROC_ERR',str.upper(mtd)+'_AUPR_AVG',str.upper(mtd)+'_AUPR_ERR']       
    
    if embedding_params != None:
        columns = columns + list(embedding_params.keys())
    results_df = pandas.DataFrame(columns=columns)
    temp = {
                    "Dataset": dataset,
                    "Train %": 80,
                    "Algorithm": algorithm,
                    "Num of Folds": num_folds,
                    "Embedding Size":   emb_size,
                    "share_embeddings":share_embeddings,
                    "l2_normalize":l2_normalize

                    }
    if(dot_product):
        mtd = 'dot_product'
        temp[str.upper(mtd)+'_ROC_AVG'] = results['avg_roc_'+mtd]
        temp[str.upper(mtd)+'_ROC_ERR'] = results['std_roc_'+mtd]
        
        temp[str.upper(mtd)+'_AUPR_AVG'] = results['avg_aupr_'+mtd]
        temp[str.upper(mtd)+'_AUPR_ERR'] = results['std_aupr_'+mtd]
    else:                
        for j,i in enumerate(edge_features):
            temp[str.upper(i)+'_ROC_AVG'] = results['avg_roc_'+i]
            temp[str.upper(i)+'_ROC_ERR'] = results['std_roc_'+i]
            
            temp[str.upper(i)+'_AUPR_AVG'] = results['avg_aupr_'+i]
            temp[str.upper(i)+'_AUPR_ERR'] = results['std_aupr_'+i]

    if embedding_params != None:
        temp.update(embedding_params)
    results_df = results_df.append(temp, ignore_index=True) 
    if(args.dot_product):
        output_file_name = output_file_name + '_dot_product.csv'
        
    with open(output_file_name,'a') as file:
        results_df.to_csv(file,index=False,header=file.tell()==0)




def link_prediction(input_graph_dir,file_name,dataset,writetofile,output_file_name,input_embedding0_dir,embedding0_file_name,input_embedding1_dir,embedding1_file_name,\
    share_embeddings,word2vec_format,l2_normalize,num_folds,emb_size,algorithm,no_gridsearch,embedding_params,edge_features,percent_edges,n_jobs,dot_product):
    
    all_roc_list = []
    all_aupr_list = []
    if(dot_product):
        
        all_roc_list.append([])
        all_aupr_list.append([])
    else:
    
        for i in range(len(edge_features)):
            
            all_roc_list.append([])
            all_aupr_list.append([])


    # Iterate over each fold and then average the results    
    for fold_id in range(num_folds):

        input_graph_file_name = file_name+"_"+str(fold_id)
        input_graph_path = os.path.join(input_graph_dir,input_graph_file_name)
        data_dict = sp.io.loadmat(input_graph_path)
        train_adj = data_dict['train_network']
        num_nodes = train_adj.shape[0]
        test_adj = data_dict['test_network']
        adj = data_dict['network']

        embedding0_file_name_fold = embedding0_file_name+"_"+str(fold_id)
        embedding0_path = os.path.join(input_embedding0_dir,embedding0_file_name_fold)
        embedding1_file_name_fold = embedding1_file_name+"_"+str(fold_id)
        embedding1_path = os.path.join(input_embedding1_dir,embedding1_file_name_fold)

        
        u = load_embeddings(embedding0_path,word2vec_format,num_nodes,l2_normalize)
        if(share_embeddings):
            v = u
        else:
            v = load_embeddings(embedding1_path,word2vec_format,num_nodes,l2_normalize)



        pos_tr = data_dict['positive_train_edges'] 
        pos_te = data_dict['positive_test_edges']
        neg_tr = data_dict['negative_train_edges'] 
        directed = data_dict['directed']
        
        neg_te = data_dict['negative_test_edges']   


        
        if(dot_product):
            train_roc,test_roc,train_aupr,test_aupr = compute_roc_aupr_score(pos_tr,neg_tr,pos_te,neg_te,u,v,n_jobs,dot_product,None,not no_gridsearch,percent_edges)
            all_roc_list[0].append(test_roc)
            all_aupr_list[0].append(test_aupr)
        else:
            for counter,feature in enumerate(edge_features):
            
                train_roc,test_roc,train_aupr,test_aupr = compute_roc_aupr_score(pos_tr,neg_tr,pos_te,neg_te,u,v,n_jobs,dot_product,feature,not no_gridsearch,percent_edges)
                all_roc_list[counter].append(test_roc)
                all_aupr_list[counter].append(test_aupr)

            
            
    #print(all_ap_list)
    dict_results = {}

    if(dot_product):
        avg_roc = np.mean(all_roc_list[0])
        std_roc = stats.sem(all_roc_list[0])
        
        avg_aupr = np.mean(all_aupr_list[0])
        std_aupr = stats.sem(all_aupr_list[0])
        dict_results['avg_roc_'+'dot_product'] = avg_roc
        dict_results['std_roc_'+'dot_product'] = std_roc
        
        dict_results['avg_aupr_'+'dot_product'] = avg_aupr
        dict_results['std_aupr_'+'dot_product'] = std_aupr

    else:    

        for counter,feature in enumerate(edge_features):
            avg_roc = np.mean(all_roc_list[counter])
            std_roc = stats.sem(all_roc_list[counter])
            
            avg_aupr = np.mean(all_aupr_list[counter])
            std_aupr = stats.sem(all_aupr_list[counter])
            dict_results['avg_roc_'+feature] = avg_roc
            dict_results['std_roc_'+feature] = std_roc
            
            dict_results['avg_aupr_'+feature] = avg_aupr
            dict_results['std_aupr_'+feature] = std_aupr

    print(dict_results)
            
    if(writetofile):
        writetofile_fn(dict_results,edge_features,embedding_params,dataset,algorithm,num_folds,emb_size,share_embeddings,l2_normalize,output_file_name,dot_product)


if __name__ == '__main__':
    args = argument_parse()
    # Compute results for both l2 normalized and unnormalized embeddings
    for l2_normalize_bool in [True,False]:
        link_prediction(args.input_graph_dir,args.file_name,args.dataset,True,args.output_file_name,args.input_embedding0_dir,\
            args.embedding0_file_name,args.input_embedding1_dir,args.embedding1_file_name,args.share_embeddings,args.word2vec_format,l2_normalize_bool,\
            args.num_folds,args.emb_size,args.algorithm,args.no_gridsearch,args.embedding_params,args.edge_features,args.percent_edges,args.n_jobs,args.dot_product)





