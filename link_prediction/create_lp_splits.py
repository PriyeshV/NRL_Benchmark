# Acknowledgement:
# Parts of the scripts have been inspired from:
# 1) https://github.com/palash1992/GEM
# 2) https://github.com/google/asymproj_edge_dnn


'''
Example Usage:
To create PPI split (5 folds - 80 - 20% learning-prediction split)

Assumption:
ppi.mat file in /home/usr/Downloads/ppi.mat
Directory for Output : link_prediction_data/ppi_80_20 (assuming these directories are already created)
Name of the folds, we wish to save :'fold'

# For undirected graphs like PPI, Blog, youtube etc.
python create_lp_splits.py --input /home/usr/Downloads/ppi.mat --output_dir link_prediction_data/ppi_80_20/ --num_folds 5 --file_name fold

Following is the structure created:

|_ link_prediction_data
  |_ppi_80_20
    |_ fold_1.mat
    |_ fold_2.mat
    |_ fold_3.mat
    |_ fold_4.mat
    |_ fold_5.mat

# For directed graphs like pubmed:
python create_lp_splits.py --input /home/usr/Downloads/dblp/pubmed.mat --output_dir link_prediction_data/pubmed_80_20/ --num_folds 5 --file_name fold --directed


'''
 
import copy
import random
import networkx as nx
import numpy as np
import os
import sys
import scipy as sp
import scipy.io
import scipy.sparse
from argparse import ArgumentParser

np.random.seed(10)



parser = ArgumentParser()
parser.add_argument('--input',required=True,help='path to .mat file containing the original network', type=str)
parser.add_argument('--train_percent', type=float,help='Split ratio. Between 0 and 1' ,default=0.8)
parser.add_argument('--output_dir', type=str,help='Place where the folds are stored. It is preferrabe to create a new directory for each dataset, split ratio')
parser.add_argument('--num_folds', type=int,help='How many folds to create', default=5)
parser.add_argument('--file_name', type=str,help='Name of the file will be appended with the fold_id',default="fold")
parser.add_argument("--directed", action="store_true", help='Must be set if graph is directed.') 

args = parser.parse_args()


def SampleNegativeEdges(graph, num_edges):
  """Samples edges from compliment of graph."""
  random_negatives = set()
  nodes = list(graph.nodes())
  while len(random_negatives) < num_edges:
    i1 = random.randint(0, len(nodes) - 1)
    i2 = random.randint(0, len(nodes) - 1)
    if i1 == i2:
      continue
    if i1 > i2:
      i1, i2 = i2, i1
    n1 = nodes[i1]
    n2 = nodes[i2]
    if graph.has_edge(n1, n2):
      continue
    random_negatives.add((n1, n2))

  return random_negatives



def MakeDirectedNegatives(positive_edges):
  """ Reverse the positive set and create negative edges"""
  positive_set = set([(u, v) for (u, v) in list(positive_edges)])
  directed_negatives = []
  for (u, v) in positive_set:
    if (v, u) not in positive_set:
      directed_negatives.append((v, u))
  return np.array(directed_negatives, dtype='int32')

 

def splitDiGraphToTrainTest(di_graph, train_ratio, is_undirected=True):
    # Split the graph to train and test graphs
    train_digraph = di_graph.copy()
    test_digraph = di_graph.copy()
    node_num = di_graph.number_of_nodes()
    edges = [e for e in di_graph.edges()]
    
    for (st, ed) in edges:
        if(is_undirected and st >= ed):
            continue
        if(np.random.uniform() <= train_ratio):
            test_digraph.remove_edge(st, ed)
            if(is_undirected):
                test_digraph.remove_edge(ed, st)
        else:
            train_digraph.remove_edge(st, ed)
            if(is_undirected):
                train_digraph.remove_edge(ed, st)
    # If trai graph not connected, then take the largest weakly connected component            
    if not nx.is_connected(train_digraph.to_undirected()):

        train_digraph = max(
            nx.weakly_connected_component_subgraphs(train_digraph),
            key=len
        )
        tdl_nodes = train_digraph.nodes()
        tdl_test_nodes = test_digraph.nodes()
        # Remove nodes not present in train graph, from the test graph
        tdl_nodes = list(set(tdl_nodes) & set(tdl_test_nodes))
        
        nodeListMap = dict(zip(tdl_nodes, range(len(tdl_nodes))))
        
        nx.relabel_nodes(train_digraph, nodeListMap, copy=False)
        test_digraph = test_digraph.subgraph(tdl_nodes)
        test_digraph_1 =  nx.DiGraph(test_digraph)
        nx.relabel_nodes(test_digraph_1, nodeListMap, copy=False)
        return (train_digraph, test_digraph_1)

    return (train_digraph, test_digraph)



def main():
  
  for fold_id in range(args.num_folds):
    print("Fold_{}".format(fold_id))
    file_name = args.file_name+'_'+str(fold_id)+'.mat'

    if args.directed:
      graph = nx.DiGraph()
    else:
      graph = nx.Graph()

    
  
    graph = nx.from_scipy_sparse_matrix(sp.io.loadmat(args.input)['network'], create_using=graph).to_directed()
    graph.remove_edges_from(graph.selfloop_edges())
    train_graph, test_graph = splitDiGraphToTrainTest(graph,train_ratio=args.train_percent,is_undirected=not args.directed)
    # Split graph into test and train
    


    data_dict={}
    data_dict['train_network'] = nx.to_scipy_sparse_matrix(train_graph).astype(np.float64)  
    data_dict['test_network'] = nx.to_scipy_sparse_matrix(test_graph).astype(np.float64)
    data_dict['network'] = ((data_dict['train_network']+data_dict['test_network'])!=0).astype(np.float64)
    graph = nx.from_scipy_sparse_matrix(data_dict['network'])

    if(not args.directed):
      
      train_graph_undirected = train_graph.to_undirected()
      test_graph_undirected = test_graph.to_undirected()
      
      test_edges = test_graph_undirected.edges()
      train_edges = train_graph_undirected.edges() 
    else:
      test_edges = test_graph.edges()
      train_edges = train_graph.edges()

    data_dict['positive_train_edges'] = np.array(train_edges,dtype=np.int32)
    data_dict['positive_test_edges'] = np.array(test_edges,dtype=np.int32)



    random_negatives = list(
        SampleNegativeEdges(graph, len(test_edges) + len(train_edges)))
    random.shuffle(random_negatives)

    test_negatives = random_negatives[:len(test_edges)]
  
    train_eval_negatives = random_negatives[len(test_edges):]

    test_negatives = np.array(test_negatives, dtype='int32')
    test_edges = np.array(test_edges, dtype='int32')
    train_edges = np.array(train_edges, dtype='int32')
    train_eval_negatives = np.array(train_eval_negatives, dtype='int32')

    data_dict['negative_train_edges'] = train_eval_negatives
    
    

    if args.directed:
      # If graph is directed, then add 10% - reversed edges to negative set
      test_edges_size = len(test_negatives)
      normal_negatives_size = int(0.9*test_edges_size)
      reversed_negatives_size = int(0.1*test_edges_size)
      np.random.shuffle(test_negatives)
      normal_negatives = test_negatives[:normal_negatives_size]
      np.random.shuffle(train_edges) 
      reversed_negatives = MakeDirectedNegatives(train_edges[:reversed_negatives_size])
      all_negatives = np.concatenate([reversed_negatives, normal_negatives],
                                             axis=0)
      data_dict['negative_test_edges'] = all_negatives
      data_dict['directed'] = True
      
       
  

    else:
      data_dict['negative_test_edges'] = test_negatives
      data_dict['directed'] = False
    sp.io.savemat(os.path.join(args.output_dir, file_name),data_dict)
  


      

  


if __name__ == '__main__':
  main()

