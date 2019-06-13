# coding: utf-8

import sys
import snap
import json
import time
import numpy
import pandas
import shutil
import os.path
import scipy as sp
import scipy.sparse
import multiprocessing
import pandas as pd
import networkx as nx

from six import iteritems
from scipy.io import loadmat
from collections import defaultdict
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter


def sparse_tocoo(temp_y_labels):
  y_labels = [[] for x in range(temp_y_labels.shape[0])]
  cy =  temp_y_labels.tocoo()
  for i, j in zip(cy.row, cy.col):
    y_labels[i].append(j)
  assert sum(len(l) for l in y_labels) == temp_y_labels.nnz
  return y_labels


def sparse2graph(x):
  G = defaultdict(lambda: set())
  cx = x.tocoo()
  for i,j,v in zip(cx.row, cx.col, cx.data):
    G[i].add(j)
  return {str(k): [str(x) for x in v] for k,v in iteritems(G)}


def load_graph(matfile, adj_matrix_name, label_matrix_name):
    print ("Loading Matfile from: {}".format(matfile))
    mat = loadmat(matfile)
    labels_matrix = mat[label_matrix_name]
    labels_sum = labels_matrix.sum(axis=1)
    indices = numpy.where(labels_sum>0)[0]
    labels_matrix = sp.sparse.csc_matrix(labels_matrix[indices])
    A = mat[adj_matrix_name][indices][:,indices]
    graph = sparse2graph(A)
    labels_count = labels_matrix.shape[1]
    return mat, A, graph, labels_matrix, labels_count, indices


def main():

    parser = ArgumentParser("node_heu",formatter_class=ArgumentDefaultsHelpFormatter,conflict_handler='resolve')

    # Required arguments
    parser.add_argument("--network", type=str, required=True, help='The path and name of the .mat file containing the adjacency matrix and node labels of the input network')
    parser.add_argument("--edgelist", type=str, required=True, help='The path and name of the edgelist file with no weights containing the edgelist of the input network')
    parser.add_argument("--dataset", type=str, required=True, help='The name of your dataset (used for output)')

    # Optional arguments
    parser.add_argument("--adj_matrix_name", default='network', help='The name of the adjacency matrix inside the .mat file')
    parser.add_argument("--label_matrix_name", default='group', help='The name of the labels matrix inside the .mat file')
    args = parser.parse_args()

    print (args)

    mat, A, graph, labels_matrix, labels_count, indices = load_graph(args.network, args.adj_matrix_name, args.label_matrix_name)
    
    s_time = time.time()

    # Load edgelist as undirected graph in SNAP
    G = snap.LoadEdgeList(snap.PUNGraph, args.edgelist)
    print ("Loading graph in SNAP ... {}".format(str(args.edgelist)))

    # Load edgelist for networkx
    G_NETX = nx.read_edgelist(args.edgelist)
    print ("Loading graph in NetworkX .... {}".format(str(args.edgelist)))

    # Get Average Neighbor Degreeh from NetworkX (only time NetworkX is used)
    AvgNeighDe = nx.average_neighbor_degree(G_NETX)

    # Calculate Page Rank
    p_time = time.time()
    PRankH = snap.TIntFltH()
    snap.GetPageRank(G, PRankH)
    print ("Finished in Page rank in {}".format(str(time.time()-p_time)))

    # Calculate Hub and Authrity Scores
    h_time = time.time()
    NIdHubH = snap.TIntFltH()
    NIdAuthH = snap.TIntFltH()
    snap.GetHits(G, NIdHubH, NIdAuthH)
    print ("Finished in Hub and Auth Scores in {}".format(str(time.time()-h_time)))

    count = 0
    node_data = []
    fl_100 = time.time()
    print ("Num of nodes: {}".format(len(PRankH)))
    print ("Num of nodes with labels: {}".format(len(indices)))
    print ("Collecting other features for each node ...")
    for n in G.Nodes():
        nid = n.GetId()
        if nid in indices:
            node_data.append((nid, n.GetInDeg(), PRankH[n.GetId()], snap.GetNodeClustCf(G, nid), NIdHubH[n.GetId()], NIdAuthH[n.GetId()], AvgNeighDe[str(nid)], snap.GetNodeEcc(G, nid)))
            count = count + 1
            if count % 1000 == 0:
                print ("Processed {} nodes".format(str(count)))
                print (time.time() - fl_100)
                fl_100 = time.time()
                nhdf = pd.DataFrame(node_data, columns=('NodeId', 'Degree', 'PageRankScore', 'NodeClustCf', 'HubScore', 'AuthScore', 'AverageNeighborDegree', 'NodeEcc'))
                nhdf.to_csv((args.network.replace(".mat", "") + "_node_heuristic_features.csv"), index=False)
    		print ("File saved at {}".format((args.network.replace(".mat", "") + "_node_heuristic_features.csv")))

    nhdf = pd.DataFrame(node_data, columns=('NodeId', 'Degree', 'PageRankScore', 'NodeClustCf', 'HubScore', 'AuthScore', 'AverageNeighborDegree', 'NodeEcc'))
    nhdf.to_csv((args.network.replace(".mat", "") + "_node_heuristic_features.csv"), index=False)
    print ("File saved at {}".format((args.network.replace(".mat", "") + "_node_heuristic_features.csv")))


    print ("Finished in {}".format(str(time.time()-s_time)))


if __name__ == "__main__":
    main()

