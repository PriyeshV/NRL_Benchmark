import ast
import sys
import json
import time
import torch
import numpy
import pandas
import shutil
import os.path
import sklearn
import logging
import warnings
import scipy as sp
import scipy.sparse
import multiprocessing
import sklearn.metrics
import sklearn.datasets
import sklearn.model_selection

from six import iteritems
from scipy.io import loadmat
from collections import defaultdict
from sklearn.metrics import f1_score
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import normalize
from sklearn.model_selection import GridSearchCV
from sklearn.utils import shuffle as skshuffle
from gensim.models import Word2Vec, KeyedVectors
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer

from sklearn.utils import shuffle
from sklearn.externals import joblib
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import RobustScaler 
from sklearn.preprocessing import StandardScaler

from sklearn.exceptions import DataConversionWarning
from sklearn.exceptions import UndefinedMetricWarning
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

from eigenpro import init_train_Gaussian, line_search, Gaussian, FKR_EigenPro, get_n_class, encoder_

warnings.filterwarnings(action='ignore', category=UserWarning)
warnings.filterwarnings(action='ignore', category=DataConversionWarning)
warnings.filterwarnings(action='ignore', category=UndefinedMetricWarning)


# Set seed:
numpy.random.seed(42)

program = os.path.basename(sys.argv[0])

training_percents_default = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
training_percents_default = [0.1, 0.5, 0.9]
col_names = ""

# Checks if a string is a valid list
def arg_as_list(string_list):
  parsed_list = ast.literal_eval(string_list)
  if type(parsed_list) is not list:
    raise argparse.ArgumentTypeError("Argument \"%s\" is not a list" % (string_list))
  return parsed_list

def load_graph(matfile, adj_matrix_name, label_matrix_name):
    print ("Matfile name: ", matfile)
    mat = loadmat(matfile)
    labels_matrix = mat[label_matrix_name]
    labels_sum = labels_matrix.sum(axis=1)
    indices = numpy.where(labels_sum>0)[0]
    labels_matrix = sp.sparse.csc_matrix(labels_matrix[indices])
    A = mat[adj_matrix_name][indices][:,indices]
    graph = sparse2graph(A)
    labels_count = labels_matrix.shape[1]
    multi_label_binarizer = MultiLabelBinarizer(range(labels_count))
    return mat, A, graph, labels_matrix, labels_count, multi_label_binarizer, indices

# Map nodes to their features (note:  assumes nodes are labeled as integers 1:N)
def load_node_hue(node_hue_file, graph, indices):
    model = None
    features_matrix = None
    model = pandas.read_csv(node_hue_file, header=0, index_col=False )
    model = model.sort_values(["NodeId"])
    model = model.drop(['NodeId'], axis=1)
    
    scaler = RobustScaler()
    columns = list(model)
   
    for col in columns:
        scaler.fit(model[[col]])
        temp_cols = scaler.transform(model[[col]])
        model[col] = temp_cols
    
    model.to_csv(node_hue_file.replace(".csv", "_normalized_robust.csv"))
    
    try:
        features_matrix = numpy.asarray(model)
	return features_matrix 
    except:
        model = model[:-1]
        features_matrix = numpy.asarray(model)
	return features_matrix

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


def calc_metrics(normalized, dataset, algorithm, num_shuffles, all_results, best_classifiers, writetofile, embedding_params, emb_size, clf, output_dir_and_classifier_name):

    columns=["Algorithm", "Dataset", "Train %", "Normalized Embeddings", "Micro-F1", "Macro-F1", "Accuracy", "Num of Shuffles", "Embedding Size", "Classifier"]

    if embedding_params != None:
        columns = columns + list(embedding_params.keys())
    print ('-------------------')
    if writetofile:
        results_df = pandas.DataFrame(columns=columns)

    print (",".join(columns))
    for train_percent in sorted(all_results.keys()):

        avg_score = defaultdict(float)
        for score_dict in all_results[train_percent]:

            for metric, score in iteritems(score_dict):
                avg_score[metric] += score

        for metric in avg_score:
            avg_score[metric] /= len(all_results[train_percent])

        if writetofile:
            temp = {
                     "Dataset": dataset,
                     "Train %": train_percent,
                     "Normalized Embeddings": str(normalized),
                     "Algorithm": algorithm,
                     "Micro-F1": avg_score["micro"],
                     "Macro-F1": avg_score["macro"],
                     "Accuracy": avg_score["accuracy"],
                     "Num of Shuffles": num_shuffles,
                     "Embedding Size":    emb_size,
                     "Classifiers": best_classifiers[train_percent]
                    }
            if embedding_params != None:
                temp.update(embedding_params)
            results_df = results_df.append(temp, ignore_index=True)

        clf_params_string = str(clf.get_params()["estimator"]).replace("\n", "")

        embedd_params_file_name = ""
        if embedding_params != None:
            embedding_params_string = ""
            for key in embedding_params.keys():
                embedd_params_file_name = embedd_params_file_name + "_" + key + "_" + str(embedding_params[key])
                if embedding_params_string == "":
                    embedding_params_string = str(embedding_params[key])
                else:
                    embedding_params_string = embedding_params_string + "," + str(embedding_params[key])
            print ("{},{},{},{},{},{},{},{},{},{},{}".format(algorithm, dataset, normalized, train_percent,                 avg_score["micro"], avg_score["macro"], avg_score["accuracy"],                 num_shuffles, emb_size, clf_params_string, embedding_params_string))
        else:
            print ("{},{},{},{},{},{},{},{},{},{}".format(algorithm, dataset, normalized, train_percent,                 avg_score["micro"], avg_score["macro"], avg_score["accuracy"], num_shuffles,                 emb_size, clf_params_string))

    if writetofile:
        print (col_names)
        output_file_name = os.path.join(output_dir_and_classifier_name + "_" + dataset + "_" + "classi_results" + "_" + algorithm + embedd_params_file_name + col_names + ".csv")
        output_file_obj = open(output_file_name, "a")
        results_df.to_csv(output_file_obj, index = False, sep=',', header=output_file_obj.tell()==0)
        print ("File saved at: {}".format(output_file_name))
        output_file_obj.close()


def predict_top_k(classifier, X, top_k_list):
    assert X.shape[0] == len(top_k_list)
    probs = numpy.asarray(classifier.predict_proba(X))
    all_labels = []
    for i, k in enumerate(top_k_list):
        probs_ = probs[i, :]
        try:
            labels = classifier.classes_[probs_.argsort()[-k:]].tolist()
        except AttributeError:  # for eigenpro
            labels = probs_.argsort()[-k:].tolist()
        all_labels.append(labels)
    return all_labels


def get_dataset_for_classification(X, y, train_percent):
    X_train, X_test, y_train_, y_test_ = train_test_split(X, y, test_size=1-train_percent)
    y_train = sparse_tocoo(y_train_)
    y_test = sparse_tocoo(y_test_)
    return X_train, X_test, y_train_, y_train, y_test_, y_test


def get_classifer_performace(classifer, X_test, y_test, multi_label_binarizer):
    top_k_list_test = [len(l) for l in y_test]
    y_test_pred = predict_top_k(classifer, X_test, top_k_list_test)

    y_test_transformed = multi_label_binarizer.transform(y_test)
    y_test_pred_transformed = multi_label_binarizer.transform(y_test_pred)

    results = {}
    averages = ["micro", "macro"]
    for average in averages:
        results[average] = f1_score(y_test_transformed, y_test_pred_transformed, average=average)
    results["accuracy"] = accuracy_score(y_test_transformed, y_test_pred_transformed)

    print ("======================================================")
    print("Best Scores with best params: {}".format(str(classifer.get_params()["estimator"]).replace("\n", "")))
    for metric_score in results:
        print (metric_score, ": ", results[metric_score])
    print ("======================================================")
    return results



def logistic_regression_classification(X_train, X_test, y_train_, y_train, y_test_, y_test, grid_search, multi_label_binarizer):
    lf_classifer = OneVsRestClassifier(LogisticRegression())
    if grid_search:
        parameters = {
            "estimator__penalty" : ["l1", "l2"],
            "estimator__C": [0.001, 0.01, 0.1, 1, 10, 100]
        }
        lf_classifer = GridSearchCV(lf_classifer, param_grid=parameters, cv=5, scoring='f1_micro', n_jobs=1, verbose=0, pre_dispatch=1)
        lf_classifer.fit(X_train, y_train_.toarray())


    lf_classifer.fit(X_train, y_train_.toarray())

    if grid_search:
        lf_classifer = lf_classifer.best_estimator_

    results = get_classifer_performace(lf_classifer, X_test, y_test, multi_label_binarizer)
    return lf_classifer, results



def svc_classification(X_train, X_test, y_train_, y_train, y_test_, y_test, grid_search, multi_label_binarizer, test_kernel):

    if test_kernel == "linear":
        svc_classifer = OneVsRestClassifier(SVC(kernel="linear", probability=True),n_jobs=-1)

    if test_kernel == "rbf":
        svc_classifer = OneVsRestClassifier(SVC(kernel="rbf", probability=True), n_jobs=-1)

    if test_kernel == 'eigenpro':

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device == 'cpu': print ('Running Eigenpro on cpu will be much slower than on GPU. Consider using GPU.')
        print_flag = False # for debug

        n_class = get_n_class(y_train, y_test)
        y_train_eig, y_test_eig = encoder_(y_train, n_class), encoder_(y_test, n_class)
        X_train, y_train_eig, X_test, y_test_eig = X_train.astype('float32'), y_train_eig.astype('float32'), X_test.astype('float32'), y_test_eig.astype('float32')
        n_train = int(len(X_train)*0.8) # number of datapoints used for hyper-parameter search
        X_val, y_val = X_train[n_train:], y_train_eig[n_train:] # validation set for hyperparameter search
        eval_f = lambda s: init_train_Gaussian(s, X_train[:n_train], y_train_eig[:n_train], X_val, y_val, epochs=[10], n_class=n_class, mem_gb=12, device=device, print_flag=print_flag)
        try:
            best_val, best_param = line_search(eval_f, 1, 50, print_flag=print_flag)  # search kernel bandwidth in [1, 50]
        except numpy.linalg.linalg.LinAlgError:
            print('Bad random seed')
            best_param = 10

        print "best (Gaussian kenrel) bandwidth = " + str(best_param)
        kernel = lambda x, y: Gaussian(x, y, s=best_param)
        model = FKR_EigenPro(kernel, X_train, n_class, device=device)

        res, prob = model.fit(X_train[:n_train], y_train_eig[:n_train], X_val, y_val, epochs=[1, 2, 5, 10, 20], mem_gb=12, print_flag=print_flag)
        results = get_classifer_performace(model, X_test, y_test, multi_label_binarizer)
        return model, results

    if grid_search:

        if test_kernel == "linear":
            parameters = {
                "estimator__C": [0.01, 0.1, 1, 10, 100, 1000],
            }
        if test_kernel == "rbf":
            parameters = {
                "estimator__C": [0.01, 0.1, 1, 10, 100, 1000],
                "estimator__gamma": [0.001, 0.01, 0.1, 1, 10, 100]
            }

        svc_classifer = GridSearchCV(svc_classifer, param_grid=parameters, cv=5, scoring='f1_micro', n_jobs=-1, verbose=0)

    svc_classifer.fit(X_train, y_train_.toarray())

    if grid_search:
        svc_classifer = svc_classifer.best_estimator_
    results =  get_classifer_performace(svc_classifer, X_test, y_test, multi_label_binarizer)
    return svc_classifer, results


def validate_inputs(network, training_percents, dataset, algorithm, writetofile):
    if network == "":
        print ("Missing network file path.")
        sys.exit(1)
    elif (os.path.exists(network) == False):
        print ("Network file path does not exist.")
        sys.exit(1)
    if type(training_percents) != type([]):
        print ("Training percents must be given in a list format. (i.e. [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])")
        sys.exit(1)
    if dataset == "" and writetofile == True:
        print ("Missing dataset name. Dataset name need for file output")
        sys.exit(1)
    if algorithm == "" and writetofile == True:
        print ("Missing algorithm name. Algorithm name need for file output")
        sys.exit(1)


def classify(emb="", network="", training_percents=training_percents_default, dataset="", writetofile=True, algorithm="", adj_matrix_name="network", label_matrix_name="group", num_shuffles=10, word2vec_format=True, embedding_params=None, classifier="LR",  test_kernel="linear", grid_search=True, output_dir="./"):

    # 0. Validate inputs parameters:
    validate_inputs(network, training_percents, dataset, algorithm, writetofile)

    # 1. Load labels
    mat, A, graph, labels_matrix, labels_count, multi_label_binarizer, indices = load_graph(network, adj_matrix_name, label_matrix_name)

    # 2. Load embeddings
    features_matrix = load_node_hue(emb, graph, indices)
    feature_matrices = [(False, features_matrix), ]
    
    # 3. Fit multi-label binarizer
    y_fitted = multi_label_binarizer.fit(labels_matrix)

    # 4. Store classifier name for file
    classifer_name_for_file = classifier

    if classifier == 'SVM':
        classifer_name_for_file = classifer_name_for_file + "_" + test_kernel

    # 5. Train
    for features_matrix_tuple in feature_matrices:

        # Dict of lists to score each train/test group
        all_results = defaultdict(list)
        best_classifiers = defaultdict(list)
        best_classifiers_coef = defaultdict(list)
        normalized = features_matrix_tuple[0]
        features_matrix = features_matrix_tuple[1]

        print ("========================= Normalized Embeddings: " + str(normalized) + " =========================")

        for train_percent in training_percents:

            print ("Starting with split %s", train_percent)

            for x in range(num_shuffles):

                print ("Shuffle number %s", x)

                # Shuffle, to create train/test groups
                shuf = skshuffle(features_matrix, labels_matrix)

                X, y = shuf
                X_train, X_test, y_train_, y_train, y_test_, y_test = get_dataset_for_classification(X, y, train_percent)
                
		clf = None
                results ={}
                if classifier == 'LR':
                    clf, results = logistic_regression_classification(X_train, X_test, y_train_, y_train, y_test_, y_test, grid_search, multi_label_binarizer)

                elif classifier == 'SVM':
                     clf, results = svc_classification(X_train, X_test, y_train_, y_train, y_test_, y_test, grid_search, multi_label_binarizer, test_kernel)

                elif classifier == 'EigenPro':
                    clf, results = svc_classification(X_train, X_test, y_train_, y_train, y_test_, y_test, grid_search,
                                                      multi_label_binarizer, test_kernel='eigenpro')

                all_results[train_percent].append(results)
                best_classifiers[train_percent].append(clf)
                
		del X_train, X_test, y_train_, y_train, y_test_, y_test, shuf
            
	    print ("Done with %s", train_percent)
	calc_metrics(normalized, dataset, algorithm, num_shuffles, all_results, best_classifiers, writetofile, embedding_params, X.shape[1], clf, output_dir + classifer_name_for_file)

def main():
    parser = ArgumentParser("scoring",formatter_class=ArgumentDefaultsHelpFormatter,conflict_handler='resolve')

    # Required arguments
    parser.add_argument("--heuristic_file", type=str, required=True, help='The path and name of the heuristic file')
    parser.add_argument("--network", type=str, required=True, help='The path and name of the .mat file containing the adjacency matrix and node labels of the input network')
    parser.add_argument("--dataset", type=str, required=True, help='The name of your dataset (used for output)')

    # Optional arguments
    parser.add_argument("--num_shuffles", default=10, type=int, help='The number of shuffles for training')
    parser.add_argument("--writetofile", action="store_true", help='If the flag is set, then the results are written to a file')
    parser.add_argument("--adj_matrix_name", default='network', help='The name of the adjacency matrix inside the .mat file')
    parser.add_argument("--word2vec_format", action="store_true", help='If the flag is set, then genisim is used to load the embeddings')
    parser.add_argument("--embedding_params", type=json.loads, help='"embedding_params": Dictionary of parameters used for embedding generation (used to print/save results), type:dict')
    parser.add_argument("--training_percents", default=training_percents_default, type=arg_as_list,  help='List of split "percents" for training and test sets (default is [0.1, 0.5, 0.9]')
    parser.add_argument("--label_matrix_name", default='group', help='The name of the labels matrix inside the .mat file')
    parser.add_argument("--classifier", default="LR",choices=["LR","SVM", "EigenPro"], help='Classifier to be used; Choose from "LR" or "SVM"')
    parser.add_argument("--test_kernel", default="linear",choices=["linear", "rbf", "eigenpro"], help='Kernel to be used for SVM classifier; Choose from "linear" or "rbf"')
    parser.add_argument("--grid_search", action="store_false", help='If the flag is set, then grid search is NOT used.')
    parser.add_argument("--output_dir", default="./", type=str, help='Specify the path to store results')
    args = parser.parse_args()

    print (args)
    if args.embedding_params == None:
        args.embedding_params = {}
    assert args.embedding_params != None

    classify(emb=args.heuristic_file, network=args.network, dataset=args.dataset, algorithm="heuristics",             num_shuffles=args.num_shuffles, writetofile=args.writetofile,             classifier=args.classifier, adj_matrix_name=args.adj_matrix_name,             word2vec_format=args.word2vec_format, embedding_params=args.embedding_params,             training_percents=args.training_percents,             test_kernel=args.test_kernel, label_matrix_name=args.label_matrix_name, grid_search=True)


if __name__ == "__main__":
    main()
