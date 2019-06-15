# Network Representation learning Benchmarks

This repository provides evaluation scripts and datasets with train/test splits for the paper, Network Representation Learning: Consolidation and Renewed Bearing, arXiv preprint arXiv:1905.00987 (2019).
Here is the [link](https://arxiv.org/pdf/1905.00987.pdf) to the paper

## Link Prediction

### Pre-created LP splits
[Here](https://app.box.com/s/idz6xv8l4mlivtgmyw0sd85l2jhi0pji) is the link to the link prediction splits used in the paper.

### Create your own LP split
```
Example Usage:
To create PPI split (5 folds - 80 - 20% learning-prediction split)

Assumption:
ppi.mat file in /home/usr/Downloads/ppi.mat
Directory for Output : link_prediction_data/ppi_80_20 (assuming these directories are already created)
Name of the folds, we wish to save :'fold'

For undirected graphs like PPI, Blog, youtube etc.
python link_prediction/create_lp_splits.py --input /home/usr/Downloads/ppi.mat --output_dir link_prediction_data/ppi_80_20/ --num_folds 5 --file_name fold

Following is the structure created:

|_ link_prediction_data
  |_ppi_80_20
    |_ fold_1.mat
    |_ fold_2.mat
    |_ fold_3.mat
    |_ fold_4.mat
    |_ fold_5.mat

For directed graphs like pubmed:
python link_prediction/create_lp_splits.py --input /home/usr/Downloads/dblp/pubmed.mat --output_dir link_prediction_data/pubmed_80_20/ --num_folds 5 --file_name fold --directed
```
### Link Prediction Evaluation
```
Example Usage for  ppi 5 folds
To generate the required split, see create_datasets_v1_0.py

Assuming the splitted files are in the following organization:
|--> ppi_80_20
    |->fold_0.mat
    |->fold_1.mat
    |->fold_2.mat
    |->fold_3.mat
    |->fold_4.mat

Assuming the node embeddings for each fold obtained from methods are saved in the following format:
|--> ppi_80_20_embeddings_u
    |->ppi_embedding_0.npy
    |->ppi_embedding_1.npy
    |->ppi_embedding_2.npy
    |->ppi_embedding_3.npy
    |->ppi_embedding_4.npy

Assuming the context embeddings are saved in the following format:
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
```

## Link Prediction Heuristics
```
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
```


## Node Classification
[Here](https://app.box.com/s/wbj3zafd61ymp2uechr4oh6jsu6n7nry) is the link to the node classification datasets.
```
There are two ways of using node classification script:
1) As an import
OR
2) As a stand alone script
----------------------------------------------
Using this as an imported module:
1) Add "import classification"
2) Use the classify function by calling it:
   classification.classify(...) ----> Note ... are the arguments you need to set
3) The classify function takes the following arguments:
    Required arguments:
        - "emb": The path and name of the embeddings file, type:string
        - "network": The path and name of the .mat file containing the adjacency matrix and node labels of the input network, type:string
        - "dataset": The name of your dataset (used for output), type:string
        - "algorithm", The name of the algorithm used to generate the embeddings (used for output), type:string
    Optional arguments
        - "num_shuffles": The number of shuffles for training, type:int
            - default value: 10
        - "writetofile": If true, output classification results to file'), type:bool
            - default value: True
        - "adj_matrix_name": The name of the adjacency matrix inside the .mat file, type:string
            - default value: "network"
        - "word2vec_format": If true, genisim is used to load the embeddings, type:bool
            - default value: True
        - "embedding_params": Dictionary of parameters used for embedding generation (used to print/save results), type:dict
            - default value: {}
        - "training_percents": List of split "percents" for training and test sets '), type:list
            - default value: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        - "label_matrix_name": The name of the labels matrix inside the .mat file, type:string
            - default value: "group"
        - "classifier": Classifier to be used; Choose from "LR", "SVM", or "EigenPro", type:string
            - default value = "LR"
        - "test_kernel": The kernel to use for the SVM/SVC classifer (not improtant for LR classifier); Choose from "eigenpro", "linear" or "rbf", type:string
            - default value = "linear"
        - "grid_search": If true, best parameters are found for classifier (increases run time), type:bool
            - default value = True
        - "output_dir": Specify the path to store results, type:str
            - default value = "./"
Example for LR classification:
#classification.classify(emb="../emb/blogcatalog.emb", network="../blogcatalog.mat", dataset="blogcatalog", algorithm="walkets", word2vec_format=True, embedding_params={"walk" :10, "p" : 1})
Example for SVM (linear kernel) classification:
#classification.classify(emb="../emb/embed_blogcatalog.npy", network="../blogcatalog.mat", dataset="blogcatalog", algorithm="walkets", classifier="SVC", word2vec_format=False, embedding_params={"walk" :10, "p" : 1})
Example for SVM (rbf kernel) classification:
classification.classify(emb="../emb/embed_blogcatalog.npy", network="../blogcatalog.mat", dataset="blogcatalog", algorithm="walkets", classifier="SVC", test_kernel="rbf", word2vec_format=False, embedding_params={"walk" :10, "p" : 1})

Using this as a stand alone module/script:
1) Can be called from command line with the following arugments:
	# Required arguments (must be given in the correct order):
	"--emb", type=string, required=True, help='The path and name of the embeddings file
	"--network", type=string, required=True, help='The path and name of the .mat file containing the adjacency matrix and node labels of the input network
	"--dataset", type=string,required=True, help='The name of your dataset (used for output)
	"--algorithm", type=string, required=True, help='The name of the algorithm used to generate the embeddings (used for output)
	# Flags (use if true, they don't require additional parameters):
	"--writetofile": If used, classification results to are written to a file
	"--word2vec_format": If used, genisim is used to load the embeddings
	# Optional arguments
	"--num_shuffles", default=10, type=int, help='The number of shuffles for training'
	"--adj_matrix_name", default='network', help='The name of the adjacency matrix inside the .mat file'
	"--word2vec_format", action="store_false", help='If true, genisim is used to load the embeddings'
	"--embedding_params", type=json.loads, help='"embedding_params": Dictionary of parameters used for embedding generation (used to print/save results), type:dict
	"--training_percents", default=training_percents_default, type=arg_as_list,  help='List of split "percents" for training and test sets (i.e. [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]')
	"--label_matrix_name", default='group', help='The name of the labels matrix inside the .mat file'
	"--classifier", default="LR",choices=["LR","SVM","EigenPro"], help='Classifier to be used; Choose from "EigenPro", "LR" or "SVM"'
	"--test_kernel", default="linear",choices=["linear","rbf","eigenpro"], help='Kernel to be used for SVM classifier; Choose from "linear" or "rbf"'
	"--output_dir", default="./", type=str, help='Specify the path to store results'
Example for LR classification:
python classification.py --emb ../emb/blogcatalog.npy --network ../blogcatalog.mat --dataset blogcatalog --algorithm walkets --training_percents '[0.1, 0.5]' --embedding_params '{"walk" :10, "p" : 1}' --adj_matrix_name network --label_matrix_name group  --writetofile
Example for SVM (linear kernel) classification:
python classification.py --classifier SVM --test_kernel linear --emb ../emb/blogcatalog.emb --network ../blogcatalog.mat --dataset blogcatalog --algorithm walkets --training_percents '[0.1, 0.5]' --embedding_params '{"walk" :10, "p" : 1}' --adj_matrix_name network --label_matrix_name group  --word2vec_format --grid_search
Example for SVM (rbf kernel) classification:
python classification.py --classifier SVM --test_kernel rbf --emb ../emb/blogcatalog.emb --network ../blogcatalog.mat --dataset blogcatalog --algorithm walkets --training_percents '[0.1, 0.5]' --embedding_params '{"walk" :10, "p" : 1}' --adj_matrix_name network --label_matrix_name group  --word2vec_format --grid_search
```
## Node Classification Heuristics

## Requirements
environment.yml contains the necessary virtualenv requirements for running all the scripts in the repository. 

## Citing
If you find our paper or the code relevant and useful for your research, we request you to cite our paper
```
@article{gurukar2019network,
  title={Network Representation Learning: Consolidation and Renewed Bearing},
  author={Gurukar, Saket and Vijayan, Priyesh and Srinivasan, Aakash and Bajaj, Goonmeet and Cai, Chen and Keymanesh, Moniba and Kumar, Saravana and Maneriker, Pranav and Mitra, Anasua and Patel, Vedang and others},
  journal={arXiv preprint arXiv:1905.00987},
  year={2019}
}
```

