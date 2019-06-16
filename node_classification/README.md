# Node Classification
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

### Node Heuristics feature generation
```
node_heuristics.py is used to generate node huerisitc features.
Using this as a stand alone module/script:
1) Can be called from command line with the following arugments:
    # Required arguments (must be given in the correct order):
    "--edgelist", type=string, required=True, help='The path and name of the edgelist file with no weights containing the edgelist of the input network'
    "--network", type=string, required=True, help='The path and name of the .mat file containing the adjacency matrix and node labels of the input network'
    "--dataset", type=string,required=True, help='The name of your dataset (used for output)'
    # Optional arguments
    "--adj_matrix_name", default='network', help='The name of the adjacency matrix inside the .mat file'
    "--label_matrix_name", default='group', help='The name of the labels matrix inside the .mat file'
Example for LR classification:
python node_heuristics.py --network ../blogcatalog.mat --edgelist ../blogcatalog.edgelisht --dataset blogcatalog --adj_matrix_name network --label_matrix_name group
```

### Node Heuristics Evaluation
```
node_heuristics_classification.py is a modification of the classification script used to conduct the node classification experiments.
node_heuristics_classification.py reads in the node heurisitcs and evaluates them in a similar manner to the classification script. 
A couple of key differences include:
	- Input: We require the heuristics .csv file (generated from 'Node Heuristics feature generation' step, using node_heuristics.py)
	- Normalization: We do NOT conducted any l2 normalization, instead we use Scikit Learn's RobustScaler to perform columwise normalization
```
