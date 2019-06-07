# Network Representation learning Benchmarks

This repository provides evaluation scripts and datasets with train/test splits for the paper, Network Representation Learning: Consolidation and Renewed Bearing, arXiv preprint arXiv:1905.00987 (2019).

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

## Node Classification

## Node Classification Heuristics

