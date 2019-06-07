# Network Representation learning Benchmarks

This repository provides evaluation scripts and datasets with train/test splits for the paper, Network Representation Learning: Consolidation and Renewed Bearing, arXiv preprint arXiv:1905.00987 (2019).

## Link Prediction

### Pre-created LP splits
[Here](https://pip.pypa.io/en/stable/) is the link to the link prediction splits used in the paper.

### Create your own LP split
'''
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
'''
