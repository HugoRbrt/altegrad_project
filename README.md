# Author 

- Callard Baptiste
- Marengo Matteo 
- Robert Hugo

# Objective 

The aim of the project, as described in the document, is to address the advanced task of retrieving molecules using natural language descriptions as queries, introduced by the paper "Text2Mol: Cross-Modal Molecule Retrieval with Natural Language Queries." This challenge involves developing a method to match textual data with molecular structures from a large database effectively. The project was undertaken as part of the ALTEGRAD course during the MVA master's semester and is part of a Kaggle challenge. The goal was to explore the applications of this task and analyze methods proposed in the literature, leading to the proposition of a new architecture based on the concepts studied in the ALTEGRAD course.

# What you can find in the repo

After running the experiments, the losses, LRAP, .csv submission, validation set alignment matrix and UMAP can be found at the following address 
Wandb : 

- model.py: constrains all our models. It contains graph and text encoders.
- train.py: This is the pipeline for training and recording the various metrics. 
- configuration.py: This file contains a configuration that can be easily modified. 
- data_loader.py : Data manipulation

Pipeline : 

As we don't have access to powerfull cluster we adapted a pipeline from : https://github.com/balthazarneveu/mva_pepites. So you can find some file to pull the code from our own Repositery GitHub and the push it on Kaggle. This code in contains in : 

- local_train.py
- remote_training_template.ipynb
- remote_training.py

The code can be easily launched : 

```
python remote_training.py --user user0 --branch your_branch_name [OPTION] --nowandb
```

# Credit 

We have adapted the code from https://github.com/balthazarneveu/mva_pepites for our pipeline. This is generic code that allows deep learning code to be executed using the Kaggle API and facilitates collaboration within the team. 