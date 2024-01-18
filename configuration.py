NB_ID = "notebook-altegrad"  # This will be the name which appears on Kaggle.
GIT_USER = "HugoRbrt"  # Your git user name
GIT_REPO = "altegrad_project"  # Your current git repo
# Keep free unless you need to acess kaggle datasets. You'll need to modify the remote_training_template.ipynb.
KAGGLE_DATASET_LIST = ['hugorbrt/nlplsv3']
CFG_EXPERIMENTS = {
    0:{  
        'who': 'matteo',
        'name_exp': "SAGEConv(3) + Linear(3) + scheduler + skip + big training",
        'scheduler': '',
        'nb_epochs': 30,
        'batch_size': 24,
        'learning_rate': 2e-5,
        'model_name': 'distilbert-base-uncased',
        'num_node_features': 300,
        'nout':  768,
        'nhid': 450,
        'graph_hidden_channels': 450,
        'heads': 0,
        'comment': 'Guess who is back',
        'T_max': 10,
    },
}

# scibert_scivocab_uncased