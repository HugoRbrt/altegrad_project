NB_ID = "notebook-altegrad"  # This will be the name which appears on Kaggle.
GIT_USER = "HugoRbrt"  # Your git user name
GIT_REPO = "altegrad_project"  # Your current git repo
# Keep free unless you need to acess kaggle datasets. You'll need to modify the remote_training_template.ipynb.
KAGGLE_DATASET_LIST = ['hugorbrt/nlplsv3']
CFG_EXPERIMENTS = {
    0:{  
        'who': GIT_USER,
        'name_exp': "MLP 15 epoch",
        'nb_epochs': 15,
        'batch_size': 24,
        'learning_rate': 2e-5,
        'model_name': 'distilbert-base-uncased',
        'num_node_features': 300,
        'nout':  768,
        'nhid': 300,
        'heads': 30,
        'graph_hidden_channels': 300,
        'scheduler': "CosineAnnealingLR",
        'comment': '',
        
    },
}