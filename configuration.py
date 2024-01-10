NB_ID = "test-notebook"  # This will be the name which appears on Kaggle.
GIT_USER = "HugoRbrt"  # Your git user name
GIT_REPO = "altegrad_project"  # Your current git repo
# Keep free unless you need to acess kaggle datasets. You'll need to modify the remote_training_template.ipynb.
KAGGLE_DATASET_LIST = ['hugorbrt/nlplsv3']
CFG_EXPERIMENTS = {
    '0':{  
        'who': GIT_USER,
        'name_exp': "Config repo",
        'nb_epochs': 1,
        'batch_size': 32,
        'learning_rate': 2e-5,
        'model_name': 'distilbert-base-uncased',
        'num_node_features': 50,
        'nout':  768,
        'nhid': 20,
        'graph_hidden_channels': 50,
        'comment': '',
    },
    '1':{  
        'who': GIT_USER,
        'name_exp': "Config repo",
        'nb_epochs': 1,
        'batch_size': 16,
        'learning_rate': 2e-5,
        'model_name': 'distilbert-base-uncased',
        'num_node_features': 100,
        'nout':  768,
        'nhid': 50,
        'graph_hidden_channels': 30,
        'comment': '',
    },
}