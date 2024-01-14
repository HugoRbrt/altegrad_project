NB_ID = "notebook-altegrad"  # This will be the name which appears on Kaggle.
GIT_USER = "HugoRbrt"  # Your git user name
GIT_REPO = "altegrad_project"  # Your current git repo
# Keep free unless you need to acess kaggle datasets. You'll need to modify the remote_training_template.ipynb.
KAGGLE_DATASET_LIST = ['hugorbrt/nlplsv3']
CFG_EXPERIMENTS = {
    # 0:{  
    #     'who': 'baptiste',
    #     'name_exp': "maxpooling + GATv2Conv(3) + skip + distilbert-base-cased",
    #     'scheduler': 'CosineAnnealingLR',
    #     'nb_epochs': 15,
    #     'batch_size': 24,
    #     'learning_rate': 2e-5,
    #     'model_name': 'distilbert-base-cased',
    #     'num_node_features': 300,
    #     'nout':  768,
    #     'nhid': 300,
    #     'graph_hidden_channels': 300,
    #     'heads': 30,
    #     'comment': '',
    # },
    0:{  
        'who': 'baptiste',
        'name_exp': "RGCNConv(3) + scheduler",
        'scheduler': '',
        'nb_epochs': 15,
        'batch_size': 24,
        'learning_rate': 2e-5,
        'model_name': 'distilbert-base-uncased',
        'num_node_features': 300,
        'nout':  768,
        'nhid': 512,
        'graph_hidden_channels': '',
        'heads': '',
        'comment': '',
        'T_max': 10,
    },
}