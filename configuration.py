NB_ID = "notebook-altegrad"  # This will be the name which appears on Kaggle.
GIT_USER = "HugoRbrt"  # Your git user name
GIT_REPO = "altegrad_project"  # Your current git repo
# Keep free unless you need to acess kaggle datasets. 
# You'll need to modify the remote_training_template.ipynb.
KAGGLE_DATASET_LIST = ['hugorbrt/nlplsv3']
CFG_EXPERIMENTS = {
    0:{  
        'who': 'matteo',
        'name_exp': "SMALL Size + GNN(1)",
        'scheduler': 'Linear',
        'nb_epochs': 20,
        'batch_size': 48,
        'learning_rate': 2e-5,
        'model_name': 'distilbert-base-uncased',
        'num_node_features': 300,
        'nout':  768,
        'nhid': 450,
        'graph_hidden_channels': 450,
        'heads': 20,
        'comment': 'Guess who is back',
        'T_max': 10,
        'num_warmup_steps': 1000,
    },
    1:{
        'who': 'baptiste',
        'learning_rate': 2e-5,
        'name_exp': "(test) GAT + skip(3) + scheduler linear + float16 + n_head=4 + n_layer=4",
        'scheduler': 'lineair',
        'nb_epochs': 5,
        'batch_size': 100,
        'model_name': 'distilbert-base-uncased',
        'num_node_features': 300,
        'nout':  768,
        'nhid': 300,
        'graph_hidden_channels': 300,
        'num_warmup_steps': 1000,
        'heads': 20,
        'comment': '',
        'T_max': '',
        'n_heads_text':4, 
        'n_layers_text':4, 
        'hidden_dim_text':3072, 
        'dim_text':768,
        'device_1': 'cuda:0',
        'device_2': 'cuda:0',
    },
}
