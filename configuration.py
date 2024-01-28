NB_ID = "notebook-altegrad"  # This will be the name which appears on Kaggle.
GIT_USER = "HugoRbrt"  # Your git user name
GIT_REPO = "altegrad_project"  # Your current git repo
# Keep free unless you need to acess kaggle datasets. You'll need to modify the remote_training_template.ipynb.
KAGGLE_DATASET_LIST = ['hugorbrt/nlplsv3']
CFG_EXPERIMENTS = {
    0:{  
        'who': GIT_USER,
        'learning_rate': 2e-5,
        'name_exp': "MLP",
        'scheduler': 'lineair',
        'nb_epochs': 50,
        'batch_size': 200,
        'model_name': 'distilbert-base-uncased',
        'num_node_features': 500,
        'nout':  768,
        'nhid': 900,
        'graph_hidden_channels': 300,
        'num_warmup_steps': 1000,
        'heads': 20,
        'comment': '',
        'T_max': '',
        'n_heads_text':12, 
        'n_layers_text':6, 
        'hidden_dim_text':3072, 
        'dim_text':768,
        'device_1': 'cuda:0',
        'device_2': 'cuda:1',
        'with_fast_tokenizer': True,
    },
}
