NB_ID = "notebook-altegrad"  # This will be the name which appears on Kaggle.
GIT_USER = "HugoRbrt"  # Your git user name
GIT_REPO = "altegrad_project"  # Your current git repo
# Keep free unless you need to acess kaggle datasets. You'll need to modify the remote_training_template.ipynb.
KAGGLE_DATASET_LIST = ['hugorbrt/nlplsv3']
CFG_EXPERIMENTS = {
    0:{  
        'who': 'matteo',
        'learning_rate': 9e-4,
        'name_exp': "SuperGAT 4 Layers + Accelerate + 30 heads + skip + High LR + 70 epochs ",
        'scheduler': 'linear',
        'nb_epochs': 70,
        'batch_size': 64,
        'model_name': 'distilbert-base-uncased',
        'num_node_features': 300,
        'nout':  768,
        'nhid': 1500,
        'graph_hidden_channels': 300,
        'num_warmup_steps': 1000,
        'heads': 25,
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
