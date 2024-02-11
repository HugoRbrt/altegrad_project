NB_ID = "notebook-altegrad"  # This will be the name which appears on Kaggle.
GIT_USER = "HugoRbrt"  # Your git user name
GIT_REPO = "altegrad_project"  # Your current git repo
# Keep free unless you need to acess kaggle datasets. You'll need to modify the remote_training_template.ipynb.
KAGGLE_DATASET_LIST = ['hugorbrt/nlplsv3']
CFG_EXPERIMENTS = {
    0: {  
        'who': 'baptiste',
        'learning_rate': 8e-4,
        'name_exp': "UMAP",
        'scheduler': 'linear',
        'nb_epochs':75,
        'batch_size': 200,
        'model_name': 'distilbert-base-uncased',  # scibert_scivocab_uncased
        'num_node_features': 300,
        'nout':  768,
        'nhid': 900,
        'graph_hidden_channels': 300,
        'num_warmup_steps': 1000,
        'heads': 25,
        'comment': '',
        'T_max': '',
        'n_heads_text':8, 
        'n_layers_text':5, 
        'hidden_dim_text':3072, 
        'dim_text':768,
        'device_1': 'cuda:0',
        'device_2': 'cuda:1',
        'with_fast_tokenizer': True,
    },
}

    

