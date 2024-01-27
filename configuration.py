NB_ID = "notebook-altegrad"  # This will be the name which appears on Kaggle.
GIT_USER = "HugoRbrt"  # Your git user name
GIT_REPO = "altegrad_project"  # Your current git repo
# Keep free unless you need to acess kaggle datasets. 
# You'll need to modify the remote_training_template.ipynb.
KAGGLE_DATASET_LIST = ['hugorbrt/nlplsv3']
CFG_EXPERIMENTS = {
    0:{
        'who': 'matteo',
        'learning_rate': 2e-5,
        'name_exp': " GAT + SAGE + LEConv + skip(3) + scheduler linear + BIG Training + 2 devices training + FLOAT16 ",
        'scheduler': 'linear',
        'nb_epochs': 60,
        'batch_size': 64,
        'model_name': 'distilbert-base-uncased',
        'num_node_features': 300,
        'nout':  768,
        'nhid': 300,
        'graph_hidden_channels': 300,
        'num_warmup_steps': 1000,
        'heads': 30,
        'comment': 'Try to combine GAT / SAGE / LEConv with a big training',
        'T_max': '',
        'n_heads_text':12, 
        'n_layers_text':10, 
        'hidden_dim_text':3072, 
        'dim_text':768,
        'device_1': 'cuda:0',
        'device_2': 'cuda:1',
        'with_fast_tokenizer': True
    },
}
