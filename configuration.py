NB_ID = "notebook-altegrad"  # This will be the name which appears on Kaggle.
GIT_USER = "HugoRbrt"  # Your git user name
GIT_REPO = "altegrad_project"  # Your current git repo
# Keep free unless you need to acess kaggle datasets. You'll need to modify the remote_training_template.ipynb.
KAGGLE_DATASET_LIST = ['hugorbrt/nlplsv3']
CFG_EXPERIMENTS = {
    0:{  
        'who': GIT_USER,
        'name_exp': "bioMegatron & MLP 3 layers",
        'nb_epochs': 15,
        'batch_size': 8,
        'learning_rate': 3e-5,
        'model_name': 'EMBO/BioMegatron345mUncased',
        'num_node_features': 300,
        'nout':  1024,
        'nhid': 600,
        'heads': 20,
        'graph_hidden_channels': 300,
        'scheduler': "linear_with_warmup",
        'num_warmup_steps': 1000,
        'comment': '',
    },
}