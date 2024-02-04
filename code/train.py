from .model import Model
from .data_loader import GraphTextDataset, GraphDataset, TextDataset
from torch import optim
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import label_ranking_average_precision_score
import time
from torch_geometric.data import DataLoader
import wandb
import torch
import uuid
import os
import pandas as pd
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader as TorchDataLoader
import numpy as np
from transformers import AutoTokenizer
from transformers.optimization import get_linear_schedule_with_warmup
import umap.umap_ as umap
import matplotlib.pyplot as plt

CE = torch.nn.CrossEntropyLoss()

def contrastive_loss(v1, v2):
    v1, v2 = v1.float(), v2.float()
    logits = torch.matmul(v1,torch.transpose(v2, 0, 1))
    labels = torch.arange(logits.shape[0], device=v1.device)
    return CE(logits, labels) + CE(torch.transpose(logits, 0, 1), labels)

def hard_contrastive_loss(v1, v2, t=0.07, beta=0.1):
    v1, v2 = v1.float(), v2.float()
    logits = torch.matmul(v1, v2.T) / t
    N = logits.size(1) - 1  # Assuming square matrix excluding self-comparison

    pos_exp = torch.exp(logits.diag())
    neg_exp = torch.exp(logits.fill_diagonal_(0)).sum(axis=1)
    reweight = (beta * neg_exp) / neg_exp.mean()

    # Calculate the hard negative samples with reweighting
    Neg = torch.max(((-N * pos_exp + reweight * neg_exp)), torch.tensor(0.0).to(v1.device))

    # Hard sampling loss calculation
    hard_loss = -torch.log(pos_exp / (pos_exp + Neg))
    return hard_loss.mean() 

def run_experiment(cfg, cpu=False, no_wandb=False):
    """this function allows to run an experiments with the given configuration in cfg
    (see local_train.py->configure_experiment for the format of cfg)
    You can add configurations possibilities in the cfg

    Args:
        cfg (dict): contains all informations to run the experiments
        cpu (bool, optional): if True, force CPU. Defaults to False.
        no_wandb (bool, optional): if True, disable wandb. Defaults to False.
    """
    if not no_wandb:
        run = wandb.init(
        project="text2mol",
        entity='team-nlpls',
        name=cfg['name_exp'],
        config=cfg,
        )
    nb_epochs = cfg['nb_epochs']
    batch_size = cfg['batch_size']
    learning_rate =cfg['learning_rate']
    model_name =cfg['model_name']
    
    if cfg['with_fast_tokenizer']:
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)

    gt = np.load("/kaggle/input/nlplsv3/kaggle/working/token_embedding_dict.npy", allow_pickle=True)[()]
    val_dataset = GraphTextDataset(root='/kaggle/input/nlplsv3/kaggle/working/', gt=gt, split='val', tokenizer=tokenizer)
    train_dataset = GraphTextDataset(root='/kaggle/input/nlplsv3/kaggle/working/', gt=gt, split='train', tokenizer=tokenizer)
    test_cids_dataset = GraphDataset(root='/kaggle/input/nlplsv3/kaggle/working/', gt=gt, split='test_cids')
    test_text_dataset = TextDataset(file_path='/kaggle/input/nlplsv3/kaggle/working/test_text.txt', tokenizer=tokenizer)
    
    # device = "cpu" if cpu else ("cuda" if torch.cuda.is_available() else "cpu")
    
    device_1  = cfg['device_1']
    device_2 = cfg['device_2']
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_cids_dataset, batch_size=batch_size//4, shuffle=False)
    test_text_loader = TorchDataLoader(test_text_dataset, batch_size=batch_size//4, shuffle=False)


    model = Model(
        model_name=model_name, 
        num_node_features=cfg['num_node_features'], 
        nout=cfg['nout'], 
        nhid=cfg['nhid'], 
        graph_hidden_channels=cfg['graph_hidden_channels'], 
        heads=cfg['heads'], 
        device_1=device_1, 
        device_2=device_2, 
        n_heads_text=cfg['n_heads_text'], 
        n_layers_text=cfg['n_layers_text'], 
        hidden_dim_text=cfg['hidden_dim_text'], 
        dim_text=cfg['dim_text']
        ) # nout = bert model hidden dim
    # model.to(device)
    print(model)
    
    
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate,
                                    betas=(0.9, 0.999),
                                    weight_decay=0.01)
    
    scaler = GradScaler()
    
    num_warmup_steps = cfg['num_warmup_steps']
    num_training_steps = nb_epochs * len(train_loader) - num_warmup_steps
    scheduler_lr = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = num_warmup_steps, num_training_steps = num_training_steps) 
    
    # checkpoint = torch.load('/kaggle/input/models-retrain/model100(2).pt')
    
    # model.load_state_dict(checkpoint['model_state_dict'])
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    # scheduler_lr.load_state_dict(checkpoint['scheduler_state_dict'])
    # scaler.load_state_dict(checkpoint['scaler_state_dict'])
    

    
    scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    
    # scheduler_expo = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.1, last_epoch=-1, verbose=False)

    epoch = 0
    loss = 0
    losses = []
    count_iter = 0
    time1 = time.time()
    printEvery = 50
    best_validation_loss = 1000000
    best_lrap = -1000000
    for i in range(nb_epochs):
        print('-----EPOCH{}-----'.format(i+1))
        model.train()
        for batch in train_loader:
            input_ids = batch.input_ids
            batch.pop('input_ids')
            attention_mask = batch.attention_mask
            batch.pop('attention_mask')
            graph_batch = batch
            with autocast():
                x_graph, x_text = model(graph_batch.to(device_1), 
                                        input_ids.to(device_2), 
                                        attention_mask.to(device_2))
            current_loss = contrastive_loss(x_graph.to(device_1), x_text.to(device_1))   
            optimizer.zero_grad()
            # current_loss.backward()
            # optimizer.step()
            scaler.scale(current_loss).backward()  # Backpropagation
            scaler.step(optimizer)         # Unscales gradients and calls optimizer.step()
            scaler.update() 
            scheduler_lr.step()
            loss += current_loss.item()
            
            count_iter += 1
            if count_iter % printEvery == 0:
                time2 = time.time()
                print("Iteration: {0}, Time: {1:.4f} s, training loss: {2:.4f}".format(count_iter,
                                                                            time2 - time1, loss/printEvery))
                if not no_wandb:
                    wandb.log({
                        "epoch/train": i, 'loss/train': loss/printEvery, 'loss/train2': loss/(batch_size*printEvery),
                    })
                losses.append(loss)
                loss = 0 
        
        model.eval()  
        # scheduler_expo.step()     
        val_loss = 0
        graph_embeddings = []
        text_embeddings = []
        with torch.no_grad():    
            for batch in val_loader:
                input_ids = batch.input_ids
                batch.pop('input_ids')
                attention_mask = batch.attention_mask
                batch.pop('attention_mask')
                graph_batch = batch
                x_graph, x_text = model(graph_batch.to(device_1), 
                                        input_ids.to(device_2), 
                                        attention_mask.to(device_2))
                for j in range(x_graph.shape[0]):
                    graph_embeddings.append(x_graph[j].tolist())
                    text_embeddings.append(x_text[j].tolist())
                current_loss = contrastive_loss(x_graph.to(device_1), x_text.to(device_1))   
                val_loss += current_loss.item()
        
        similarity = cosine_similarity(text_embeddings, graph_embeddings)
        
        n_samples = len(text_embeddings)
        labels = np.eye(n_samples)

        # Calculate the LRAP score
        lrap_score = label_ranking_average_precision_score(labels, similarity)
        
        best_validation_loss = min(best_validation_loss, val_loss)
        best_lrap = max(best_lrap, lrap_score)
        print('-----EPOCH'+str(i+1)+'----- done.  Validation loss: ', str(val_loss/(batch_size*len(val_loader))) )
        print('LRAP score: ', str(lrap_score))
        if not no_wandb:
            
            wandb.log({
                'epoch/val': i,
                'loss/val':  val_loss/len(val_loader),
                'loss/val2':  val_loss/(batch_size*len(val_loader)),
                'LRAP score': lrap_score,
                'accuract/val': 0,
            })
        if best_lrap==lrap_score:
            print('lrap_score improved saving checkpoint...')
            save_path = os.path.join('./', 'model'+str(i)+'.pt')
            if i>0:
                torch.save({
                'epoch': i,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler_lr.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'validation_accuracy': val_loss,
                'loss': loss,
                }, save_path)
                print('checkpoint saved to: {}'.format(save_path))

        # scheduler_cosine.step()
    print('Loading in wanddb')
    
    
    if not no_wandb:        
        model_artifact = wandb.Artifact('model'+str(uuid.uuid1()).replace("-",""), type='model')
        model_artifact.add_file(save_path)
        wandb.log_artifact(model_artifact)
        
        description_artifact = wandb.Artifact('description_model'+str(uuid.uuid1()).replace("-",""), type='python')
        description_artifact.add_file("/root/altegrad_project/code/model.py")
        description_artifact.add_file("/root/altegrad_project/code/train.py")
        description_artifact.add_file("/root/altegrad_project/code/data_loader.py")
        wandb.log_artifact(description_artifact)

    print('loading best model...')
    checkpoint = torch.load(save_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    graph_model = model.get_graph_encoder()
    text_model = model.get_text_encoder()

    idx_to_cid = test_cids_dataset.get_idx_to_cid()

    with torch.no_grad():
        graph_embeddings = []
        for batch in test_loader:
            for output in graph_model(batch.to(device_1)):
                graph_embeddings.append(output.tolist())

        text_embeddings = []
        for batch in test_text_loader:
            for output in text_model(batch['input_ids'].to(device_2), 
                                    attention_mask=batch['attention_mask'].to(device_2)):
                text_embeddings.append(output.tolist())

    similarity = cosine_similarity(text_embeddings, graph_embeddings)
    solution = pd.DataFrame(similarity)
    solution['ID'] = solution.index
    solution = solution[['ID'] + [col for col in solution.columns if col!='ID']]
    solution.to_csv('submission.csv', index=False)
    
    # Assuming graph_embeddings_array and text_embeddings_array are your numpy arrays
    # with shapes (n_samples, n_features) for graph and text embeddings respectively

    # Step 2: Apply UMAP to reduce dimensions separately
    reducer = umap.UMAP(random_state=42)


    reducer.fit(np.array(graph_embeddings))
    
    umap_graph = reducer.transform(np.array(graph_embeddings))
    umap_text = reducer.transform(np.array(text_embeddings))

    # Step 3: Generate a unique color for each pair of points
    # This creates a list of colors, one for each sample
    num_samples = 3301
    colors = plt.cm.rainbow(np.linspace(0, 1, umap_graph.shape[0]))

    plot_filename = f"umap_plot_{str(uuid.uuid4())}.png"
    plt.figure(figsize=(12, 8))
    plt.scatter(umap_graph[:, 0], umap_graph[:, 1], color=colors, alpha=0.5, label='Graph Embeddings')
    plt.scatter(umap_text[:, 0], umap_text[:, 1], color=colors, alpha=0.5, label='Text Embeddings')
    plt.legend()
    plt.title('UMAP Projection of Graph and Text Embeddings')
    plt.savefig(plot_filename)
    plt.close()  # Close the plot to free up memory

    # Step 2: Log the figure to wandb
    if not no_wandb:  # Assuming no_wandb is a variable that controls wandb logging
        # Log the image file as an artifact
        plot_artifact = wandb.Artifact('plot_artifact_' + str(uuid.uuid4()).replace("-", ""), type='plot')
        plot_artifact.add_file(plot_filename)
        wandb.log_artifact(plot_artifact)
        
        # Optionally, directly log the image under the current run (not as an artifact)
        wandb.log({"UMAP Projections": wandb.Image(plot_filename)})

    # Remember to delete the local file if it's no longer needed
    os.remove(plot_filename)
    
    # with torch.no_grad():
    #     # Compute and store graph embeddings
    #     graph_embeddings = []
    #     for graph_batch in test_loader:
    #         graph_batch = graph_batch.to(device_1)
    #         graph_proj = graph_model(graph_batch)
    #         graph_embeddings.extend(graph_proj.tolist())

    #     # Initialize similarity matrix
    #     similarity_matrix = []

    #     # Iterate over text batches
        
    #     i=0
    #     for text_batch in test_text_loader:
    #         print(len(text_batch))
    #         print(i)
    #         i+=1
    #         input_ids = text_batch['input_ids'].to(device_2)
    #         attention_mask = text_batch['attention_mask'].to(device_2)

    #         batch_similarity = []

    #         # Compute text embeddings for each graph batch and calculate similarity
    #         size = len(graph_batch)
    #         for idx, graph_batch in enumerate(test_loader):
    #             try:
    #                 graph_batch = graph_batch.to(device_1)
    #                 _, graph_latent = graph_model(graph_batch, with_latent=True)
    #                 text_x = text_model(input_ids, attention_mask, graph_batch, graph_latent)
    #             except:
    #                 graph_batch = graph_batch.to(device_1)
    #                 print(len(graph_batch))
    #                 print(input_ids.shape)
    #                 print(attention_mask.shape)
    #                 print(graph_latent.shape)
    #                 _, graph_latent = graph_model(graph_batch, with_latent=True)
    #                 text_x = text_model(input_ids[:-1, :], attention_mask[:-1, :], graph_batch, graph_latent)
                    


    #             # Calculate similarity with all graph embeddings
    #             similarity = cosine_similarity(text_x.detach().cpu(), graph_embeddings)
    #             batch_similarity.extend(similarity.tolist())

    #         # Add the batch similarities to the similarity matrix
    #         similarity_matrix.extend(batch_similarity)

    #     # Convert to DataFrame and save
    #     solution = pd.DataFrame(similarity_matrix)
    #     solution['ID'] = solution.index
    #     solution = solution[['ID'] + [col for col in solution.columns if col != 'ID']]
    #     solution.to_csv('submission.csv', index=False)

    
    if not no_wandb:
        
        submission_artifact = wandb.Artifact('submission'+str(uuid.uuid1()).replace("-",""), type='csv')
        submission_artifact.add_file('submission.csv')
        wandb.log_artifact(submission_artifact)

    # vizualise result on validation_set
    with torch.no_grad(): 
        graph_embeddings = []
        text_embeddings = []
        for batch in val_loader:
            for output in graph_model(batch.to(device_1)):
                graph_embeddings.append(output.tolist())
            for output in text_model(batch['input_ids'].to(device_2), 
                                    attention_mask=batch['attention_mask'].to(device_2)):
                text_embeddings.append(output.tolist())
                
    similarity = cosine_similarity(text_embeddings, graph_embeddings)
    solution = pd.DataFrame(similarity)
    solution['ID'] = solution.index
    solution = solution[['ID'] + [col for col in solution.columns if col!='ID']]
    solution.to_csv('validation_results.csv', index=False)
    
    if not no_wandb:
        validation_artifact = wandb.Artifact('validation_results'+str(uuid.uuid1()).replace("-",""), type='csv')
        validation_artifact.add_file('validation_results.csv')
        wandb.log_artifact(validation_artifact)
        wandb.finish()
