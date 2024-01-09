import torch
from torch.utils.data import DataLoader, Dataset
from shared import TRAIN, VALIDATION, TEST, DATALOADER, BATCH_SIZE
from typing import List
import os
import os.path as osp
import torch
from torch_geometric.data import Dataset 
from torch_geometric.data import Data
from torch.utils.data import Dataset as TorchDataset
import pandas as pd


class WavesDataloader(Dataset):
    def __init__(
        self,
        length=64,
        n_samples=1000,
        device: str = "cuda",
        freeze: bool = False,
        freq_range: List[float] = [0, 10.],
        seed: int = 42,
    ):
        self.freeze = freeze
        self.device = device
        self.n_samples = n_samples
        if self.freeze:
            torch.manual_seed(seed)
            self.freqs = torch.linspace(freq_range[0], freq_range[1], n_samples, device=device)
            self.phases = torch.rand(n_samples, device=device)*2*torch.pi
        else:
            self.freq_range = torch.Tensor(freq_range).to(device)
        self.x = torch.linspace(0., 1., length, device=device)

    def __getitem__(self, index):
        if self.freeze:
            freq = self.freqs[index]
            phase = self.phases[index]
        else:
            freq = self.freq_range[0] + (self.freq_range[1]-self.freq_range[0])*torch.rand(1, device=self.device)
            phase = torch.rand(1, device=self.device)*2*torch.pi
        sig = torch.cos(freq*self.x+phase)
        return sig.unsqueeze(-2), freq.squeeze()

    def __len__(self):
        return self.n_samples


def get_dataloaders(config: dict, factor=1, device: str = "cuda"):
    dl_train = WavesDataloader(n_samples=800*factor, freeze=False, device=device)
    dl_valid = WavesDataloader(n_samples=100, freeze=True, device=device)
    dl_test = WavesDataloader(n_samples=100, freeze=True, freq_range=[0., 30.], device=device)  # Test generalization
    dl_dict = {
        TRAIN: DataLoader(dl_train, shuffle=True, batch_size=config[DATALOADER][BATCH_SIZE][TRAIN]),
        VALIDATION: DataLoader(dl_valid, shuffle=False, batch_size=config[DATALOADER][BATCH_SIZE][VALIDATION]),
        TEST: DataLoader(dl_test, shuffle=False, batch_size=config[DATALOADER][BATCH_SIZE][TEST])
    }
    return dl_dict


if __name__ == "__main__":
    config = {
        DATALOADER: {
            BATCH_SIZE: {
                TRAIN: 4,
                VALIDATION: 8,
                TEST: 8
            }
        }
    }
    import matplotlib.pyplot as plt
    dl_dict = get_dataloaders(config, factor=1)
    for run_index in range(2):
        for idx, mode in enumerate([TRAIN, VALIDATION, TEST]):
            signals, freqs = next(iter(dl_dict[mode]))
            plt.subplot(2, 3, run_index*3+1+idx)
            plt.plot(signals.cpu().numpy().T)
            plt.title(mode)
    plt.show()




########### HUGO ###########

class GraphTextDataset(Dataset):
    def __init__(self, root, gt, split, tokenizer=None, transform=None, pre_transform=None):
        self.root = root
        self.gt = gt
        self.split = split
        self.tokenizer = tokenizer
        self.description = pd.read_csv(os.path.join(self.root, split+'.tsv'), sep='\t', header=None)   
        self.description = self.description.set_index(0).to_dict()
        self.cids = list(self.description[1].keys())
        
        self.idx_to_cid = {}
        i = 0
        for cid in self.cids:
            self.idx_to_cid[i] = cid
            i += 1
        super(GraphTextDataset, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        return [str(cid) + ".graph" for cid in self.cids]

    @property
    def processed_file_names(self):
        return ['data_{}.pt'.format(cid) for cid in self.cids]
    
    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, 'raw')

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, 'processed/', self.split)

    def download(self):
        pass
        
    def process_graph(self, raw_path):
      edge_index  = []
      x = []
      with open(raw_path, 'r') as f:
        next(f)
        for line in f: 
          if line != "\n":
            edge = *map(int, line.split()), 
            edge_index.append(edge)
          else:
            break
        next(f)
        for line in f: #get mol2vec features:
          substruct_id = line.strip().split()[-1]
          if substruct_id in self.gt.keys():
            x.append(self.gt[substruct_id])
          else:
            x.append(self.gt['UNK'])
        return torch.LongTensor(edge_index).T, torch.FloatTensor(x)

    def process(self):
        i = 0        
        for raw_path in self.raw_paths:
            cid = int(raw_path.split('/')[-1][:-6])
            text_input = self.tokenizer([self.description[1][cid]],
                                   return_tensors="pt", 
                                   truncation=True, 
                                   max_length=256,
                                   padding="max_length",
                                   add_special_tokens=True,)
            edge_index, x = self.process_graph(raw_path)
            data = Data(x=x, edge_index=edge_index, input_ids=text_input['input_ids'], attention_mask=text_input['attention_mask'])

            torch.save(data, osp.join(self.processed_dir, 'data_{}.pt'.format(cid)))
            i += 1

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, 'data_{}.pt'.format(self.idx_to_cid[idx])))
        return data

    def get_cid(self, cid):
        data = torch.load(osp.join(self.processed_dir, 'data_{}.pt'.format(cid)))
        return data
    
    
class GraphDataset(Dataset):
    def __init__(self, root, gt, split, transform=None, pre_transform=None):
        self.root = root
        self.gt = gt
        self.split = split
        self.description = pd.read_csv(os.path.join(self.root, split+'.txt'), sep='\t', header=None)
        self.cids = self.description[0].tolist()
        
        self.idx_to_cid = {}
        i = 0
        for cid in self.cids:
            self.idx_to_cid[i] = cid
            i += 1
        super(GraphDataset, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        return [str(cid) + ".graph" for cid in self.cids]

    @property
    def processed_file_names(self):
        return ['data_{}.pt'.format(cid) for cid in self.cids]
    
    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, 'raw')

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, 'processed/', self.split)

    def download(self):
        pass
        
    def process_graph(self, raw_path):
      edge_index  = []
      x = []
      with open(raw_path, 'r') as f:
        next(f)
        for line in f: 
          if line != "\n":
            edge = *map(int, line.split()), 
            edge_index.append(edge)
          else:
            break
        next(f)
        for line in f:
          substruct_id = line.strip().split()[-1]
          if substruct_id in self.gt.keys():
            x.append(self.gt[substruct_id])
          else:
            x.append(self.gt['UNK'])
        return torch.LongTensor(edge_index).T, torch.FloatTensor(x)

    def process(self):
        i = 0        
        for raw_path in self.raw_paths:
            cid = int(raw_path.split('/')[-1][:-6])
            edge_index, x = self.process_graph(raw_path)
            data = Data(x=x, edge_index=edge_index)
            torch.save(data, osp.join(self.processed_dir, 'data_{}.pt'.format(cid)))
            i += 1

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, 'data_{}.pt'.format(self.idx_to_cid[idx])))
        return data

    def get_cid(self, cid):
        data = torch.load(osp.join(self.processed_dir, 'data_{}.pt'.format(cid)))
        return data
    
    def get_idx_to_cid(self):
        return self.idx_to_cid
    
class TextDataset(TorchDataset):
    def __init__(self, file_path, tokenizer, max_length=256):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.sentences = self.load_sentences(file_path)

    def load_sentences(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
        return [line.strip() for line in lines]

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = self.sentences[idx]

        encoding = self.tokenizer.encode_plus(
            sentence,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze()
        }
