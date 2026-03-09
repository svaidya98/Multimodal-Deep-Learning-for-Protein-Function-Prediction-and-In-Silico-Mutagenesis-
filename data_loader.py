import torch
import numpy as np
import pandas as pd
import scripts.config as config
import torch.utils.data.dataset

class ProteinDataset(torch.utils.data.Dataset):
    def __init__(self, split = 'train'):
        self.split = split

        if split == 'test':
            self.embeddings = np.load(config.processed_dir / 'test_embeddings.npy')
            self.ids = np.load(config.processed_dir / 'test_ids.npy')
            self.taxons = np.load(config.processed_dir / f'test_taxon_ids.npy')
            if config.debug:
                print(f'DEBUG MODE: Loading only {config.debug_size} samples')
                self.embeddings = self.embeddings[:config.debug_size]
                self.ids = self.ids[:config.debug_size]
            self.labels = np.zeros((len(self.ids), config.num_labels), dtype='float32')
            return

        self.embeddings = np.load(config.processed_dir / 'train_embeddings.npy')
        self.ids = np.load(config.processed_dir / 'train_ids.npy')
        self.taxons = np.load(config.processed_dir / f'train_taxon_ids.npy')


        #Randomized Train Test split
        indices = np.arange(len(self.ids))
        np.random.seed(42) 
        np.random.shuffle(indices)
        self.ids = self.ids[indices]
        self.embeddings = self.embeddings[indices]
        self.taxons = self.taxons[indices]

        limit = int(0.9 * len(self.ids))
        
        if split == 'train':
            self.ids = self.ids[:limit]
            self.embeddings = self.embeddings[:limit]
            self.taxons = self.taxons[:limit]
        elif split == 'validation':
            self.ids = self.ids[limit:]
            self.embeddings = self.embeddings[limit:]
            self.taxons = self.taxons[limit:]
    
        train_terms = pd.read_csv(config.train_data / 'train_terms.tsv', sep='\t')
        term_counts = train_terms['term'].value_counts()
        self.target_labels = term_counts.index[:config.num_labels].tolist()
        self.term_to_index = {term: i for i, term in enumerate(self.target_labels)}

        self.labels = np.zeros((len(self.ids), config.num_labels), dtype='float32')
        id_to_terms = train_terms.groupby('EntryID')['term'].apply(set).to_dict()
        for i, protein_id in enumerate(self.ids):
            if protein_id in id_to_terms:
                terms = id_to_terms[protein_id]
                for term in terms:
                    if term in self.term_to_index:
                        col_idx = self.term_to_index[term]
                        self.labels[i, col_idx] = 1
    
    def __len__(self):
        return len(self.ids)
    
    def __getitem__(self, index):
        embedding = self.embeddings[index]
        label = self.labels[index]
        taxon = self.taxons[index]
        embedding = torch.tensor(embedding, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.float32)
        taxon = torch.tensor(taxon, dtype=torch.long)
        return (embedding, taxon, label)





