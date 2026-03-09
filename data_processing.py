import os
import torch
import pandas as pd
import numpy as np
import json
from Bio import SeqIO
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from transformers import AutoTokenizer, AutoModel, T5Tokenizer, T5EncoderModel
import tqdm
import scripts.config as config

class DataProcessor():
    def __init__(self):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print("Loading ESM-2...")
        self.esm_tokenizer = AutoTokenizer.from_pretrained(config.esm_model_name)
        self.esm_model = AutoModel.from_pretrained(config.esm_model_name).to(self.device)
        self.esm_model.eval()

        print("Loading ProtT5...")
        self.t5_tokenizer = T5Tokenizer.from_pretrained(config.prott5_model_name, do_lower_case=False)
        self.t5_model = T5EncoderModel.from_pretrained(config.prott5_model_name).to(self.device)
        self.t5_model.eval()
        self.taxon_map = {'<UNK>': 0}
    
    def get_concatenated_embedding(self, sequence):
        with torch.no_grad():
            # --- ESM-2 PROCESSING ---
            esm_inputs = self.esm_tokenizer(sequence, return_tensors="pt", padding=True, truncation=True, max_length=1024)
            esm_inputs = {k: v.to(self.device) for k, v in esm_inputs.items()}
            esm_outputs = self.esm_model(**esm_inputs)
            # Get the mean pooling of the last hidden state
            esm_emb = esm_outputs.last_hidden_state.mean(dim=1).squeeze()

            # --- PROTT5 PROCESSING ---
            # ProtT5 requires sequences to be space-separated!
            t5_sequence = " ".join(list(sequence))
            t5_inputs = self.t5_tokenizer(t5_sequence, return_tensors="pt", padding=True, truncation=True, max_length=1024)
            t5_inputs = {k: v.to(self.device) for k, v in t5_inputs.items()}
            t5_outputs = self.t5_model(**t5_inputs)
            # Get the mean pooling
            t5_emb = t5_outputs.last_hidden_state.mean(dim=1).squeeze()

            # --- THE CONCATENATION ---
            # Glue the 1280 vector and the 1024 vector together into a 2304 vector
            combined_emb = torch.cat([esm_emb, t5_emb], dim=0)

            return combined_emb.cpu().numpy()

    def build_taxon_map(self):
        print('Building Taxonomy Map')
        train_tax = pd.read_csv(self.config.train_data / 'train_taxonomy.tsv', sep = '\t')
        test_tax = pd.read_csv(self.config.test_data / 'testsuperset-taxon-list.tsv', sep = '\t', encoding = 'ISO-8859-1')

        train_tax_ids = train_tax.iloc[:, 1].unique()
        test_tax_ids = test_tax.iloc[:, 0].unique()
        all_tax_ids = set(train_tax_ids) | set(test_tax_ids)

        for i, tax_id in enumerate(all_tax_ids, start=1):
            self.taxon_map[int(tax_id)] = i
            
        print(f'Found {len(self.taxon_map)} unique species.')

        os.makedirs(self.config.processed_dir, exist_ok=True)

        with open(self.config.processed_dir / 'taxon_map.json', 'w') as f:
            json.dump(self.taxon_map, f)

    def process_seqs(self, fasta_path, taxon_path, save_name):
        print(f"Loading Taxonomy for {save_name}...")
        
        prot_to_tax = {}
        if save_name == 'train':
            df_tax = pd.read_csv(taxon_path, sep='\t')
            prot_to_tax = dict(zip(df_tax.iloc[:, 0], df_tax.iloc[:, 1]))

        sequences = list(SeqIO.parse(fasta_path, 'fasta'))
        if self.config.debug:
            sequences = sequences[:self.config.debug_size]

        embeddings = []
        ids = []
        taxon_indices = []
        
        for record in tqdm.tqdm(sequences, desc=f"Processing {save_name}"):
            seq = str(record.seq)
            if len(seq) > 1022:
                vector = self.embed_long_sequence(seq)
            else:
                vector = self.embed_short_sequence(seq)
            
            p_id = record.id
            if '|' in p_id:
                p_id = p_id.split('|')[1]
            
            raw_tax = None
            if save_name == 'train':
                raw_tax = prot_to_tax.get(p_id, None)
            elif save_name == 'test':
                parts = record.description.split()
                if len(parts) >= 2:
                    raw_tax = parts[1]
            
            if raw_tax is not None and int(raw_tax) in self.taxon_map:
                tax_idx = self.taxon_map[int(raw_tax)]
            else:
                tax_idx = 0 
            
            clean_seq = seq.replace('X', '').replace('U', '').replace('O', '').replace('B', '').replace('Z', '')
            if len(clean_seq) > 0:
                analysis = ProteinAnalysis(clean_seq)
                mw = analysis.molecular_weight() / 100000.0  # Normalize to ~[0, 1]
                iso = analysis.isoelectric_point() / 14.0    # Normalize to [0, 1]
            else:
                mw, iso = 0.5, 0.5 
                
            length_norm = len(seq) / 1000.0 # Normalize length
            
            meta_vector = np.array([length_norm, mw, iso], dtype=np.float32)
            
            final_combined_vector = np.concatenate([vector, meta_vector])
            
            embeddings.append(final_combined_vector)
            ids.append(p_id)
            taxon_indices.append(tax_idx)
            
        os.makedirs(self.config.processed_dir, exist_ok=True)
        np.save(self.config.processed_dir / f'{save_name}_embeddings.npy', np.array(embeddings))
        np.save(self.config.processed_dir / f'{save_name}_ids.npy', np.array(ids))
        np.save(self.config.processed_dir / f'{save_name}_taxon_ids.npy', np.array(taxon_indices))
    
    def embed_short_sequence(self, seq):
        return self.get_concatenated_embedding(seq)
    
    def embed_long_sequence(self, seq):
        windows = []
        slice_length = 1022
        stride = 512
        for i in range(0, len(seq), stride):
            chunk = seq[i : i + slice_length]
            seq_slice = self.get_concatenated_embedding(chunk)
            windows.append(seq_slice)
        stacked_vectors = np.stack(windows)
        final_vector = np.mean(stacked_vectors, axis=0)
        return final_vector
    
    def run(self):
        self.build_taxon_map()
        print('Processing Training Data')
        self.process_seqs(
            self.config.train_data / 'train_sequences.fasta',
            self.config.train_data / 'train_taxonomy.tsv',
            'train')

        print('Processing Test Data')
        self.process_seqs(
            self.config.test_data / 'testsuperset.fasta',
            self.config.test_data / 'testsuperset-taxon-list.tsv',
            'test')








