import os
import json
import numpy as np
import pandas as pd
from Bio import SeqIO
import scripts.config as config

def get_id_from_name(obo_path, target_name):
    print(f'Searching for GO ID for target: {target_name}')
    current_id = None
    with open(obo_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('id: '):
                current_id = line.split('id: ')[1]
            elif line.startswith('name: '):
                name = line.split('name: ')[1]
                if name == target_name:
                    print(f'Found GO ID: {current_id}')
                    return current_id
    raise ValueError(f"Could not find a GO term named '{target_name}' in the OBO file.")

def get_target_index(target_go_id):
    print(f"Locating column index for {target_go_id}...")
    train_terms = pd.read_csv(config.train_data / 'train_terms.tsv', sep='\t')
    term_counts = train_terms['term'].value_counts()
    target_labels = term_counts.index[:config.num_labels].tolist()

    if target_go_id in target_labels:
        target_index = target_labels.index(target_go_id)
    else:
        raise ValueError(f"Target {target_go_id} is not in the top 1500 trained labels!")

    return target_index

def get_top_candidate_indices(probs_matrix_path, target_col_idx, top_n = 10):
    print(f"Mining probabilities for the top {top_n} candidates...")
    probs = np.load(probs_matrix_path)

    target_col = probs[:, target_col_idx]
    top_indices = np.argsort(target_col)[::-1][:top_n]
    top_scores = target_col[top_indices]

    return top_indices, top_scores

def get_candidate_seqs(top_indices, top_scores, ids_matrix_path, fasta_path):
    protein_ids = np.load(ids_matrix_path)
    candidate_ids = protein_ids[top_indices]

    def extract_core_id(raw_header):
        if '|' in raw_header:
            return raw_header.split('|')[1]
        return raw_header
    fasta_dict = SeqIO.index(str(fasta_path), "fasta", key_function=extract_core_id)

    candidates_list = []

    zip_data = zip(candidate_ids, top_scores)
    candidate_dict = {}

    for p_id, score in zip_data:
        seq_record = fasta_dict[p_id]
        seq = str(seq_record.seq)
        candidates_list.append({
            "protein_id": p_id,
            "sequence": seq,
            "score": float(score)})
    return candidates_list

def run_candidate_selection(target_function = 'DNA repair', top_n = 10):
    obo_path = config.data_dir / 'go-basic.obo'
    target_go_id = get_id_from_name(obo_path, target_function)
    target_col_idx = get_target_index(target_go_id)
    probs_path = config.results_dir / 'val_probs.npy'
    top_idx, top_scores = get_top_candidate_indices(probs_path, target_col_idx, top_n)
    ids_path = config.processed_dir / 'train_ids.npy'
    fasta_path = config.train_data / 'train_sequences.fasta'
    final_candidates = get_candidate_seqs(top_idx, top_scores, ids_path, fasta_path)
    out_path = config.results_dir / 'candidates.json'
    with open(out_path, 'w') as f:
        json.dump(final_candidates, f, indent=4)
    print(f"Saved {len(final_candidates)} candidates to {out_path}\n")

if __name__ == "__main__":
    run_candidate_selection()


    
        
