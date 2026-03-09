import json
import torch
import numpy as np
from Bio.SeqUtils.ProtParam import ProteinAnalysis
import tqdm
import scripts.config as config
from scripts.data_processing import DataProcessor
from scripts.model import ProteinDNN
from scripts.mutation_candidates import get_id_from_name, get_target_index

def calculate_metadata(sequence):
    clean_seq = sequence.replace('X', '').replace('U', '').replace('O', '').replace('B', '').replace('Z', '')
    if len(clean_seq) > 0:
        analysis = ProteinAnalysis(clean_seq)
        mw = analysis.molecular_weight() / 100000.0  
        iso = analysis.isoelectric_point() / 14.0    
    else:
        mw, iso = 0.5, 0.5 
        
    length_norm = len(sequence) / 1000.0 
    return np.array([length_norm, mw, iso], dtype=np.float32)

def get_mutated_embeddings(processor, sequence):
    plm_vector = processor.get_concatenated_embedding(sequence)
    meta_vector = calculate_metadata(sequence)
    final_vector = np.concatenate([plm_vector, meta_vector])
    return final_vector

def run_delta_calc(target_function = 'DNA repair'):
    print(f"--- STARTING PHASE 3: CALCULATING FUNCTIONAL DELTA FOR '{target_function}' ---")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    obo_path = config.data_dir / 'go-basic.obo'
    target_go_id = get_id_from_name(obo_path, target_function)
    target_col_idx = get_target_index(target_go_id)
    processor = DataProcessor()
    model = ProteinDNN().to(device)
    model.load_state_dict(torch.load(config.results_dir / 'model_fold_0.pth', map_location=device))
    model.eval()
    experiments_path = config.results_dir / 'mutated_experiments.json'
    with open(experiments_path, 'r') as f:
        experiments = json.load(f)
    print(f"Analyzing {len(experiments)} mutations...")
    
    results = []

    with torch.no_grad():
       for exp in experiments:
            mut_seq = exp['mut_sequence']
            wt_conf = exp['wt_confidence'] 
            mutant_emb = get_mutated_embeddings(processor, mut_seq)
            inputs = torch.tensor(mutant_emb).unsqueeze(0).to(device)
            taxon = torch.tensor([1], dtype=torch.long).to(device)
            outputs = model(inputs, taxon)
            probs = torch.sigmoid(outputs).cpu().numpy()
            mut_conf = probs[0][target_col_idx]
            delta = mut_conf - wt_conf
            exp['mut_confidence'] = float(mut_conf)
            exp['delta'] = float(delta)
            results.append(exp)
            print(f"{exp['mutation']}: WT = {wt_conf:.4f} | Mutated = {mut_conf:.4f} | Delta = {delta:.4f}")
    
    out_path = config.results_dir / 'delta_results.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Saved {len(results)} delta results to {out_path}\n")

if __name__ == "__main__":
    run_delta_calc()          


