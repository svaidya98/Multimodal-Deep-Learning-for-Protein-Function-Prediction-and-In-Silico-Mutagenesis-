import json
import os
import scripts.config as config

def find_mutation_target(sequence):
    valuable_targets = ['W', 'Y', 'F', 'R', 'H']
    safe_start = int(len(sequence) * 0.1)
    safe_end = int(len(sequence) * 0.9)

    for i in range(safe_start, safe_end):
        residue = sequence[i]
        if residue in valuable_targets:
            return i, residue
    midpoint = len(sequence) // 2
    return midpoint, sequence[midpoint]

def alanine_mutation(sequence, index):
    residue_to_add = 'A'
    return sequence[:index] + residue_to_add + sequence[index+1:]

def alanine_scan():
    print('Starting Alanine Mutagenesis')
    candidates_path = config.results_dir / 'candidates.json'

    with open(candidates_path, 'r') as f:
        candidates = json.load(f)

    mutation_experiments = []

    for protein in candidates:
        wildtype_seq = protein['sequence']
        mutated_idx, original_nucleotide = find_mutation_target(wildtype_seq)
        mutated_seq = alanine_mutation(wildtype_seq, mutated_idx)

        mutation_name = f"{original_nucleotide}{mutated_idx + 1}A"
        print(f"Protein {protein['protein_id']} -> Mutating {mutation_name}")

        mutation_experiments.append({
            "protein_id": protein['protein_id'],
            "mutation": mutation_name,
            "wt_sequence": wildtype_seq,
            "mut_sequence": mutated_seq,
            "wt_confidence": protein['score']
        })
    out_path = config.results_dir / 'mutated_experiments.json'
    with open(out_path, 'w') as f:
        json.dump(mutation_experiments, f, indent=4)
        
    print(f'Saved {len(mutation_experiments)} mutated sequences to {out_path}\n')

if __name__ == "__main__":
    alanine_scan()
