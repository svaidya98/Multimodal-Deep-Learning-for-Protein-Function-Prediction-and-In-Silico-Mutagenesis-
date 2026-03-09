import os
import json
import torch
import numpy as np
from transformers import AutoTokenizer, EsmForProteinFolding
import scripts.config as config
from transformers.models.esm.openfold_utils.protein import to_pdb, Protein as OFProtein
from transformers.models.esm.openfold_utils.feats import atom14_to_atom37

def get_pdb_from_sequence(model, tokenizer, sequence, device):

    inputs = tokenizer([sequence], return_tensors = 'pt', add_special_tokens = False)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
    final_atom_positions = atom14_to_atom37(outputs['positions'][-1], outputs)
    outputs = {k: v.to('cpu').numpy() for k, v in outputs.items()}
    final_atom_positions = final_atom_positions.cpu().numpy()
    final_atom_mask = outputs['atom37_atom_exists']
    aa = outputs['aatype'][0]
    pred_pos = final_atom_positions[0]
    mask = final_atom_mask[0]
    residue_index = outputs['residue_index'][0] + 1
    plddt = outputs['plddt'][0]
    if plddt.ndim == 1:
        b_factors = plddt[:, None] * mask
    else:
        b_factors = plddt * mask
        
    b_factors = b_factors.astype(float) * 100.0
    protein = OFProtein(
        aatype=aa,
        atom_positions=pred_pos,
        atom_mask=mask,
        residue_index=residue_index,
        b_factors=b_factors,
        chain_index=np.zeros_like(residue_index)
    )
    pdb_string = to_pdb(protein)
    
    return pdb_string


def run_3d_folding():
    print('3D Structure Modeling (ESMFold)')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model_name = 'facebook/esmfold_v1'
    print(f'Loading {model_name}')

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = EsmForProteinFolding.from_pretrained(model_name)
    model.to(device)
    model.eval()

    results_path = config.results_dir / 'delta_results.json'
    with open(results_path, 'r') as f:
        experiments = json.load(f)
        
    print(f'Generating 3D models for {len(experiments)} experiments')
    
    pdb_dir = config.results_dir / 'pdb_files'
    os.makedirs(pdb_dir, exist_ok=True)
    
    for exp in experiments:
        p_id = exp['protein_id']
        mutation = exp['mutation']
        wt_seq = exp['wt_sequence'] # You might need to make sure this key was saved in Phase 3!
        mut_seq = exp['mut_sequence']
        
        print(f'Folding {p_id} (WT and {mutation})')
        
        wt_pdb_string = get_pdb_from_sequence(model, tokenizer, wt_seq, device)
        mut_pdb_string = get_pdb_from_sequence(model, tokenizer, mut_seq, device)
        
        wt_path = pdb_dir / f'{p_id}_WT.pdb'
        mut_path = pdb_dir / f'{p_id}_{mutation}.pdb'
        
        with open(wt_path, 'w') as f:
            f.write(wt_pdb_string)
            
        with open(mut_path, 'w') as f:
            f.write(mut_pdb_string)

    print(f'\nPhase 4 Complete. All PDB files saved to {pdb_dir}')

if __name__ == "__main__":
    run_3d_folding()