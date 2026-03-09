import os
from pathlib import Path

debug = False
debug_size = 50

#Project Root Logic
project_root = Path(__file__).resolve().parent.parent

#Data Directories
data_dir = project_root / 'data'
train_data = data_dir / 'train'
test_data = data_dir / 'test'
processed_dir = data_dir / 'processed'
results_dir = project_root / 'results'

#Model Hyperparameters
esm_model_name = 'facebook/esm2_t33_650M_UR50D'
esm_dim = 1280
prott5_model_name = 'Rostlab/prot_t5_xl_half_uniref50-enc'
prott5_dim = 1024
batch_size = 8 if debug else 32
learning_rate = 1e-3
epochs = 2 if debug else 20 
embed_dim = esm_dim + prott5_dim + 3

#Ontology Settings
num_labels = 1500
taxon_embed_dim = 16
