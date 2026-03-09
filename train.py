import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, SubsetRandomSampler
from sklearn.model_selection import KFold
import pandas as pd
import numpy as np
import tqdm
import scripts.config as config
import scripts.data_loader as loader
import scripts.model as model_module

class Trainer:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'Hardware: {self.device}')

        self.full_dataset = loader.ProteinDataset(split = 'all_train')
        self.global_history = []

    def train_fold(self, fold_idx, train_indices, val_indices):
        print(f'Fold Number: {fold_idx+1}/5')
        
        train_sampler = SubsetRandomSampler(train_indices)
        val_sampler = SubsetRandomSampler(val_indices)
        
        train_loader = DataLoader(self.full_dataset, 
                                  batch_size = config.batch_size, 
                                  sampler = train_sampler)
        val_loader = DataLoader(self.full_dataset, 
                                batch_size = config.batch_size, 
                                sampler = val_sampler)
        
        model = model_module.ProteinDNN().to(self.device)
        
        optimizer = optim.AdamW(model.parameters(), lr = config.learning_rate, weight_decay = 1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                         mode = 'min', 
                                                         factor = 0.5, 
                                                         patience = 1)
        criterion = nn.BCEWithLogitsLoss() 
        
        best_val_loss = float('inf')

        history = {'train_loss': [], 'val_loss': [], 'f1_score': []}
        
        for epoch in range(config.epochs):
            model.train()
            train_loss = 0.0
            for inputs, taxon_ids, labels in train_loader:
                inputs, taxon_ids, labels = inputs.to(self.device), taxon_ids.to(self.device), labels.to(self.device)

                smooth_labels = labels * (1 - 0.1) + 0.05
                
                optimizer.zero_grad()
                outputs = model(inputs, taxon_ids)
                loss = criterion(outputs, smooth_labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                train_loss += loss.item()
            
            avg_train_loss = train_loss / len(train_loader)

            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for inputs, taxon_ids, labels in val_loader:
                    inputs, taxon_ids, labels = inputs.to(self.device), taxon_ids.to(self.device), labels.to(self.device)
                    outputs = model(inputs, taxon_ids)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
            
            avg_val_loss = val_loss / len(val_loader)
            scheduler.step(avg_val_loss)
            
            self.global_history.append({
                'fold': fold_idx + 1,
                'epoch': epoch + 1,
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss
            })
            
            print(f'Fold {fold_idx+1} | Epoch {epoch+1} | Train: {avg_train_loss:.4f} | Val: {avg_val_loss:.4f}')
            
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), config.results_dir / f'model_fold_{fold_idx}.pth')
                
        print(f'Fold {fold_idx+1} Completed. Best Loss: {best_val_loss:.4f}')

        if fold_idx == 0:
            pd.DataFrame(history).to_csv(config.results_dir / 'history.csv', index=False)

    def run_cross_validation(self, n_splits=5):
        kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        
        for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(self.full_dataset)):
            if config.debug and fold_idx > 0:
                print('DEBUG MODE: Skipping remaining folds.')
                break
            self.train_fold(fold_idx, train_idx, val_idx)

            if fold_idx == 0:
                self.generate_validation_predictions(val_idx)

        pd.DataFrame(self.global_history).to_csv(config.results_dir / 'history.csv', index=False)
        print('Global training history saved to history.csv')

    def generate_validation_predictions(self, val_indices):
        print('Generating Validation Predictions for Visualization')

        model = model_module.ProteinDNN().to(self.device)
        model.load_state_dict(torch.load(config.results_dir / 'model_fold_0.pth', map_location = self.device))
        model.eval()
        
        val_sampler = SubsetRandomSampler(val_indices)
        val_loader = DataLoader(self.full_dataset, 
                                batch_size = config.batch_size, 
                                sampler = val_sampler)
        
        all_probs = []
        all_targets = []
        
        with torch.no_grad():
            for inputs, taxon_ids, labels in tqdm.tqdm(val_loader, desc = 'Generating Viz Data'):
                inputs, taxon_ids = inputs.to(self.device), taxon_ids.to(self.device)
                outputs = model(inputs, taxon_ids)
                probs = torch.sigmoid(outputs).cpu().numpy()
                all_probs.append(probs)
                all_targets.append(labels.numpy())
                
        np.save(config.results_dir / 'val_probs.npy', np.concatenate(all_probs))
        np.save(config.results_dir / 'val_targets.npy', np.concatenate(all_targets))