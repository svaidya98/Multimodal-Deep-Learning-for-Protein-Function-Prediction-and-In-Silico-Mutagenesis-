import os
import sys
import torch
import numpy as np
import pandas as pd
import tqdm
from datetime import datetime
import scripts.config as config
import scripts.data_loader as loader
import scripts.model as model_module
import scripts.visualize as visualize
from cafaeval import evaluation

def build_true_path_dictionary(obo_path):
    print("Building Gene Ontology DAG for True Path Rule...")
    term_parents = {}
    current_term = None
    
    with open(obo_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith("id: GO:"):
                current_term = line.split("id: ")[1]
                if current_term not in term_parents:
                    term_parents[current_term] = []
            elif line.startswith("is_a: GO:"):
                parent_term = line.split("is_a: ")[1].split(" !")[0]
                if current_term:
                    term_parents[current_term].append(parent_term)
                    
    return term_parents

class CAFA_Evaluator_Bridge:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.obo_path = config.data_dir / 'go-basic.obo'
        if not self.obo_path.exists():
            print(f"❌ ERROR: Missing {self.obo_path}!")
            sys.exit(1)

        # Load Ensemble
        self.models = []
        num_folds = 1 if config.debug else 5
        print(f"Loading {num_folds} Ensemble Models...")
        
        for i in range(num_folds):
            path = config.results_dir / f'model_fold_{i}.pth'
            if path.exists():
                m = model_module.ProteinDNN()
                m.load_state_dict(torch.load(path, map_location=self.device))
                m.to(self.device)
                m.eval()
                self.models.append(m)
                
        if len(self.models) == 0:
            print("❌ No models loaded. Check your results directory.")
            sys.exit(1)
            
    def run_evaluation(self):
        print("Generating Predictions for Official Metrics...")
        val_dataset = loader.ProteinDataset(split='validation')
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
        
        term_list = val_dataset.target_labels
        current_idx = 0
        
        # Setup directories
        cafa_pred_dir = config.results_dir / 'cafa_preds'
        os.makedirs(cafa_pred_dir, exist_ok=True)
        
        # Clean up old predictions to avoid reading duplicates
        for f in cafa_pred_dir.glob("*.txt"):
            f.unlink()
            
        pred_file_path = cafa_pred_dir / 'ensemble_preds.txt'
        gt_file_path = config.results_dir / 'cafa_gt.txt'
        
        print("Writing predictions and ground truth to disk for CAFA Evaluator...")
        with open(pred_file_path, 'w') as f_pred, open(gt_file_path, 'w') as f_gt:
            with torch.no_grad():
                for inputs, taxon_ids, labels in tqdm.tqdm(val_loader):
                    inputs, taxon_ids = inputs.to(self.device), taxon_ids.to(self.device)
                    
                    avg_probs = torch.zeros((inputs.shape[0], config.num_labels)).to(self.device)
                    for m in self.models:
                        avg_probs += torch.sigmoid(m(inputs, taxon_ids))
                    avg_probs /= len(self.models)
                    avg_probs = avg_probs.cpu().numpy()
                    targets = labels.cpu().numpy()
                    
                    idx_to_term = {i: t for i, t in enumerate(term_list)}
                    term_to_idx = {t: i for i, t in enumerate(term_list)}
                    
                    # 2. Build the DAG (only need to do this once, but placed here for logic flow)
                    if not hasattr(self, 'go_dag'):
                        self.go_dag = build_true_path_dictionary(self.obo_path)
                    
                    # 3. Propagate probabilities bottom-up
                    for i in range(inputs.shape[0]):
                        # Get all terms the model predicted > 0.01 confidence
                        predicted_indices = np.where(avg_probs[i] > 0.01)[0]
                        predicted_terms = [idx_to_term[idx] for idx in predicted_indices]
                        
                        # Sort by highest probability to process strongest signals first
                        predicted_terms.sort(key=lambda t: avg_probs[i][term_to_idx[t]], reverse=True)
                        
                        for term in predicted_terms:
                            child_prob = avg_probs[i][term_to_idx[term]]
                            
                            # Recursively push this probability to all parents
                            parents_to_visit = self.go_dag.get(term, [])
                            visited = set()
                            
                            while parents_to_visit:
                                p = parents_to_visit.pop(0)
                                if p in visited: continue
                                visited.add(p)
                                
                                # If the parent is in our 1500 target labels, update it!
                                if p in term_to_idx:
                                    p_idx = term_to_idx[p]
                                    # P(parent) = max(P(parent), P(child))
                                    if child_prob > avg_probs[i][p_idx]:
                                        avg_probs[i][p_idx] = child_prob
                                        
                                # Add the parent's parents to the queue
                                parents_to_visit.extend(self.go_dag.get(p, []))

                    for i in range(inputs.shape[0]):
                        p_id = val_dataset.ids[current_idx + i]
                        
                        # Ground Truth (ProteinID  GO_ID)
                        true_indices = np.where(targets[i] == 1)[0]
                        for idx in true_indices:
                            f_gt.write(f"{p_id}\t{term_list[idx]}\n")
                        
                        # Predictions (ProteinID  GO_ID  Score)
                        pred_indices = np.where(avg_probs[i] > 0.01)[0]
                        for idx in pred_indices:
                            f_pred.write(f"{p_id}\t{term_list[idx]}\t{avg_probs[i][idx]:.4f}\n")
                    
                    current_idx += inputs.shape[0]
                    if config.debug and current_idx >= config.debug_size: break

        print("Calculating Official CAFA Metrics using CAFA-evaluator-PK...")
        ia_path = config.data_dir / 'IA.tsv'
        if not ia_path.exists():
            print("⚠️ IA.tsv not found. S-min will be unweighted.")
            ia_path = None
        else:
            ia_path = str(ia_path)

        try:
            df_all, dfs_best = evaluation.cafa_eval(
                obo_file=str(self.obo_path),
                pred_dir=str(cafa_pred_dir),
                gt_file=str(gt_file_path),
                ia=ia_path
            )
            
            if dfs_best:
                print("\n" + "="*40)
                print("🏆 FINAL CAFA METRICS 🏆")
                print("="*40)
                for metric, df in dfs_best.items():
                    if metric in ['f', 's']:
                        for (filename, ns, tau), row in df.iterrows():
                            val = row[metric]
                            print(f"Ontology: {ns:4s} | {metric.upper()}: {val:.4f} (Optimal Threshold: {tau:.2f})")
                print("="*40 + "\n")
                
                # --- ROBUST RU-MI CURVE EXTRACTION ---
                try:
                    if df_all is not None:
                        ru_col = 'ru_w' if 'ru_w' in df_all.columns else 'ru'
                        mi_col = 'mi_w' if 'mi_w' in df_all.columns else 'mi'
                        
                        if ru_col in df_all.columns and mi_col in df_all.columns:
                            namespaces = df_all.index.get_level_values('ns').unique()
                            if len(namespaces) > 0:
                                first_ns = namespaces[0]
                                ns_df = df_all.xs(first_ns, level='ns')
                                
                                filenames = ns_df.index.get_level_values('filename').unique()
                                if len(filenames) > 0:
                                    final_df = ns_df.xs(filenames[0], level='filename')
                                    np.save(config.results_dir / 'curve_ru_mi.npy', 
                                            np.stack([final_df[ru_col].values, final_df[mi_col].values]))
                                    print(f"✅ Saved RU-MI curve data for {first_ns} ({ru_col}, {mi_col})")
                except Exception as e_curve:
                    print(f"⚠️ Could not extract RU-MI curve data: {e_curve}")
                    
                # --- MASTER EXPERIMENT LOGGING ---
                print("\n📝 Saving run summary to experiment_log.csv...")
                log_data = {'Timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                
                # Grab training metrics
                try:
                    hist_df = pd.read_csv(config.results_dir / 'history.csv')
                    for fold_idx in hist_df['fold'].unique():
                        min_val = hist_df[hist_df['fold'] == fold_idx]['val_loss'].min()
                        log_data[f'Fold_{fold_idx}_Min_Val_Loss'] = round(min_val, 4)
                except Exception as e:
                    print(f"⚠️ Could not read history.csv for logging: {e}")

                # Grab final CAFA metrics
                for metric, df in dfs_best.items():
                    if metric in ['f', 's']:
                        for (filename, ns, tau), row in df.iterrows():
                            log_data[f'{ns}_{metric.upper()}'] = round(row[metric], 4)
                            if metric == 'f':
                                log_data[f'{ns}_Opt_Tau'] = round(tau, 2)
                
                # Append to or create the master log
                log_path = config.results_dir / 'experiment_log.csv'
                log_df = pd.DataFrame([log_data])
                
                if log_path.exists():
                    log_df.to_csv(log_path, mode='a', header=False, index=False)
                else:
                    log_df.to_csv(log_path, mode='w', header=True, index=False)
                    
                print(f"✅ Run successfully logged to {log_path.name}")

            else:
                print("⚠️ Evaluation completed but returned no best scores.")
        except Exception as e:
            print(f"❌ Official Evaluation Failed: {e}")

if __name__ == "__main__":
    tester = CAFA_Evaluator_Bridge()
    tester.run_evaluation()
    
    print("\nGenerating Final CAFA Plots...")
    viz = visualize.ResultsVisualizer()
    viz.plot_term_centric_performance() 
    viz.plot_ru_mi_curve()