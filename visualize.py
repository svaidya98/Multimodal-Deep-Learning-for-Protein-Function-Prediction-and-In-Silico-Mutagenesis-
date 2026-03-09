import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_curve, average_precision_score, f1_score
import scripts.config as config

sns.set_style('whitegrid')
plt.rcParams.update({'font.size': 12, 'figure.dpi': 300})

class ResultsVisualizer:
    def __init__(self):
        self.results_dir = config.results_dir
        
    def plot_learning_curves(self):
        print('Generating Figure 1: Learning Curves')
        if not (self.results_dir / 'history.csv').exists(): return
        df = pd.read_csv(self.results_dir / 'history.csv')
        
        plt.figure(figsize=(10, 6))
        plt.plot(df['train_loss'], label='Train Loss', linestyle='--')
        plt.plot(df['val_loss'], label='Val Loss', linewidth=2)
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training Dynamics (Fold 1)')
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.results_dir / 'figure1_learning_curves.png')
        plt.close()

    def plot_precision_recall(self):
        print('Generating Figure 2: Precision-Recall Curve')
        if not (self.results_dir / 'val_probs.npy').exists(): return
        probs = np.load(self.results_dir / 'val_probs.npy')
        targets = np.load(self.results_dir / 'val_targets.npy')
        
        precision, recall, _ = precision_recall_curve(targets.ravel(), probs.ravel())
        auprc = average_precision_score(targets, probs, average="micro")
        
        plt.figure(figsize=(8, 8))
        plt.plot(recall, precision, color='purple', lw=3, label=f'Micro-Avg (AUPRC={auprc:.3f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Global Precision-Recall')
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.results_dir / 'figure2_pr_curve.png')
        plt.close()

    def plot_threshold_optimization(self):
        print('Generating Figure 3: Threshold Optimization')
        if not (self.results_dir / 'val_probs.npy').exists(): return
        probs = np.load(self.results_dir / 'val_probs.npy')
        targets = np.load(self.results_dir / 'val_targets.npy')
        
        thresholds = np.arange(0.1, 0.95, 0.05)
        f1_scores = [f1_score(targets, (probs > t).astype(int), average='micro') for t in thresholds]
        
        plt.figure(figsize=(10, 6))
        plt.plot(thresholds, f1_scores, marker='o', color='green')
        plt.xlabel('Threshold')
        plt.ylabel('F1 Score')
        plt.title('Threshold Optimization')
        plt.tight_layout()
        plt.savefig(self.results_dir / 'figure3_threshold_opt.png')
        plt.close()

    def plot_term_centric_performance(self):
        print('Generating Figure 4: Term-Centric Performance')
        if not (self.results_dir / 'val_probs.npy').exists(): return
        targets = np.load(self.results_dir / 'val_targets.npy')
        probs = np.load(self.results_dir / 'val_probs.npy')
        
        term_freqs = targets.sum(axis=0)
        valid_indices = term_freqs > 0
        targets = targets[:, valid_indices]
        probs = probs[:, valid_indices]
        term_freqs = term_freqs[valid_indices]
        
        preds = (probs > 0.4).astype(int)
        f1_per_term = [f1_score(targets[:, i], preds[:, i], zero_division=0) for i in range(targets.shape[1])]
        
        df = pd.DataFrame({'Frequency': term_freqs, 'F1': f1_per_term})
        bins = [0, 50, 500, float('inf')]
        labels = ['Rare', 'Common', 'Frequent']
        df['Category'] = pd.cut(df['Frequency'], bins=bins, labels=labels)
        
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='Category', y='F1', data=df)
        plt.title('Performance by Term Frequency')
        plt.tight_layout()
        plt.savefig(self.results_dir / 'figure4_term_centric.png')
        plt.close()

    def plot_ru_mi_curve(self):
        print('Generating Figure 5: RU-MI Curve')
        path = self.results_dir / 'curve_ru_mi.npy'
        if not path.exists(): 
            print("⚠️ RU-MI curve data not found! Skipping Figure 5.")
            return
            
        data = np.load(path)
        plt.figure(figsize=(8, 8))
        plt.plot(data[0], data[1], marker='.', color='teal', label='Model Performance')

        radii = [10, 20, 30, 40]
        theta = np.linspace(0, np.pi/2, 100)
        for r in radii:
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            plt.plot(x, y, 'k:', alpha=0.3)
            plt.text(x[50], y[50], f'S={r}', fontsize=8)
            
        plt.xlabel('Remaining Uncertainty (RU)')
        plt.ylabel('Missing Information (MI)')
        plt.title('Semantic Distance (RU-MI)')
        plt.axis('equal')
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.results_dir / 'figure5_ru_mi_curve.png')
        plt.close()
        print("✅ Figure 5 Saved!")