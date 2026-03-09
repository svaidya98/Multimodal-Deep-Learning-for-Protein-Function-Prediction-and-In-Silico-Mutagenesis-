import scripts.config as config
import scripts.data_processing as data_processing
import scripts.train as train
import scripts.visualize as visualize
import os

if __name__ == "__main__":
    print(f'STARTING PIPELINE (DEBUG={config.debug})')

    if not os.path.exists('data/processed/train_embeddings.npy'):
        # 1. Process Data
        print('STEP 1: Data Processing')
        processor = data_processing.DataProcessor()
        processor.run()
    
    # 2. Train Models (Cross Validation)
    print('STEP 2: Training (5-Fold CV)')
    trainer = train.Trainer()
    trainer.run_cross_validation()
    
    # 3. Visualize Training Results
    print('STEP 3: Visualizing Training & Validation')
    viz = visualize.ResultsVisualizer()
    viz.plot_learning_curves()        # Figure 1
    viz.plot_precision_recall()       # Figure 2
    viz.plot_threshold_optimization() # Figure 3
    
    print('MAIN PIPELINE COMPLETE.')
    print('Run test_cafa.py for final official metrics.')