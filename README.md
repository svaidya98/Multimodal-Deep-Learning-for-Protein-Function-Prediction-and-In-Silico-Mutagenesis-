# Multimodal-Deep-Learning-for-Protein-Function-Prediction-and-In-Silico-Mutagenesis-
Engineered an end-to-end, cloud-deployed deep learning pipeline to predict protein ontology, simulate targeted alanine mutations, and evaluate 3D structural degradation using protein language models
# CAFA 6: Protein Function Prediction & In Silico Mutagenesis Pipeline

## Overview
This repository contains an end-to-end deep learning pipeline designed to predict protein functions from raw amino acid sequences, simulate targeted biological attacks, and render the resulting 3D structural changes. 

Built using PyTorch and deployed on a Nebius H200 cloud GPU, this project leverages two state-of-the-art Protein Language Models (Meta's ESM-2 and Rostlab's ProtT5) to extract a 2323-dimensional feature space, achieving high accuracy in predicting Gene Ontology (GO) terms.

## Pipeline Architecture & Methodology

### 1. Dual-Language Feature Extraction
* **Language Models:** Utilizes `facebook/esm2_t33_650M_UR50D` and `Rostlab/prot_t5_xl_half_uniref50-enc`.
* **Data Processing:** Cleans and processes raw FASTA sequences into consolidated `train_final` and `test_final` datasets, merging the outputs of both LLMs into a unified 2323-feature vector.

### 2. Official CAFA Evaluation & Model Performance
To ensure robust, competition-standard validation, model performance was evaluated using the official Critical Assessment of Functional Annotation (CAFA) metrics, focusing on both protein-centric and term-centric scoring.

* **Protein-Centric Metrics:** Evaluated global predictive performance using the maximum F-measure (F-max) and minimum Semantic Distance (S-min).
* **Information Theoretic Metrics:** Analyzed the trade-off between Remaining Uncertainty (RU) and Misinformation (MI) using custom RU-MI curves.
* **Threshold Optimization:** Conducted rigorous threshold tuning to balance precision and recall, visualized through comprehensive learning curves and threshold optimization plots.


### 3. Automated In Silico Mutagenesis
* **Targeting:** Scans the test dataset for top-confidence wild-type proteins associated with specific biological pathways (e.g., GO:0006281 - DNA Repair).
* **Alanine Scanning:** Automatically identifies functional residues and performs targeted Alanine substitutions. 

**Mutagenesis Results (Functional Confidence Drop):**

| Mutation | Wild-Type Confidence | Mutated Confidence | Delta Drop |
| :--- | :--- | :--- | :--- |
| **R63A** | 0.6787 | 0.0430 | -0.6357 |
| **Y87A** | 0.6701 | 0.0650 | -0.6051 |
| **Y39A** | 0.6504 | 0.0458 | -0.6046 |
| **H53A** | 0.6474 | 0.0467 | -0.6007 |

### 4. 3D Structural Folding (ESMFold) & Visualization
* Utilizes Hugging Face's `EsmForProteinFolding` to translate sequence data into 3D Cartesian coordinates, outputting standard `.pdb` files.
* Structural alignments and RMSD calculations were performed using PyMOL to visually quantify the physical geometric warping caused by the *in silico* mutations.

## Repository Structure
* `/data` - Contains processed embeddings and final dataframes.
* `/scripts` - Modular Python scripts for data processing, training, and 3D generation.
* `/results` - Contains mutagenesis logs, the `mutagenesis_summary.csv` report, and all final `.pdb` generated structures. 
* `/figures` - PR curves, RU-MI curves, and threshold optimization visualizations.

## Technologies Used
* **Deep Learning:** PyTorch, Hugging Face Transformers
* **Bioinformatics:** Biopython, ESMFold, PyMOL
* **Cloud Infrastructure:** Nebius Cloud (H200 GPU), Linux SSH, Tmux
