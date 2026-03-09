import scripts.config as config
import scripts.mutation_candidates as phase1
import scripts.mutate_candidates as phase2
import scripts.calculate_delta as phase3
import scripts.generate_3d_model as phase4

if __name__ == "__main__":
    print(f"==================================================")
    print(f"   IN SILICO MUTAGENESIS & 3D FOLDING PIPELINE    ")
    print(f"==================================================\n")
    
    # The biological function you want to target
    TARGET_FUNCTION = "DNA repair" 
    
    # Phase 1: Hunt down the Top 10 most confident wild-type proteins
    phase1.run_candidate_selection(target_function=TARGET_FUNCTION, top_n=10)
    
    # Phase 2: Perform the in silico Alanine Scan on those 10 targets
    phase2.alanine_scan()
    
    # Phase 3: (Placeholder for recalculating the GO probabilities)
    phase3.run_delta_calc()
    
    # Phase 4: (Placeholder for ESMFold .pdb generation)
    phase4.run_3d_folding()

    print("PIPELINE COMPLETE")
