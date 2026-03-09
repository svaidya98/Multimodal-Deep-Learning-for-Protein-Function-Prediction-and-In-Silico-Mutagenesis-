import scripts.config as config
import scripts.mutation_candidates as phase1
import scripts.mutate_candidates as phase2
import scripts.calculate_delta as phase3
import scripts.generate_3d_model as phase4

if __name__ == "__main__":
    print(f'IN SILICO MUTAGENESIS & 3D FOLDING PIPELINE')

    TARGET_FUNCTION = 'DNA repair' 
    phase1.run_candidate_selection(target_function=TARGET_FUNCTION, top_n=10)
    phase2.alanine_scan()
    phase3.run_delta_calc()
    phase4.run_3d_folding()

    print('PIPELINE COMPLETE')
