import os
import numpy as np
import pathlib
from goatools.obo_parser import GODag
from goatools.gosubdag.gosubdag import GoSubDag
from goatools.godag.go_tasks import get_go2parents
from goatools.base import get_go_dag
import scripts.config as cfg
import collections

class OntologyEngine:
    def __init__(self, ontology_path = None):
        if ontology_path is None:
            self.ontology_path = cfg.RAW_DIR / 'go-basic.obo'
        else:
            self.ontology_path = ontology_path
        self.go = GODag(str(self.ontology_path))

    def get_ancestors(self, term_id):
        if term_id in self.go:
            return self.go[term_id].get_all_parents()
        else:
            print(f'Term {term_id} not found in ontology')
            return set()

    def calculate_ic(self, annotations_dict):
        term_to_proteins = collections.defaultdict(set)
        for protein, terms in annotations_dict.items():
            for term in terms:
                term_to_proteins[term].add(protein)
                ancestors = self.get_ancestors(term)
                for ancestor in ancestors:
                    term_to_proteins[ancestor].add(protein)

        ic_map = {}
        num_proteins = len(annotations_dict)
        for term in self.go:
            parents = self.go[term].parents 
            if len(parents) == 0:
                ic_map[term] = 0
                continue
            parent_sets = [term_to_proteins[p.id] for p in parents]
            if parent_sets:
                set_intersection = set.intersection(*parent_sets)
            else:
                set_intersection = set()
            prob_parents = len(set_intersection) / num_proteins
            prob_f = len(term_to_proteins[term]) / num_proteins
            if prob_parents == 0:
                ic_map[term] = 0.0
            else:
                ic_map[term] = -np.log2(prob_f / prob_parents)
        return ic_map

            






        