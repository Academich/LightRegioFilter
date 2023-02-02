import numpy as np
from DescriptorCreator.find_atoms import find_identical_atoms
from DescriptorCreator.PrepAndCalcDescriptor import EASMolPreparation
from rdkit import Chem
import lightgbm


class ReactionSitePredictor:

    def __init__(self,
                 model: str,
                 n_shells: int = 5,
                 use_cip_sort: bool = True):
        self.model = lightgbm.Booster(model_file=model)
        self.predictor = EASMolPreparation()
        self.des = ('GraphChargeShell', {'charge_type': 'cm5',
                                         'n_shells': n_shells,
                                         'use_cip_sort': use_cip_sort})

    def run(self, smiles: str):
        smiles = Chem.MolToSmiles(Chem.MolFromSmiles(smiles), isomericSmiles=True)  # canonicalize input smiles
        cm5_list = self.predictor.calc_CM5_charges(smiles, name='', optimize=False, save_output=True)
        atom_indices, descriptor_vector = self.predictor.create_descriptor_vector(self.des[0], **self.des[1])
        pred_proba = self.model.predict(descriptor_vector, num_iteration=self.model.best_iteration)
        pred = np.rint(pred_proba)
        atom_reactive = [bool(x) for x in pred]
        reactive_sites = np.array(atom_indices)[atom_reactive].tolist()
        reactive_sites = find_identical_atoms(self.predictor.rdkit_mol, reactive_sites)
        labels = [int(1) if site in reactive_sites else int(0) for site in range(len(cm5_list))]
        predicted_reaction_sites = np.array(range(len(labels)))[np.array(labels).astype("bool")]
        return predicted_reaction_sites
