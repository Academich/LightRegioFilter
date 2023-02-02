from typing import List, Optional

from rdkit import Chem
from rdkit.Chem import rdChemReactions

from utils import canonical_remove_aam_mol


def map_reaction_center(rx):
    atom_map_to_reactant_map = {}
    for ri in range(rx.GetNumReactantTemplates()):
        rt = rx.GetReactantTemplate(ri)
        for atom in rt.GetAtoms():
            if atom.GetAtomMapNum():
                atom_map_to_reactant_map[atom.GetAtomMapNum()] = (atom.GetAtomicNum(), ri)
    return atom_map_to_reactant_map


class MappingFromTemplate:

    def __init__(self,
                 template: str,
                 reaction: str
                 ) -> None:
        reactants, products = reaction.split(">>")
        self.products = products
        self.rxn = rdChemReactions.ReactionFromSmarts(template)
        self.reactant_smiles = reactants.split(".")
        self.reactant_mols = []
        self.aam_ranges = []  # ranges of AAM indexes for every index of a molecule

    def _set_up_reactants(self) -> None:

        for i, smi in enumerate(self.reactant_smiles):
            mol = Chem.MolFromSmiles(smi)

            n_atoms_so_far = 0
            if self.aam_ranges:
                n_atoms_so_far = self.aam_ranges[-1][-1]
            self.aam_ranges.append(
                range(1 + n_atoms_so_far,
                      1 + n_atoms_so_far + len([a for a in mol.GetAtoms()]))
            )

            for a in mol.GetAtoms():
                a.SetAtomMapNum(a.GetIdx() + 1 + n_atoms_so_far)
            self.reactant_mols.append(mol)

    def run(self):

        self._set_up_reactants()
        products = self.rxn.RunReactants(self.reactant_mols)

        for p in range(len(products)):

            # === Discarding all the generated products which are not the desired one ===
            product = products[p][0]
            product.UpdatePropertyCache()
            if canonical_remove_aam_mol(Chem.MolToSmiles(product)) != Chem.CanonSmiles(self.products):
                continue

            # === Mapping indexes of product atoms to indexes of reactant atoms ===
            rxn_center_atom_origin = map_reaction_center(self.rxn)

            # === Getting the index of the carbon atom that got targeted in the reaction ===
            for atom in product.GetAtoms():
                props = atom.GetPropsAsDict()
                if "old_mapno" in props:
                    element, atom_origin = rxn_center_atom_origin[props["old_mapno"]]
                    if props.get("_ReactionDegreeChanged") == 1 and element == 6:
                        return props["react_atom_idx"], self.reactant_smiles[atom_origin]


if __name__ == '__main__':
    # diels_alder_template = "[C:1]=[C:2][C:3]=[C:4].[C:5]=[C:6]>>[C:1]1[C:2]=[C:3][C:4][C:5][C:6]1"
    eas_template = "O=C1-C-C-C(=O)-N-1-[Br;H0;D1;+0:1].[c:2]:[cH;D2;+0:3]:[c:4]>>[Br;H0;D1;+0:1]-[c;H0;D3;+0:3](:[c:2]):[c:4]"
    # diene = "C=CC=C"
    # dienophile = "C=CC#N"
    rxn = "O=C1CCC(=O)N1Br.CNC(=S)Nc1ccccc1>>CNC(=S)Nc1ccc(Br)cc1"
    report = MappingFromTemplate(eas_template, rxn).run()
    print(report)
