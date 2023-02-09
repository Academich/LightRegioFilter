from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import AllChem
from IPython.display import SVG


def draw_reaction_smarts(rxn_smiles: str, use_smiles: bool = True, highlight: bool = True) -> 'SVG':
    """
    Draws a reaction from a reaction smiles string
    """
    _rxn = rxn_smiles.split("|")[0]
    rxn = AllChem.ReactionFromSmarts(_rxn, useSmiles=use_smiles)
    d = Draw.MolDraw2DSVG(900, 300)
    if highlight:
        colors = [(0.3, 0.7, 0.9), (0.9, 0.7, 0.9), (0.6, 0.9, 0.3), (0.9, 0.9, 0.1)]
        d.DrawReaction(rxn, highlightByReactant=True, highlightColorsReactants=colors)
    else:
        d.DrawReaction(rxn, highlightByReactant=False)
    d.FinishDrawing()

    svg = d.GetDrawingText()
    return SVG(svg.replace('svg:', '').replace(':svg', ''))


def canonical_remove_aam_mol(smi: str) -> str:
    """
    Removes atom mapping from a Mol object using RDKit
    :param smi:
    :return: Canonicalized SMILES with no atom mapping
    """
    mol = Chem.MolFromSmiles(smi)
    [a.SetAtomMapNum(0) for a in mol.GetAtoms()]
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def map_reaction_center(rx):
    atom_map_to_reactant_map = {}
    for ri in range(rx.GetNumReactantTemplates()):
        rt = rx.GetReactantTemplate(ri)
        for atom in rt.GetAtoms():
            if atom.GetAtomMapNum():
                atom_map_to_reactant_map[atom.GetAtomMapNum()] = (atom.GetAtomicNum(), ri)
    return atom_map_to_reactant_map


def canonicalize_reaction(smi: str) -> str:
    left, center, right = smi.split(">")
    left = ".".join([Chem.MolToSmiles(Chem.MolFromSmiles(s), isomericSmiles=False) for s in left.split(".")])
    center = ".".join([Chem.MolToSmiles(Chem.MolFromSmiles(s), isomericSmiles=False) for s in center.split(".")])
    right = ".".join([Chem.MolToSmiles(Chem.MolFromSmiles(s), isomericSmiles=False) for s in right.split(".")])
    return left + ">" + center + ">" + right


def _unsubstituted_part_of_an_aromatic_ring(atom, mol):
    for a in mol.GetAtoms():
        if a.GetAtomMapNum() == atom.GetAtomMapNum():
            return tuple([(i.GetIsAromatic() or "&a" in i.GetSmarts()) for i in a.GetNeighbors()]) == (True, True)


def template_relevant(template_left: str, template_right: str):
    if "Br" in template_right and "Br" in template_left:
        # Check if the bromine atom is connected to an aromatic carbon atom
        left = Chem.MolFromSmarts(template_left)
        right = Chem.MolFromSmarts(template_right)
        for a in right.GetAtoms():
            if a.GetSymbol() == "Br":
                hal_neighbor = a.GetNeighbors()[0]
                return hal_neighbor.GetAtomicNum() == 6 and hal_neighbor.GetIsAromatic() and _unsubstituted_part_of_an_aromatic_ring(hal_neighbor, left)
    return False


def relevant_site_from_mapping(rsmi: str):
    candidate_atom_map_nums = set()
    reactants, _, products = rsmi.split(">")
    reactant_mols = Chem.MolFromSmiles(reactants)
    product_mols = Chem.MolFromSmiles(products)
    if reactant_mols is None or product_mols is None:
        return
    for ar in reactant_mols.GetAtoms():
        if ar.GetAtomicNum() == 6 and ar.GetIsAromatic() and tuple([i.GetIsAromatic() for i in ar.GetNeighbors()]) == (True, True):
            candidate_atom_map_nums.add(ar.GetAtomMapNum())
    for ap in product_mols.GetAtoms():
        if ap.GetAtomMapNum() in candidate_atom_map_nums and ap.GetAtomicNum() == 6 and len(ap.GetNeighbors()) == 3 and len({"Br", "Cl", "I"} & {i.GetSymbol() for i in ap.GetNeighbors()}) > 0:
            relevant_reactant = [m for m in reactants.split(".") if f":{ap.GetAtomMapNum()}]" in m].pop()
            relevant_reactant = Chem.MolToSmiles(Chem.MolFromSmiles(relevant_reactant), isomericSmiles=True)
            relevant_reactant_mol = Chem.MolFromSmiles(relevant_reactant)
            for a in relevant_reactant_mol.GetAtoms():
                if a.GetAtomMapNum() == ap.GetAtomMapNum():
                    return a.GetIdx(), relevant_reactant
                

# === Tools for faster data processing on CPU using pool of processes ===
from pandas import Series, concat
from multiprocessing import Pool
from typing import Callable
from functools import partial
import numpy as np

def __parallelize(d: Series, func: Callable, num_of_processes: int) -> Series:
    data_split = np.array_split(d, num_of_processes)
    pool = Pool(num_of_processes)
    d = concat(pool.map(func, data_split))
    pool.close()
    pool.join()
    return d


def run_on_subset(func: Callable, use_tqdm, data_subset):
    if use_tqdm:
        return data_subset.progress_apply(func)
    return data_subset.apply(func)


def parallelize_on_rows(d: Series, func, num_of_processes: int, use_tqdm=False) -> Series:
    return __parallelize(d, partial(run_on_subset, func, use_tqdm), num_of_processes)
