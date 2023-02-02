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
