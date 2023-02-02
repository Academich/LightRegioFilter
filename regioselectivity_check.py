from site_prediction import ReactionSitePredictor
from mapping import MappingFromTemplate
from utils import canonicalize_reaction


def regioselectivity_check(reaction: str,
                           template: str,
                           site_predictor: 'ReactionSitePredictor'):
    reaction = canonicalize_reaction(reaction)
    site_atom_index_actual, relevant_reactant = MappingFromTemplate(template, reaction).run()
    sites_atom_index_expected = site_predictor.run(relevant_reactant)
    return site_atom_index_actual in sites_atom_index_expected


if __name__ == '__main__':
    site_pred = ReactionSitePredictor("models/LGBM_measured_allData_final_model.txt")
    r1 = "O=C1CCC(=O)N1Br.NC(=O)Nc1cccc2ccc(=O)[nH]c12>>NC(=O)Nc1cc(Br)cc2ccc(=O)[nH]c12"
    t1 = "O=C1-C-C-C(=O)-N-1-[Br:1].[c:2]:[cH:3]:[c:4]>>[Br:1]-[c:3](:[c:2]):[c:4]"
    print(regioselectivity_check(r1, t1, site_pred))
