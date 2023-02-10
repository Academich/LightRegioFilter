from site_prediction import ReactionSitePredictor
from mapping import MappingFromTemplate
from utils import canonicalize_reaction, relevant_site_from_mapping


def _check(reported_site: int, mol: str, site_predictor: 'ReactionSitePredictor'):
    sites_atom_index_expected = site_predictor.run(mol)
    if len(sites_atom_index_expected) == 0:
        return "undecided"
    if reported_site in sites_atom_index_expected:
        return "correct"
    return "wrong"


def check_with_template(reaction: str,
                        template: str,
                        site_predictor: 'ReactionSitePredictor'):
    reaction = canonicalize_reaction(reaction)
    site_atom_index_actual, relevant_reactant = MappingFromTemplate(template, reaction).run()
    return _check(site_atom_index_actual, relevant_reactant, site_predictor)


def check_with_aam(reaction: str,
                   site_predictor: 'ReactionSitePredictor'):
    site_atom_index_actual, relevant_reactant = relevant_site_from_mapping(reaction)
    return _check(site_atom_index_actual, relevant_reactant, site_predictor)


if __name__ == '__main__':
    site_pred = ReactionSitePredictor("models/LGBM_measured_allData_final_model.txt")
    r1 = "O=C1CCC(=O)N1Br.NC(=O)Nc1cccc2ccc(=O)[nH]c12>>NC(=O)Nc1cc(Br)cc2ccc(=O)[nH]c12"
    t1 = "O=C1-C-C-C(=O)-N-1-[Br:1].[c:2]:[cH:3]:[c:4]>>[Br:1]-[c:3](:[c:2]):[c:4]"
    print(check_with_template(r1, t1, site_pred))
