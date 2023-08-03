# coding: utf-8

"""
Definition of variables.
"""

import order as od

from columnflow.columnar_util import EMPTY_FLOAT


def add_variables(config: od.Config) -> None:
    """
    Adds all variables to a *config*.
    """
    config.add_variable(
        name="event",
        expression="event",
        binning=(1, 0.0, 1.0e9),
        x_title="Event number",
        discrete_x=True,
    )
    config.add_variable(
        name="run",
        expression="run",
        binning=(1, 100000.0, 500000.0),
        x_title="Run number",
        discrete_x=True,
    )
    config.add_variable(
        name="lumi",
        expression="luminosityBlock",
        binning=(1, 0.0, 5000.0),
        x_title="Luminosity block",
        discrete_x=True,
    )
    config.add_variable(
        name="n_jet",
        expression="n_jet",
        binning=(11, -0.5, 10.5),
        x_title="Number of jets",
        discrete_x=True,
    )
    config.add_variable(
        name="n_hhbtag",
        expression="n_hhbtag",
        binning=(4, -0.5, 3.5),
        x_title="Number of HH b-tags",
        discrete_x=True,
    )
    config.add_variable(
        name="ht",
        binning=[0, 80, 120, 160, 200, 240, 280, 320, 400, 500, 600, 800],
        unit="GeV",
        x_title="HT",
    )
    # Jets
    config.add_variable(
        name="jet1_pt",
        expression="Jet.pt[:,0]",
        null_value=EMPTY_FLOAT,
        binning=(40, 0.0, 400.0),
        unit="GeV",
        x_title=r"Jet 1 $p_{T}$",
    )
    config.add_variable(
        name="jet1_p",
        expression="Jet.p_mag[:,0]",
        null_value=EMPTY_FLOAT,
        binning=(40, 0.0, 400.0),
        unit="GeV",
        x_title=r"Jet 1 $|p|$",
    )
    config.add_variable(
        name="jet1_eta",
        expression="Jet.eta[:,0]",
        null_value=EMPTY_FLOAT,
        binning=(30, -3.0, 3.0),
        x_title=r"Jet 1 $\eta$",
    )
    config.add_variable(
        name="jet1_m",
        expression="Jet.mass[:,0]",
        null_value=EMPTY_FLOAT,
        binning=(30, 0.0, 150.0),
        unit="GeV",
        x_title=r"Jet 1 $m_{Jet}$",
    )
    config.add_variable(
        name="jet1_dfb",
        expression="Jet.btagDeepFlavB[:,0]",
        null_value=EMPTY_FLOAT,
        binning=(50, 0.0, 1.0),
        x_title=r"Jet 1 DeepFlavourB",
    )
    config.add_variable(
        name="jet2_pt",
        expression="Jet.pt[:,1]",
        null_value=EMPTY_FLOAT,
        binning=(40, 0.0, 400.0),
        unit="GeV",
        x_title=r"Jet 2 $p_{T}$",
    )
    config.add_variable(
        name="jet2_p",
        expression="Jet.p_mag[:,1]",
        null_value=EMPTY_FLOAT,
        binning=(40, 0.0, 400.0),
        unit="GeV",
        x_title=r"Jet 2 $|p|$",
    )
    config.add_variable(
        name="jet2_eta",
        expression="Jet.eta[:,1]",
        null_value=EMPTY_FLOAT,
        binning=(30, -3.0, 3.0),
        x_title=r"Jet 2 $\eta$",
    )
    config.add_variable(
        name="jet2_m",
        expression="Jet.mass[:,1]",
        null_value=EMPTY_FLOAT,
        binning=(30, 0.0, 150.0),
        unit="GeV",
        x_title=r"Jet 2 $m_{Jet}$",
    )
    config.add_variable(
        name="jet2_dfb",
        expression="Jet.btagDeepFlavB[:,1]",
        null_value=EMPTY_FLOAT,
        binning=(50, 0.0, 1.0),
        x_title=r"Jet 2 DeepFlavourB",
    )
    #METs
    config.add_variable(
        name="met_phi",
        expression="MET.phi",
        null_value=EMPTY_FLOAT,
        binning=(33, -3.3, 3.3),
        x_title=r"MET $\phi$",
    )
    config.add_variable(
        name="met_pt",
        expression="MET.pt",
        null_value=EMPTY_FLOAT,
        binning=(40, 0.0, 400.0),
        unit="GeV",
        x_title=r"MET $p_{T}$",
    )
    #Taus
    config.add_variable(
        name="tau1_pt",
        expression="Tau.pt[:,0]",
        null_value=EMPTY_FLOAT,
        binning=(40, 0.0, 400.0),
        unit="GeV",
        x_title=r"Tau 1 $p_{T}$",
    )
    config.add_variable(
        name="tau1_p",
        expression="Tau.p_mag[:,0]",
        null_value=EMPTY_FLOAT,
        binning=(40, 0.0, 400.0),
        unit="GeV",
        x_title=r"Tau 1 $|p|$",
    )
    config.add_variable(
        name="tau1_eta",
        expression="Tau.eta[:,0]",
        null_value=EMPTY_FLOAT,
        binning=(30, -3.0, 3.0),
        x_title=r"Tau 1 $\eta$",
    )
    config.add_variable(
        name="tau1_m",
        expression="Tau.mass[:,0]",
        null_value=EMPTY_FLOAT,
        binning=(30, 0.0, 150.0),
        unit="GeV",
        x_title=r"Tau 1 $m_{\tau}$",
    )
    config.add_variable(
        name="tau2_pt",
        expression="Tau.pt[:,1]",
        null_value=EMPTY_FLOAT,
        binning=(40, 0.0, 400.0),
        unit="GeV",
        x_title=r"Tau 2 $p_{T}$",
    )
    config.add_variable(
        name="tau2_p",
        expression="Tau.p_mag[:,1]",
        null_value=EMPTY_FLOAT,
        binning=(40, 0.0, 400.0),
        unit="GeV",
        x_title=r"Tau 2 $|p|$",
    )
    config.add_variable(
        name="tau2_eta",
        expression="Tau.eta[:,1]",
        null_value=EMPTY_FLOAT,
        binning=(30, -3.0, 3.0),
        x_title=r"Tau 2 $\eta$",
    )
    config.add_variable(
        name="tau2_m",
        expression="Tau.mass[:,1]",
        null_value=EMPTY_FLOAT,
        binning=(30, 0.0, 150.0),
        unit="GeV",
        x_title=r"Tau 2 $m_{\tau}$",
    )
    #BJets
    config.add_variable(
        name="bjet1_pt",
        expression="BJet.pt[:,0]",
        null_value=EMPTY_FLOAT,
        binning=(40, 0.0, 400.0),
        unit="GeV",
        x_title=r"B-Jet 1 $p_{T}$",
    )
    config.add_variable(
        name="bjet1_p",
        expression="BJet.p_mag[:,0]",
        null_value=EMPTY_FLOAT,
        binning=(40, 0.0, 400.0),
        unit="GeV",
        x_title=r"BJet 1 $|p|$",
    )
    config.add_variable(
        name="bjet1_eta",
        expression="BJet.eta[:,0]",
        null_value=EMPTY_FLOAT,
        binning=(30, -3.0, 3.0),
        x_title=r"BJet 1 $\eta$",
    )
    config.add_variable(
        name="bjet1_m",
        expression="BJet.mass[:,0]",
        null_value=EMPTY_FLOAT,
        binning=(30, 0.0, 150.0),
        unit="GeV",
        x_title=r"BJet 1 $m_{BJet}$",
    )
    config.add_variable(
        name="bjet1_dfb",
        expression="BJet.btagDeepFlavB[:,0]",
        null_value=EMPTY_FLOAT,
        binning=(50, 0.0, 1.0),
        x_title=r"BJet 1 DeepFlavourB",
    )
    config.add_variable(
        name="bjet2_pt",
        expression="BJet.pt[:,1]",
        null_value=EMPTY_FLOAT,
        binning=(40, 0.0, 400.0),
        unit="GeV",
        x_title=r"B-Jet 2 $p_{T}$",
    )
    config.add_variable(
        name="bjet2_p",
        expression="BJet.p_mag[:,1]",
        null_value=EMPTY_FLOAT,
        binning=(40, 0.0, 400.0),
        unit="GeV",
        x_title=r"BJet 2 $|p|$",
    )
    config.add_variable(
        name="bjet2_eta",
        expression="BJet.eta[:,1]",
        null_value=EMPTY_FLOAT,
        binning=(30, -3.0, 3.0),
        x_title=r"BJet 2 $\eta$",
    )
    config.add_variable(
        name="bjet2_m",
        expression="BJet.mass[:,1]",
        null_value=EMPTY_FLOAT,
        binning=(30, 0.0, 150.0),
        unit="GeV",
        x_title=r"BJet 2 $m_{BJet}$",
    )
    config.add_variable(
        name="bjet2_dfb",
        expression="Jet.btagDeepFlavB[:,1]",
        null_value=EMPTY_FLOAT,
        binning=(50, 0.0, 1.0),
        x_title=r"BJet 2 DeepFlavourB",
    )
    #VBFJets
    config.add_variable(
        name="vbfjet1_pt",
        expression="VBFJet.pt[:,0]",
        null_value=EMPTY_FLOAT,
        binning=(40, 0.0, 400.0),
        unit="GeV",
        x_title=r"VBFJet 1 $p_{T}$",
    )
    config.add_variable(
        name="vbfjet1_p",
        expression="VBFJet.p_mag[:,0]",
        null_value=EMPTY_FLOAT,
        binning=(40, 0.0, 400.0),
        unit="GeV",
        x_title=r"VBFJet 1 $|p|$",
    )
    config.add_variable(
        name="vbfjet1_eta",
        expression="VBFJet.eta[:,0]",
        null_value=EMPTY_FLOAT,
        binning=(50, -5.0, 5.0),
        x_title=r"VBFJet 1 $\eta$",
    )
    config.add_variable(
        name="vbfjet1_m",
        expression="VBFJet.mass[:,0]",
        null_value=EMPTY_FLOAT,
        binning=(20, 0.0, 200.0),
        unit="GeV",
        x_title=r"VBFJet 1 $m_{Jet}$",
    )
    config.add_variable(
        name="vbfjet1_dfb",
        expression="VBFJet.btagDeepFlavB[:,0]",
        null_value=EMPTY_FLOAT,
        binning=(50, 0.0, 1.0),
        x_title=r"VBFJet 1 DeepFlavourB",
    )
    #delta_r
    config.add_variable(
        name="delta_r1",
        expression="Delta_r_Jet[:,0]",
        null_value=EMPTY_FLOAT,
        binning=(50, 0.0, 5.0),
        x_title=r"delta_r",
    )
    #invariant_mass
#    config.add_variable(
#        name="hardest_jet_pair_mass",
#        expression="mjj",
#        null_value=EMPTY_FLOAT,
#        binning=(40, 0.0, 400.0),
#        unit="GeV",
#        x_title=r"Hardest Jet Pair Mass",
#    )
#    config.add_variable(
#        name="tau_pair_mass",
#        expression="mtautau",
#        null_value=EMPTY_FLOAT,
#        binning=(40, 0.0, 400.0),
#        unit="GeV",
#        x_title=r"Tau Pair Mass",
#    )
#    config.add_variable(
#        name="bjet_pair_mass",
#        expression="mbjetbjet",
#        null_value=EMPTY_FLOAT,
#        binning=(40, 0.0, 400.0),
#        unit="GeV",
#        x_title=r"BJet Pair Mass",
#    )
#    config.add_variable(
#        name="HH_pair_mass",
#        expression="mHH",
#        null_value=EMPTY_FLOAT,
#        binning=(40, 0.0, 400.0),
#        unit="GeV",
#        x_title=r"HH Pair Mass",
#    )
#    config.add_variable(
#        name="hardest_jet_pair_pt",
#        expression="(Jet[:, 0] + Jet[:, 1]).pt",
#        null_value=EMPTY_FLOAT,
#        binning=(40, 0.0, 400.0),
#        unit="GeV",
#        x_title=r"Sum(0,1) $p_{T}$",
 #   )


    # weights
    config.add_variable(
        name="mc_weight",
        expression="mc_weight",
        binning=(200, -10, 10),
        x_title="MC weight",
    )
    config.add_variable(
        name="pu_weight",
        expression="pu_weight",
        binning=(40, 0, 2),
        x_title="Pileup weight",
    )
    config.add_variable(
        name="normalized_pu_weight",
        expression="normalized_pu_weight",
        binning=(40, 0, 2),
        x_title="Normalized pileup weight",
    )
    config.add_variable(
        name="btag_weight",
        expression="btag_weight",
        binning=(60, 0, 3),
        x_title="b-tag weight",
    )
    config.add_variable(
        name="normalized_btag_weight",
        expression="normalized_btag_weight",
        binning=(60, 0, 3),
        x_title="Normalized b-tag weight",
    )
    config.add_variable(
        name="normalized_njet_btag_weight",
        expression="normalized_njet_btag_weight",
        binning=(60, 0, 3),
        x_title="$N_{jet}$ normalized b-tag weight",
    )

    # cutflow variables
    config.add_variable(
        name="cf_njet",
        expression="cutflow.n_jet",
        binning=(17, -0.5, 16.5),
        x_title="Jet multiplicity",
        discrete_x=True,
    )
    config.add_variable(
        name="cf_ht",
        expression="cutflow.ht",
        binning=(40, 0.0, 400.0),
        unit="GeV",
        x_title=r"$H_{T}$",
    )
    config.add_variable(
        name="cf_jet1_pt",
        expression="cutflow.jet1_pt",
        binning=(40, 0.0, 400.0),
        unit="GeV",
        x_title=r"Jet 1 $p_{T}$",
    )
    config.add_variable(
        name="cf_jet1_eta",
        expression="cutflow.jet1_eta",
        binning=(40, -5.0, 5.0),
        x_title=r"Jet 1 $\eta$",
    )
    config.add_variable(
        name="cf_jet1_phi",
        expression="cutflow.jet1_phi",
        binning=(32, -3.2, 3.2),
        x_title=r"Jet 1 $\phi$",
    )
    config.add_variable(
        name="cf_jet2_pt",
        expression="cutflow.jet2_pt",
        binning=(40, 0.0, 400.0),
        unit="GeV",
        x_title=r"Jet 2 $p_{T}$",
    )
