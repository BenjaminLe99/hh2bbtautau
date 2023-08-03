# coding: utf-8

"""
Wrappers for some default sets of producers.
"""

from columnflow.production import Producer, producer
from columnflow.production.normalization import normalization_weights
from columnflow.production.categories import category_ids
from columnflow.production.cms.electron import electron_weights
from columnflow.production.cms.muon import muon_weights
from columnflow.util import maybe_import

from hbt.production.features import features
from hbt.production.weights import normalized_pu_weight, normalized_pdf_weight, normalized_murmuf_weight
from hbt.production.btag import normalized_btag_weights
from hbt.production.tau import tau_weights, trigger_weights
from columnflow.production.util import attach_coffea_behavior


from columnflow.columnar_util import EMPTY_FLOAT, Route, set_ak_column
import functools


np = maybe_import("numpy")
ak = maybe_import("awkward")

set_ak_column_f32 = functools.partial(set_ak_column, value_type=np.float32)

pad = EMPTY_FLOAT


# Invariant Mass Producers
@producer(
    uses={
        "Jet.pt", "Jet.nJet", "Jet.eta", "Jet.phi", "Jet.mass",
        attach_coffea_behavior,
    },
    produces={
        "mjj",
    },
)
def invariant_mass_jets(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    # category ids
    # invariant mass of two hardest jets
    events = self[attach_coffea_behavior](events, collections=["Jet"], **kwargs)
    events = set_ak_column(events, "Jet", ak.pad_none(events.Jet, 2))
    events = set_ak_column(events, "mjj", (events.Jet[:, 0] + events.Jet[:, 1]).mass)
    events = set_ak_column(events, "mjj", ak.fill_none(events.mjj, EMPTY_FLOAT))

    return events


@producer(
    uses={
        "Tau.pt", "Tau.eta", "Tau.phi", "Tau.mass",
        "Tau.charge",
        attach_coffea_behavior,
    },
    produces={
        "mtautau",
    },
)
def invariant_mass_tau(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    # invarinat mass of tau 1 and 2
    events = self[attach_coffea_behavior](events, collections=["Tau"], **kwargs)
    ditau = events.Tau[:, :2].sum(axis=1)
    ditau_mask = ak.num(events.Tau, axis=1) >= 2
    events = set_ak_column_f32(
        events,
        "mtautau",
        ak.where(ditau_mask, ditau.mass, EMPTY_FLOAT),
    )
    return events


@producer(
    uses={
        "BJet.pt", "BJet.eta", "BJet.phi", "BJet.mass",
        attach_coffea_behavior,
    },
    produces={
        "mbjetbjet",
    },
)
def invariant_mass_bjets(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    # invarinat mass of bjet 1 and 2, sums b jets with highest pt
    events = self[attach_coffea_behavior](events, collections={"BJet": {"type_name": "Jet"}}, **kwargs)
    diBJet = events.BJet[:, :2].sum(axis=1)
    diBJet_mask = ak.num(events.BJet, axis=1) >= 2
    events = set_ak_column_f32(
        events,
        "mbjetbjet",
        ak.where(diBJet_mask, diBJet.mass, EMPTY_FLOAT),
    )
    return events


@producer(
    uses={
        "BJet.pt", "BJet.eta", "BJet.phi", "BJet.mass",
        "Tau.pt", "Tau.eta", "Tau.phi", "Tau.mass", "Tau.charge",
        attach_coffea_behavior,
    },
    produces={
        "mHH",
    },
)
def invariant_mass_HH(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    # invarinat mass of bjet 1 and 2, sums b jets with highest pt
    events = self[attach_coffea_behavior](events, collections={"BJet": {"type_name": "Jet"}, "Tau": {"type_name": "Tau"}}, **kwargs)
    diHH = events.BJet[:, :2].sum(axis=1) + events.Tau[:, :2].sum(axis=1)
    dibjet_mask = ak.num(events.BJet, axis=1) >= 2
    ditau_mask = ak.num(events.Tau, axis=1) >= 2
    diHH_mask = np.logical_and(dibjet_mask, ditau_mask)
    events = set_ak_column_f32(
        events,
        "mHH",
        ak.where(diHH_mask, diHH.mass, EMPTY_FLOAT),
    )
    return events


# Producers for the columns of the kinetmatic variables of the jets, bjets and taus
@producer(
    uses={
        "Jet.p_mag", "Jet.pt", "Jet.nJet", "Jet.eta", "Jet.phi", "Jet.mass", "Jet.E", "Jet.area",
        "Jet.nConstituents", "Jet.jetID",
        attach_coffea_behavior,
    },
    produces={
        f"{obj}_{var}"
        for obj in ["jet1", "jet2", "jet3", "jet4"]
        for var in ["p", "pt", "eta", "phi", "mass", "e"]
    },
)
def kinematic_vars_jets(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    # Jet 1 and 2 kinematic variables
    # Jet 1
    
    events = self[attach_coffea_behavior](events, collections=["Jet"], **kwargs)
    requested_jets = 8
    jets_e = ak.pad_none(events.Jet.E, requested_jets, axis=1)
    jets_mass = ak.pad_none(events.Jet.mass, requested_jets, axis=1)
    jets_pt = ak.pad_none(events.Jet.pt, requested_jets, axis=1)
    jets_p = ak.pad_none(events.Jet.p, requested_jets, axis=1)
    jets_eta = ak.pad_none(events.Jet.eta, requested_jets, axis=1)
    jets_phi = ak.pad_none(events.Jet.phi, requested_jets, axis=1)
    for i in range(0,requested_jets):
        events = set_ak_column_f32(events, f"jet{i+1}_e", ak.fill_none(jets_e[:, i], pad))
        events = set_ak_column_f32(events, f"jet{i+1}_mass", ak.fill_none(jets_mass[:, i], pad))
        events = set_ak_column_f32(events, f"jet{i+1}_pt", ak.fill_none(jets_pt[:, i], pad))
        events = set_ak_column_f32(events, f"jet{i+1}_p", ak.fill_none(jets_p[:, i], pad))
        events = set_ak_column_f32(events, f"jet{i+1}_eta", ak.fill_none(jets_eta[:, i], pad))
        events = set_ak_column_f32(events, f"jet{i+1}_phi", ak.fill_none(jets_phi[:, i], pad))

    #events = set_ak_column_f32(events, "jet1_e", Route("Jet.E[:,0]").apply(events, EMPTY_FLOAT))
    #events = set_ak_column_f32(events, "jet1_pt", Route("Jet.pt[:,0]").apply(events, EMPTY_FLOAT))
    #events = set_ak_column_f32(events, "jet1_p", Route("Jet.p_mag[:,0]").apply(events, EMPTY_FLOAT))
    #events = set_ak_column_f32(events, "jet1_eta", Route("Jet.eta[:,0]").apply(events, EMPTY_FLOAT))
    #events = set_ak_column_f32(events, "jet1_phi", Route("Jet.phi[:,0]").apply(events, EMPTY_FLOAT))
    return events


@producer(
    uses={
        "BJet.p_mag", "BJet.pt", "BJet.eta", "BJet.phi", "BJet.mass", "BJet.E",
        attach_coffea_behavior,
    },
    produces={
        f"{obj}_{var}"
        for obj in ["bjet1", "bjet2", "bjet3", "bjet4"]
        for var in ["p", "pt", "eta", "phi", "mass", "e"]
    },
)
def kinematic_vars_bjets(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    events = self[attach_coffea_behavior](events, collections=["Jet"], **kwargs)
    requested_jets = 4
    bjets_e = ak.pad_none(events.BJet.E, requested_jets, axis=1)
    bjets_mass = ak.pad_none(events.BJet.mass, requested_jets, axis=1)
    bjets_pt = ak.pad_none(events.BJet.pt, requested_jets, axis=1)
    bjets_p = ak.pad_none(events.BJet.p, requested_jets, axis=1)
    bjets_eta = ak.pad_none(events.BJet.eta, requested_jets, axis=1)
    bjets_phi = ak.pad_none(events.BJet.phi, requested_jets, axis=1)
    for i in range(0,requested_jets):
        events = set_ak_column_f32(events, f"bjet{i+1}_e", ak.fill_none(bjets_e[:, i], pad))
        events = set_ak_column_f32(events, f"bjet{i+1}_mass", ak.fill_none(bjets_mass[:, i], pad))
        events = set_ak_column_f32(events, f"bjet{i+1}_pt", ak.fill_none(bjets_pt[:, i], pad))
        events = set_ak_column_f32(events, f"bjet{i+1}_p", ak.fill_none(bjets_p[:, i], pad))
        events = set_ak_column_f32(events, f"bjet{i+1}_eta", ak.fill_none(bjets_eta[:, i], pad))
        events = set_ak_column_f32(events, f"bjet{i+1}_phi", ak.fill_none(bjets_phi[:, i], pad))
    return events


@producer(
    uses={
        "Tau.p_mag", "Tau.pt", "Tau.eta", "Tau.phi", "Tau.mass", "Tau.E",
        "Tau.charge",
        attach_coffea_behavior,
    },
    produces={
        f"{obj}_{var}"
        for obj in ["tau1", "tau2"]
        for var in ["p", "pt", "eta", "phi", "mass", "e"]
    },
)
def kinematic_vars_taus(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    events = self[attach_coffea_behavior](events, collections=["Tau"], **kwargs)
    requested_tau = 2
    tau_e = ak.pad_none(events.Tau.E, requested_tau, axis=1)
    tau_mass = ak.pad_none(events.Tau.mass, requested_tau, axis=1)
    tau_pt = ak.pad_none(events.Tau.pt, requested_tau, axis=1)
    tau_p = ak.pad_none(events.Tau.p, requested_tau, axis=1)
    tau_eta = ak.pad_none(events.Tau.eta, requested_tau, axis=1)
    tau_phi = ak.pad_none(events.Tau.phi, requested_tau, axis=1)
    for i in range(0,requested_tau):
        events = set_ak_column_f32(events, f"tau{i+1}_e", ak.fill_none(tau_e[:, i], pad))
        events = set_ak_column_f32(events, f"tau{i+1}_mass", ak.fill_none(tau_mass[:, i], pad))
        events = set_ak_column_f32(events, f"tau{i+1}_pt", ak.fill_none(tau_pt[:, i], pad))
        events = set_ak_column_f32(events, f"tau{i+1}_p", ak.fill_none(tau_p[:, i], pad))
        events = set_ak_column_f32(events, f"tau{i+1}_eta", ak.fill_none(tau_eta[:, i], pad))
        events = set_ak_column_f32(events, f"tau{i+1}_phi", ak.fill_none(tau_phi[:, i], pad))
    return events

@producer(
    uses={
        "VBFJet.p_mag", "VBFJet.pt", "VBFJet.eta", "VBFJet.phi", "VBFJet.mass", "VBFJet.e",
        attach_coffea_behavior,
    },
    produces={
        f"{obj}_{var}"
        for obj in ["vbfjet1", "vbfjet2"]
        for var in ["p", "pt", "eta", "phi", "mass", "E"]
    },
)
def kinematic_vars_vbfjets(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    #from IPython import embed; embed();
    events = self[attach_coffea_behavior](events, collections=["Jet"], **kwargs)
    requested_vbfjet = 2
    #vbfjet_e = ak.pad_none(events.VBFJet.e, requested_vbfjet, axis=1)
    vbfjet_mass = ak.pad_none(events.VBFJet.mass, requested_vbfjet, axis=1)
    vbfjet_pt = ak.pad_none(events.VBFJet.pt, requested_vbfjet, axis=1)
    #vbfjet_p = ak.pad_none(events.VBFJet.p, requested_vbfjet, axis=1)
    vbfjet_eta = ak.pad_none(events.VBFJet.eta, requested_vbfjet, axis=1)
    vbfjet_phi = ak.pad_none(events.VBFJet.phi, requested_vbfjet, axis=1)
    for i in range(0,requested_vbfjet):
        #events = set_ak_column_f32(events, f"vbfjet{i+1}_e", ak.fill_none(vbfjet_e[:, i], pad))
        events = set_ak_column_f32(events, f"vbfjet{i+1}_mass", ak.fill_none(vbfjet_mass[:, i], pad))
        events = set_ak_column_f32(events, f"vbfjet{i+1}_pt", ak.fill_none(vbfjet_pt[:, i], pad))
        #events = set_ak_column_f32(events, f"vbfjet{i+1}_p", ak.fill_none(vbfjet_p[:, i], pad))
        events = set_ak_column_f32(events, f"vbfjet{i+1}_eta", ak.fill_none(vbfjet_eta[:, i], pad))
        events = set_ak_column_f32(events, f"vbfjet{i+1}_phi", ak.fill_none(vbfjet_phi[:, i], pad))
    return events


# Producers for additional event information on the Jets
@producer(
    uses={
        "Jet.area", "Jet.nConstituents", "nJet", "Jet.hadronFlavour",
        attach_coffea_behavior,
    },
    produces={
        *[f"{obj}_{var}"
        for obj in ["jet1", "jet2"]
        for var in ["area", "nConstituents", "hadronFlavour"]], "jets_nJets"
    },
)
def jet_information(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    events = self[attach_coffea_behavior](events, collections=["Jet"], **kwargs)
    requested_jet = 4
    jet_area = ak.pad_none(events.Jet.area, requested_jet, axis=1)
    jet_nConstituents = ak.pad_none(events.Jet.nConstituents, requested_jet, axis=1)
    jet_hadronFlavour = ak.pad_none(events.Jet.hadronFlavour, requested_jet, axis=1)
    jet_nJets = ak.pad_none(events.nJet, requested_jet, axis=1)
    for i in range(0, requested_jet):
        events = set_ak_column_f32(events, f"jet{i+1}_area", ak.fill_none(jet_area[:, i], pad))
        events = set_ak_column_f32(events, f"jet{i+1}_nConstituents", ak.fill_none(jet_nConstituents[:, i], pad))
        events = set_ak_column_f32(events, f"jet{i+1}_hadronFlavour", ak.fill_none(jet_hadronFlavour[:, i], pad))
        events = set_ak_column_f32(events, "jets_nJets", ak.fill_none(jet_nJets[:, i], pad))
    return events


# Producers for additional event information on the BJets
@producer(
    uses={
        "BJet.area", "BJet.nConstituents", "BJet.btagDeepFlavB", "BJet.nJet",
        attach_coffea_behavior,
    },
    produces={
        *[f"{obj}_{var}"
        for obj in ["bjet1", "bjet2"]
        for var in ["area", "nConstituents", "btag"]], "bjets_nJets"
    },
)
def bjet_information(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    events = self[attach_coffea_behavior](events, collections={"BJet": {"type_name": "Jet"}}, **kwargs)
    requested_jet = 2
    bjet_area = ak.pad_none(events.BJet.area, requested_jet, axis=1)
    bjet_nConstituents = ak.pad_none(events.BJet.nConstituents, requested_jet, axis=1)
    bjet_btag = ak.pad_none(events.BJet.btagDeepFlavB, requested_jet, axis=1)
    bjet_nJets = ak.pad_none(events.BJet.nJets, requested_jet, axis=1)
    for i in range(0, requested_jet):
        events = set_ak_column_f32(events, f"bjet{i+1}_area", ak.fill_none(bjet_area[:, i], pad))
        events = set_ak_column_f32(events, f"bjet{i+1}_nConstituents", ak.fill_none(bjet_nConstituents[:, i], pad))
        events = set_ak_column_f32(events, f"bjet{i+1}_btag", ak.fill_none(bjet_btag[:, i], pad))
        events = set_ak_column_f32(events, "bjets_nJets", ak.fill_none(bjet_nJets[:, i], pad))
    return events