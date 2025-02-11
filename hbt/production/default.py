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
from hbt.production.magnitude import jet_momentum_magnitude
from hbt.production.delta_r import genmatched_delta_r
from hbt.production.delta_rx import deltar
from hbt.production.invariant_mass import invariant_mass_jets, invariant_mass_tau, invariant_mass_bjets, invariant_mass_HH, kinematic_vars_taus, kinematic_vars_jets, kinematic_vars_bjets, kinematic_vars_vbfjets, jet_information, bjet_information


ak = maybe_import("awkward")


@producer(
    uses={
        category_ids, features, normalization_weights, normalized_pdf_weight,
        normalized_murmuf_weight, normalized_pu_weight, normalized_btag_weights,
        tau_weights, electron_weights, muon_weights, trigger_weights, invariant_mass_jets,
        invariant_mass_tau, invariant_mass_bjets, invariant_mass_HH, kinematic_vars_taus,
        kinematic_vars_jets, kinematic_vars_bjets, kinematic_vars_vbfjets, jet_information, bjet_information,
        jet_momentum_magnitude, genmatched_delta_r, deltar
    },
    produces={
        category_ids, features, normalization_weights, normalized_pdf_weight,
        normalized_murmuf_weight, normalized_pu_weight, normalized_btag_weights,
        tau_weights, electron_weights, muon_weights, trigger_weights, invariant_mass_jets,
        invariant_mass_tau, invariant_mass_bjets, invariant_mass_HH, kinematic_vars_taus,
        kinematic_vars_jets, kinematic_vars_bjets, kinematic_vars_vbfjets, jet_information, bjet_information,
        jet_momentum_magnitude, genmatched_delta_r, deltar
    },
)
def default(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    # category ids
    events = self[category_ids](events, **kwargs)

    # features
    events = self[features](events, **kwargs)

    # invariant masses

    events = self[invariant_mass_jets](events, **kwargs)

    events = self[invariant_mass_bjets](events, **kwargs)

    events = self[invariant_mass_tau](events, **kwargs)

    events = self[invariant_mass_HH](events, **kwargs)

    # kinetmatic vars for jets, bjets, taus
    events = self[kinematic_vars_jets](events, **kwargs)

    events = self[kinematic_vars_bjets](events, **kwargs)

    events = self[kinematic_vars_taus](events, **kwargs)

    events = self[kinematic_vars_vbfjets](events, **kwargs)

    # additional information on the jets and bjets
    # events = self[jet_information](events, **kwargs)

    # events = self[bjet_information](events, **kwargs)

    # mc-only weights
    if self.dataset_inst.is_mc:
        # normalization weights
        events = self[normalization_weights](events, **kwargs)

        # normalized pdf weight
        events = self[normalized_pdf_weight](events, **kwargs)

        # normalized renorm./fact. weight
        events = self[normalized_murmuf_weight](events, **kwargs)

        # normalized pu weights
        events = self[normalized_pu_weight](events, **kwargs)

        # btag weights
        events = self[normalized_btag_weights](events, **kwargs)

        # tau weights
        events = self[tau_weights](events, **kwargs)

        # electron weights
        events = self[electron_weights](events, **kwargs)

        # muon weights
        events = self[muon_weights](events, **kwargs)

        # trigger weights
        events = self[trigger_weights](events, **kwargs)

        # invariant mass of jets, bjets, taus, HH sys
        events = self[invariant_mass_jets](events, **kwargs)

        events = self[invariant_mass_tau](events, **kwargs)

        events = self[invariant_mass_bjets](events, **kwargs)

        events = self[invariant_mass_HH](events, **kwargs)

    events = self[jet_momentum_magnitude](events, **kwargs)

    events = self[deltar](events, **kwargs)

    events = self[genmatched_delta_r](events, **kwargs)

    return events
