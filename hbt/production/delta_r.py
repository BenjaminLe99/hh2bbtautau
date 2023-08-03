"""
module to calculate delta_r of 2 four-vectors
"""

import functools

from columnflow.production import Producer, producer
from columnflow.util import maybe_import
from columnflow.columnar_util import EMPTY_FLOAT, Route, set_ak_column
from columnflow.production.util import attach_coffea_behavior
from hbt.selection.genmatching import genmatching_selector

np = maybe_import("numpy")
ak = maybe_import("awkward")

# helper
set_ak_column_f32 = functools.partial(set_ak_column, value_type=np.float32)
set_ak_column_i32 = functools.partial(set_ak_column, value_type=np.int32)

@producer(
    uses={
       "GenmatchedJets.*", "GenmatchedHHBtagJets.*", "GenBPartons.*", "GenmatchedGenJets.*", genmatching_selector, "nGenJet", "Jet.*",
        attach_coffea_behavior
    },
    produces={
        # new columns
        "delta_r_2_matches", "delta_r_HHbtag", "delta_r_genbpartons", "delta_r_genmatchedgenjets"
    },
)
def genmatched_delta_r(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    """
    Creates new columns: 'delta_r_2_matches' for the delta r values of the first two genmatched jets
    and 'delta_r_HHbtag' for the delta r values of the first two HHBtag jets.
    """
    # TODO: Also implement delta r values for partons and for matched genjets.
    collections = {x: {"type_name" : "Jet"} for x in ["GenmatchedJets", "GenmatchedHHBtagJets"]}
    collections.update({y: {"type_name" : "GenParticle", "skip_fields": "*Idx*G",} for y in ["GenBPartons"]})
    collections.update({y: {"type_name" : "Jet", "skip_fields": "*Idx*G",} for y in ["GenmatchedGenJets"]})


    events = self[attach_coffea_behavior](events, collections=collections, **kwargs)

    def calculate_delta_r(array: ak.Array, num_objects: int=2):
        # calculate all possible delta R values (all permutations):
        all_deltars = array.metric_table(array)
        min_deltars_permutations = ak.firsts(all_deltars)
        real_deltar_mask = min_deltars_permutations != 0
        real_deltars = ak.mask(min_deltars_permutations, real_deltar_mask)
        real_deltars = min_deltars_permutations[real_deltar_mask]
        mask = ak.num(array, axis=1) == num_objects
        return ak.where(mask, ak.flatten(real_deltars), EMPTY_FLOAT)

    #events = set_ak_column_f32(events, "delta_r_genbpartons", calculate_delta_r(events.GenBPartons))
    #events = set_ak_column_f32(events, "delta_r_genmatchedgenjets", calculate_delta_r(events.GenmatchedGenJets))

    return events