"""
module to calculate the magnitude of e.g. momenta
"""


import functools

from columnflow.production import Producer, producer
from columnflow.util import maybe_import, safe_div
from columnflow.columnar_util import set_ak_column, EMPTY_FLOAT
from columnflow.production.util import attach_coffea_behavior


np = maybe_import("numpy")
ak = maybe_import("awkward")

# helper
set_ak_column_f32 = functools.partial(set_ak_column, value_type=np.float32)


@producer(
    uses={  
        attach_coffea_behavior,
        # nano columns
        "Jet.pt", "Jet.eta", "Jet.phi", "Jet.e",
        "BJet.pt", "BJet.eta", "BJet.phi", "BJet.e",
        "Tau.pt", "Tau.eta", "Tau.phi", "Tau.e",
        "VBFJet.pt", "VBFJet.eta", "VBFJet.phi", "VBFJet.e",
    },
    produces={
        "Jet.p_mag", "BJet.p_mag", "Tau.p_mag", "VBFJet.p_mag",
    }
)
def jet_momentum_magnitude(
    self,
    events: ak.Array,
    **kwargs,
):
    collections = {x: {"type_name": "Jet"} for x in "Jet BJet".split()}
    collections.update({"Tau": {"type_name": "Tau"}})
    collections.update({"VBFJet": {"type_name": "Jet"}})
    #from IPython import embed; embed()
    events = self[attach_coffea_behavior](events, collections=collections, **kwargs)
    jet_momenta = events.Jet.pvec.absolute()
    bjet_momenta = events.BJet.pvec.absolute()
    tau_momenta = events.Tau.pvec.absolute()
    vbfjet_momenta = events.VBFJet.pvec.absolute()

    events = set_ak_column_f32(events, "Jet.p_mag", ak.where(ak.num(events.Jet, axis=1)>0, jet_momenta, EMPTY_FLOAT))
    events = set_ak_column_f32(events, "BJet.p_mag", ak.where(ak.num(events.BJet, axis=1)>0, bjet_momenta, EMPTY_FLOAT))
    events = set_ak_column_f32(events, "Tau.p_mag", ak.where(ak.num(events.Tau, axis=1)>0, tau_momenta, EMPTY_FLOAT))
    events = set_ak_column_f32(events, "VBFJet.p_mag", ak.where(ak.num(events.VBFJet, axis=1)>0, vbfjet_momenta, EMPTY_FLOAT))
    return events