"""
module to calculate delta_r of the two highest pt
 four-vectors
"""

import functools

from columnflow.production import Producer, producer
from columnflow.util import maybe_import
from columnflow.columnar_util import EMPTY_FLOAT, Route, set_ak_column
from columnflow.production.util import attach_coffea_behavior

np = maybe_import("numpy")
ak = maybe_import("awkward")

# helper
set_ak_column_f32 = functools.partial(set_ak_column, value_type=np.float32)
set_ak_column_i32 = functools.partial(set_ak_column, value_type=np.int32)



@producer(
    uses={
        attach_coffea_behavior,
        "Jet.*", "BJet.*", "Tau.*", "VBFJet.*"
    },
    produces={
        "Delta_r_Jet", "Delta_r_BJet", "Delta_r_Tau", "Delta_r_VBFJet"
    }

)
def deltar(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    
    collections = {x: {"type_name": "Jet"} for x in "Jet BJet".split()}
    collections.update({"Tau": {"type_name": "Tau"}})
    collections.update({"VBFJet": {"type_name": "Jet"}})
    events = self[attach_coffea_behavior](events, collections=collections, **kwargs)
  
    def deltaR(a, b):
        mval = ak.Array(np.hypot(a.eta - b.eta, (a.phi - b.phi + np.pi) % (2 * np.pi) - np.pi))
        return mval
    
    jets_eta = ak.pad_none(events.Jet.eta, 2, axis=1)
    jets_phi = ak.pad_none(events.Jet.phi, 2, axis=1)
    deltar_jet = deltaR(events.Jet[:,0], events.Jet[:,1])

    #from IPython import embed; embed()
    #events = set_ak_column_f32(events, "Delta_r_Jet", deltaR(events.Jet[:,0], events.Jet[:,1]))
    events = set_ak_column_f32(events, "Delta_r_Jet", ak.where(ak.num(events.Jet, axis=1)>0, deltar_jet, EMPTY_FLOAT))
    #from IPython import embed; embed();
    #events = set_ak_column_f32(events, "Delta_r_BJet", deltaR(events.BJet[:,0], events.BJet[:,1]))
    #events = set_ak_column_f32(events, "Delta_r_Tau", deltaR(events.Tau[:,0], events.Tau[:,1]))
    #events = set_ak_column_f32(events, "Delta_r_VBFJet", deltaR(events.VBFJet[:,0], events.VBFJet[:,1]))

    return events