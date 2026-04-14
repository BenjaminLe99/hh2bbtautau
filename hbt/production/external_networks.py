# coding: utf-8

"""
Producers for evaluating torch-based models.
"""

from __future__ import annotations

import functools

import law

from columnflow.production import Producer
from columnflow.production.util import attach_coffea_behavior
from columnflow.columnar_util import (
    set_ak_column, attach_behavior, flat_np_view, EMPTY_FLOAT, default_coffea_collections, ak_concatenate_safe,
    layout_ak_array,
)
from columnflow.util import maybe_import, dev_sandbox, DotDict
from columnflow.types import Any, Literal

from hbt.util import MET_COLUMN

np = maybe_import("numpy")
scipy = maybe_import("scipy")
ak = maybe_import("awkward")


logger = law.logger.get_logger(__name__)

# helper functions
set_ak_column_f32 = functools.partial(set_ak_column, value_type=np.float32)
set_ak_column_i32 = functools.partial(set_ak_column, value_type=np.int32)

BTagType = Literal["pnet", "upart", "none"]


def rotate_to_phi(ref_phi: ak.Array, px: ak.Array, py: ak.Array) -> tuple[ak.Array, ak.Array]:
    """
    Rotates a momentum vector extracted from *events* in the transverse plane to a reference phi
    angle *ref_phi*. Returns the rotated px and py components in a 2-tuple.
    """
    new_phi = np.arctan2(py, px, dtype=np.float64) - ref_phi
    pt = (px**2 + py**2)**0.5
    return pt * np.cos(new_phi), pt * np.sin(new_phi)


class _external_dnn(Producer):
    """
    Base class for evaluating DNNs trained externally with PyTorch and our "standard" set of input features.
    """

    uses = {
        attach_coffea_behavior,
        "channel_id",
        "event",
        "Tau.{eta,phi,pt,mass,charge,decayMode}",
        "Electron.{eta,phi,pt,mass,charge}",
        "Muon.{eta,phi,pt,mass,charge}",
        "HHBJet.{pt,eta,phi,mass,hhbtag,btagPNet*,btagUParTAK4*}",
        "FatJet.{eta,phi,pt,mass}",
        MET_COLUMN("{pt,phi,covXX,covXY,covYY}"),
    }

    # which type of btagging variables to use
    btag_type: BTagType = "pnet"

    # limited chunk size to avoid memory issues
    max_chunk_size: int = 10_000

    # the empty value to insert to output columns in case of missing or broken values
    empty_value: float = EMPTY_FLOAT

    # optionally save input features
    produce_features: bool | None = None
    features_prefix: str = ""

    # produced columns are added in the deferred init below
    sandbox = dev_sandbox("bash::$HBT_BASE/sandboxes/venv_hbt.sh")

    # not exposed to command line selection
    exposed = False

    @property
    def output_prefix(self) -> str:
        # prefix for output columns
        return self.cls_name

    @property
    def external_name(self) -> str:
        # name of the model bundle in the external files
        return self.cls_name

    def init_func(self, **kwargs) -> None:
        # set feature production options when requested
        if self.produce_features is None:
            self.produce_features = self.config_inst.x.sync
            if not self.features_prefix:
                self.features_prefix = "sync"
        if self.features_prefix and not self.features_prefix.endswith("_"):
            self.features_prefix = f"{self.features_prefix}_"

        # add features to produced columns
        if self.produce_features:
            self.produces.add(f"{self.features_prefix}{self.cls_name}_*")

        # update shifts dynamically
        self.shifts.add("minbias_xs_{up,down}")  # variations of minbias_xs used in met phi correction
        self.shifts.update({  # all calibrations that change jet and lepton momenta
            shift_inst.name
            for shift_inst in self.config_inst.shifts
            if shift_inst.has_tag({"jec", "jer", "tec", "eec", "eer"})
        })

        # output column names
        # (could be generalized to allow inheriting classes to define different targets)
        self.output_columns = [
            f"{self.output_prefix}_{name}"
            for name in ["hh", "tt", "dy"]
        ]

        # update produced columns
        self.produces |= set(self.output_columns)

    def requires_func(self, task: law.Task, reqs: dict, **kwargs) -> None:
        super().requires_func(task, reqs, **kwargs)
        
        if "external_files" not in reqs:
            from columnflow.tasks.external import BundleExternalFiles
            reqs["external_files"] = BundleExternalFiles.req(task)

    def setup_func(self, task: law.Task, reqs: dict[str, DotDict[str, Any]], **kwargs) -> None:
        super().setup_func(task, reqs, **kwargs)

        from hbt.ml.evaluators import TorchEvaluator

        if not getattr(task, "taf_torch_evaluator", None):
            task.taf_torch_evaluator = TorchEvaluator()
        self.evaluator = task.taf_torch_evaluator

        bundle = reqs["external_files"]
        bundle.files
        model_path = getattr(bundle.files, self.external_name)
        self.evaluator.add_model(self.cls_name, model_path.abspath)

        # categorical values handled by the network
        # (names and values from training code that was aligned to KLUB notation)
        self.embedding_expected_inputs = {
            "pair_type": [0, 1, 2],  # old KLUB naming, 0: mutau, 1: etau, 2: tautau
            "decay_mode1": [-1, 0, 1, 10, 11],  # -1 for e/mu
            "decay_mode2": [0, 1, 10, 11],
            "charge1": [-1, 1],
            "charge2": [-1, 1],
            "is_boosted": [0, 1],  # whether a selected fatjet is present
            "has_jet_pair": [0, 1],  # whether two or more jets are present
        }

    def teardown_func(self, task: law.Task, **kwargs) -> None:
        """
        Stops the Torch evaluator.
        """
        if (evaluator := getattr(task, "taf_torch_evaluator", None)):
            evaluator.stop()
        task.taf_torch_evaluator = None
        self.evaluator = None

    def call_func(self, events: ak.Array, **kwargs) -> ak.Array:
        # start the evaluator
        if not self.evaluator.running:
            self.evaluator.start()

        # ensure coffea behavior
        events = self[attach_coffea_behavior](
            events,
            collections={"HHBJet": default_coffea_collections["Jet"]},
            **kwargs,
        )

        # get visible tau decay products, consider them all as tau types
        vis_taus = attach_behavior(
            ak_concatenate_safe((events.Electron, events.Muon, events.Tau), axis=1),
            type_name="Tau",
        )
        vis_tau1, vis_tau2 = vis_taus[:, 0], vis_taus[:, 1]

        # get decay mode of first lepton (e, mu or tau)
        tautau_mask = events.channel_id == self.config_inst.channels.n.tautau.id
        dm1 = -1 * np.ones(len(events), dtype=np.int32)
        if ak.any(tautau_mask):
            dm1[tautau_mask] = events.Tau.decayMode[tautau_mask][:, 0]

        # get decay mode of second lepton (also a tau, but position depends on channel)
        leptau_mask = (
            (events.channel_id == self.config_inst.channels.n.etau.id) |
            (events.channel_id == self.config_inst.channels.n.mutau.id)
        )
        dm2 = -1 * np.ones(len(events), dtype=np.int32)
        if ak.any(leptau_mask):
            dm2[leptau_mask] = events.Tau.decayMode[leptau_mask][:, 0]
        if ak.any(tautau_mask):
            dm2[tautau_mask] = events.Tau.decayMode[tautau_mask][:, 1]

        # the dnn treats dm 2 as 1, so we need to map it
        dm1 = np.where(dm1 == 2, 1, dm1)
        dm2 = np.where(dm2 == 2, 1, dm2)

        # whether the events is resolvede, boosted or neither
        has_jet_pair = ak.num(events.HHBJet) >= 2
        has_fatjet = ak.num(events.FatJet) >= 1

        # convert channel_id to pair_type
        pair_type = np.full(len(events), -1, dtype=np.int32)
        pair_type[events.channel_id == self.config_inst.channels.n.mutau.id] = 0
        pair_type[events.channel_id == self.config_inst.channels.n.etau.id] = 1
        pair_type[events.channel_id == self.config_inst.channels.n.tautau.id] = 2

        # before preparing the network inputs, define a mask of events which have caregorical features
        # that are actually covered by the networks embedding layers; other events cannot be evaluated!
        event_mask = (
            np.isin(pair_type, self.embedding_expected_inputs["pair_type"]) &
            np.isin(dm1, self.embedding_expected_inputs["decay_mode1"]) &
            np.isin(dm2, self.embedding_expected_inputs["decay_mode2"]) &
            np.isin(vis_tau1.charge, self.embedding_expected_inputs["charge1"]) &
            np.isin(vis_tau2.charge, self.embedding_expected_inputs["charge2"]) &
            (has_jet_pair | has_fatjet)
        )

        # hook to update the event mask base on additional event info
        event_mask = self.update_event_mask(events, event_mask)

        # apply to all arrays needed until now
        _events = events[event_mask]
        pair_type = pair_type[event_mask]
        vis_tau1, vis_tau2 = vis_tau1[event_mask], vis_tau2[event_mask]
        tautau_mask = tautau_mask[event_mask]
        dm1, dm2 = dm1[event_mask], dm2[event_mask]
        has_jet_pair, has_fatjet = has_jet_pair[event_mask], has_fatjet[event_mask]

        # prepare network inputs
        cont = DotDict()
        cat = DotDict()
        
        """
        - update feature update hook to accept and return two dicts
        """

        # compute angle from visible mother particle of vis_tau1 and vis_tau2
        # used to rotate the kinematics of dau{1,2}, met, bjet{1,2} and fatjets relative to it
        phi_lep = np.arctan2(vis_tau1.py + vis_tau2.py, vis_tau1.px + vis_tau2.px, dtype=np.float64)

        # MET variables
        _met = _events[self.config_inst.x.met_name]
        cont.met_px, cont.met_py = rotate_to_phi(
            phi_lep,
            _met.pt * np.cos(_met.phi),
            _met.pt * np.sin(_met.phi),
        )
        cont.met_cov00, cont.met_cov01, cont.met_cov11 = _met.covXX, _met.covXY, _met.covYY

        # lepton 1
        cont.vis_tau1_px, cont.vis_tau1_py = rotate_to_phi(phi_lep, vis_tau1.px, vis_tau1.py)
        cont.vis_tau1_pz, cont.vis_tau1_e = vis_tau1.pz, vis_tau1.energy

        # lepton 2
        cont.vis_tau2_px, cont.vis_tau2_py = rotate_to_phi(phi_lep, vis_tau2.px, vis_tau2.py)
        cont.vis_tau2_pz, cont.vis_tau2_e = vis_tau2.pz, vis_tau2.energy

        # there might be less than two jets or no fatjet, so pad them
        bjets = ak.pad_none(_events.HHBJet, 2, axis=1)
        fatjet = ak.pad_none(_events.FatJet, 1, axis=1)[:, 0]

        # bjet 1
        cont.bjet1_px, cont.bjet1_py = rotate_to_phi(phi_lep, bjets[:, 0].px, bjets[:, 0].py)
        cont.bjet1_pz, cont.bjet1_e = bjets[:, 0].pz, bjets[:, 0].energy
        if self.btag_type == "pnet":
            cont.bjet1_tag_b = bjets[:, 0].btagPNetB
            cont.bjet1_tag_cvsb = bjets[:, 0].btagPNetCvB
            cont.bjet1_tag_cvsl = bjets[:, 0].btagPNetCvL
        elif self.btag_type == "upart":
            cont.bjet1_tag_b = bjets[:, 0].btagUParTAK4B
        cont.bjet1_hhbtag = bjets[:, 0].hhbtag

        # bjet 2
        cont.bjet2_px, cont.bjet2_py = rotate_to_phi(phi_lep, bjets[:, 1].px, bjets[:, 1].py)
        cont.bjet2_pz, cont.bjet2_e = bjets[:, 1].pz, bjets[:, 1].energy
        if self.btag_type == "pnet":
            cont.bjet2_tag_b = bjets[:, 1].btagPNetB
            cont.bjet2_tag_cvsb = bjets[:, 1].btagPNetCvB
            cont.bjet2_tag_cvsl = bjets[:, 1].btagPNetCvL
        elif self.btag_type == "upart":
            cont.bjet2_tag_b = bjets[:, 1].btagUParTAK4B
        cont.bjet2_hhbtag = bjets[:, 1].hhbtag

        # fatjet variables
        cont.fatjet_px, cont.fatjet_py = rotate_to_phi(phi_lep, fatjet.px, fatjet.py)
        cont.fatjet_pz, cont.fatjet_e = fatjet.pz, fatjet.energy

        # mask values as done during training of the network
        def mask_values(mask, value, *fields):
            if not ak.any(mask):
                return
            for field in fields:
                if field not in cont:
                    continue
                arr = flat_np_view(ak.fill_none(cont[field], value, axis=0), copy=True)
                arr[flat_np_view(mask)] = value
                cont[field] = layout_ak_array(arr, cont[field]) if cont[field].ndim > 1 else arr

        mask_values(~has_jet_pair, 0.0, "bjet1_px", "bjet1_py", "bjet1_pz", "bjet1_e")
        mask_values(~has_jet_pair, 0.0, "bjet2_px", "bjet2_py", "bjet2_pz", "bjet2_e")
        mask_values(~has_jet_pair, -1.0, "bjet1_tag_b", "bjet1_tag_cvsb", "bjet1_tag_cvsl", "bjet1_hhbtag")
        mask_values(~has_jet_pair, -1.0, "bjet2_tag_b", "bjet2_tag_cvsb", "bjet2_tag_cvsl", "bjet2_hhbtag")
        mask_values(~has_fatjet, 0.0, "fatjet_px", "fatjet_py", "fatjet_pz", "fatjet_e")

        # combine daus
        cont.htt_e = cont.vis_tau1_e + cont.vis_tau2_e
        cont.htt_px = cont.vis_tau1_px + cont.vis_tau2_px
        cont.htt_py = cont.vis_tau1_py + cont.vis_tau2_py
        cont.htt_pz = cont.vis_tau1_pz + cont.vis_tau2_pz

        # combine bjets
        cont.hbb_e = cont.bjet1_e + cont.bjet2_e
        cont.hbb_px = cont.bjet1_px + cont.bjet2_px
        cont.hbb_py = cont.bjet1_py + cont.bjet2_py
        cont.hbb_pz = cont.bjet1_pz + cont.bjet2_pz
        mask_values(~has_jet_pair, 0.0, "hbb_e", "hbb_px", "hbb_py", "hbb_pz")

        # htt + hbb
        cont.htthbb_e = cont.htt_e + cont.hbb_e
        cont.htthbb_px = cont.htt_px + cont.hbb_px
        cont.htthbb_py = cont.htt_py + cont.hbb_py
        cont.htthbb_pz = cont.htt_pz + cont.hbb_pz
        mask_values(~has_jet_pair, 0.0, "htthbb_e", "htthbb_px", "htthbb_py", "htthbb_pz")

        # htt + fatjet
        cont.httfatjet_e = cont.htt_e + cont.fatjet_e
        cont.httfatjet_px = cont.htt_px + cont.fatjet_px
        cont.httfatjet_py = cont.htt_py + cont.fatjet_py
        cont.httfatjet_pz = cont.htt_pz + cont.fatjet_pz
        mask_values(~has_fatjet, 0.0, "httfatjet_e", "httfatjet_px", "httfatjet_py", "httfatjet_pz")

        # assign categorical inputs via names too
        cat.pair_type = pair_type
        cat.dm1 = dm1
        cat.dm2 = dm2
        cat.vis_tau1_charge = vis_tau1.charge
        cat.vis_tau2_charge = vis_tau2.charge
        cat.has_jet_pair = has_jet_pair
        cat.has_fatjet = has_fatjet

        # optionally update features
        cont, cat = self.update_features(cont, cat, events[event_mask], phi_lep)

        # build continuous inputs
        # (order exactly as documented in link above)
        continuous_inputs = [
            np.asarray(t[..., None], dtype=np.float32)
            for t in cont.values()
            if t is not None
        ]

        # build categorical inputs
        # (order exactly as documented in link above)
        categorical_inputs = [
            np.asarray(t[..., None], dtype=np.int32)
            for t in cat.values()
            if t is not None
        ]
        # evaluate the model
        scores = self.evaluator(
            self.cls_name,
            np.concatenate(categorical_inputs, axis=1),
            np.concatenate(continuous_inputs, axis=1),
        )

        # sanitize scores (probably replacing nans)
        scores = self.sanitize_scores(scores)

        # store scores in events
        events = self.store_scores(events, scores, event_mask)

        if self.produce_features:
            # store input columns for sync
            for name, vals in cont.items():
                values = self.empty_value * np.ones(len(events), dtype=np.float32)
                values[event_mask] = ak.flatten(np.asarray(vals[..., None], dtype=np.float32))
                events = set_ak_column_f32(events, f"{self.features_prefix}{self.cls_name}_{name}", values)
            for name, vals in cat.items():
                values = int(self.empty_value) * np.ones(len(events), dtype=np.int32)
                values[event_mask] = ak.flatten(np.asarray(vals[..., None], dtype=np.int32))
                events = set_ak_column_i32(events, f"{self.features_prefix}{self.cls_name}_{name}", values)

        return events

    def update_features(self, cont, cat, events, phi_lep):
        return cont, cat

    def sanitize_scores(self, scores: Any) -> Any:
        # in very rare cases (1 in 25k), the network output can be none, likely for numerical reasons,
        # so issue a warning and set them to a default value
        nan_mask = ~np.isfinite(scores)
        if np.any(nan_mask):
            logger.warning(
                f"{nan_mask.sum() // scores.shape[1]} out of {scores.shape[0]} events have NaN scores; "
                f"setting them to {self.empty_value}",
            )
            scores[nan_mask] = self.empty_value

        return scores

    def store_scores(self, events: ak.Array, scores: Any, event_mask: ak.Array) -> ak.Array:
        # prepare output columns with the shape of the original events and assign values into them
        for i, column in enumerate(self.output_columns):
            values = self.empty_value * np.ones(len(events), dtype=np.float32)
            values[event_mask] = scores[:, i]
            events = set_ak_column_f32(events, column, values)

        return events

    def update_event_mask(self, events: ak.Array, event_mask: ak.Array) -> ak.Array:
        return event_mask


class torch_test_dnn(_external_dnn):
    exposed = True

class torch_simple_kl01(_external_dnn):
    exposed = True


#
# end-to-end model tests
#

class _e2e_dnn(_external_dnn):

    latent_dim = 50

    def init_func(self, **kwargs) -> None:
        super(_e2e_dnn, self).init_func(**kwargs)

        # store names of output columns for latent scores
        self.latent_output_columns = [
            f"{self.output_prefix}_bin{i}"
            for i in range(self.latent_dim)
        ]
        self.produces |= set(self.latent_output_columns)

    def sanitize_scores(self, scores: Any) -> Any:
        # scores is a tuple of two arrays of scores that have no softmax applied yet, so apply it first, then perform
        # the usual checks
        return type(scores)(
            super(_e2e_dnn, self).sanitize_scores(scipy.special.softmax(_scores, axis=1))
            for _scores in scores
        )

    def store_scores(self, events: ak.Array, scores: Any, event_mask: ak.Array) -> ak.Array:
        process_scores, latent_scores = scores

        # check the latent dimension
        if latent_scores.shape[1] != self.latent_dim:
            raise ValueError(
                f"expected latent scores to have dimension {self.latent_dim}, but got {latent_scores.shape[1]}",
            )

        # store the multi-class scores as usual
        events = super(_e2e_dnn, self).store_scores(events, process_scores, event_mask)

        # store latent scores
        for i, column in enumerate(self.latent_output_columns):
            values = self.empty_value * np.ones(len(events), dtype=np.float32)
            values[event_mask] = latent_scores[:, i]
            events = set_ak_column_f32(events, column, values)

        return events


class e2e_model1(_e2e_dnn):
    exposed = True

class torch_test_dnn_be(_external_dnn):
    exposed = True

class torch_dense_0(_external_dnn):
    exposed = True

class torch_network_0(_external_dnn):
    exposed = True

class torch_network_1(_external_dnn):
    exposed = True

class torch_network_4(_external_dnn):
    exposed = True

class torch_network_4_sam(_external_dnn):
    exposed = True

class torch_simple_dense_1(_external_dnn):
    exposed = True

class torch_lbn_2_kl0(_external_dnn):
    exposed = True

class torch_lbn_1_kl1(_external_dnn):
    exposed = True

class torch_lbn_1_kl5(_external_dnn):
    exposed = True

class torch_lbn_1_kl2p45(_external_dnn):
    exposed = True

class torch_lbn_1_kl0_pairs_kl1(_external_dnn):
    exposed = True

class torch_lbn_1_kl0_pairs_kl2p45(_external_dnn):
    exposed = True

class torch_lbn_1_kl0_pairs_kl5(_external_dnn):
    exposed = True

class torch_lbn_2_kl0_pairs_kl1(_external_dnn):
    exposed = True

class torch_lbn_2_kl0_pairs_kl5(_external_dnn):
    exposed = True

class torch_lbn_1_all_kl(_external_dnn):
    exposed = True

class torch_lbn_3_kl0_prod20(_external_dnn):
    exposed = True

class torch_lbn_1_kl0_prod14_22pre(_external_dnn):
    exposed = True

class torch_lbn_1_kl1_pair_kl2p45(_external_dnn):
    exposed = True

class torch_lbn_1_kl0_prod20_fixed(_external_dnn):
    exposed = True

class torch_lbn_debug_test(_external_dnn):
    exposed = True

class torch_lbn_1_kl0_kl5_bkg_weighted_up(_external_dnn):
    exposed = True

class torch_lbn_1_kl0_kl1_bkg_weighted_up(_external_dnn):
    exposed = True

class torch_lbn_1_kl0_kl2p45_bkg_weighted_up(_external_dnn):
    exposed = True

class torch_lbn_1_kl0_bkg_weighted_up(_external_dnn):
    exposed = True

class torch_lbn_1_kl1_bkg_weighted_up(_external_dnn):
    exposed = True

class torch_lbn_1_kl2p45_bkg_weighted_up(_external_dnn):
    exposed = True

class torch_lbn_1_kl5_bkg_weighted_up(_external_dnn):
    exposed = True

class torch_lbn_1_all_kl_bkg_weighted_up(_external_dnn):
    exposed = True

class torch_lbn_4_kl0_prod20(_external_dnn):
    exposed = True

class torch_lbn_1_kl0_prod20_vbf_cut(_external_dnn):
    exposed = True

class torch_lbn_1_kl0_weight_matrix_prod20_vbf(_external_dnn):
    exposed = True

class torch_lbn_1_prod20vbf_kl0_diag111(_external_dnn):
    exposed = True

class torch_lbn_1_prod20vbf_kl0_diag155(_external_dnn):
    exposed = True

class torch_lbn_2_prod20vbf_kl0_diag111(_external_dnn):
    exposed = True

class torch_lbn_1_prod20vbf_kl0_wmtest1(_external_dnn):
    exposed = True

class torch_lbn_1_prod20vbf_kl0_diag111_flats_test(_external_dnn):
    exposed = True

class torch_lbn_1_prod20vbf_kl1_diag111_flats_test(_external_dnn):
    exposed = True

class torch_lbn_1_prod20vbf_kl2p45_diag111_for_flat_s(_external_dnn):
    exposed = True

class torch_lbn_1_prod20vbf_kl5_diag111_for_flat_s(_external_dnn):
    exposed = True

class torch_lbn_1_prod20vbf_kl0_kl1_diag111_for_flat_s(_external_dnn):
    exposed = True

class torch_lbn_1_prod20vbf_kl1_diag111_no_btag(_external_dnn):
    exposed = True

class torch_lbn_1_prod20vbf_kl0_diag111_variance_test_1(_external_dnn):
    exposed = True

class torch_lbn_1_prod20vbf_kl0_diag111_variance_test_2(_external_dnn):
    exposed = True

class torch_lbn_1_prod20vbf_kl0_diag111_variance_test_3(_external_dnn):
    exposed = True

class torch_lbn_1_prod20vbf_kl0_diag111_variance_test_4(_external_dnn):
    exposed = True

class torch_lbn_1_prod20vbf_kl0_diag111_variance_test_5(_external_dnn):
    exposed = True

class a_bognet_test(_external_dnn):
    exposed = True

class torch_sig_loss_big_batch(_external_dnn):
    exposed = True

class torch_kl1_kt1_sig_loss_big_batch(_external_dnn):
    exposed = True

class torch_kl1_kt1_sig_loss_very_big(_external_dnn):
    exposed = True

class kl0_flat_s_test_1(_external_dnn):
    exposed = True

class kl1_flat_s_test_2(_external_dnn):
    exposed = True

class Bogmod_1(_external_dnn):
    exposed = True

class cross_entropy(_external_dnn):
    exposed = True

class cross_entropy_with_checkpoint(_external_dnn):
    exposed = True

class signal_focus(_external_dnn):
    exposed = True

class background_focus(_external_dnn):
    exposed = True

class ce_plus_sig_miss(_external_dnn):
    exposed = True

class ce_plus_bkg_miss(_external_dnn):
    exposed = True

class ce_plus_any_miss(_external_dnn):
    exposed = True

class bkg_focus_plus_bkg_miss(_external_dnn):
    exposed = True

class bkg_focus_plus_bkg_miss_plus_bkg_cross_on(_external_dnn):
    exposed = True

class all_bkg(_external_dnn):
    exposed = True

class all_bkg_plus_sig_miss(_external_dnn):
    exposed = True

class new_loss_implementation(_external_dnn):
    exposed = True

class new_implementation(_external_dnn):
    exposed = True

class has_neutrinos(_external_dnn):
    uses = _external_dnn.uses | {"reg_dnn_moe_nu*"}
    require_producers = ["reg_dnn_moe"]    

    def update_features(self, cont, cat, events, phi_lep):
        cont, cat = super().update_features(cont, cat, events, phi_lep)
        # add regressed neutrinos
        for num in ('1','2'):
            for comp in ('px','py','pz'):
                cont[f'nu{num}_{comp}'] = events[f'reg_dnn_moe_nu{num}_{comp}']
        return cont, cat

class test_regressed_nu(has_neutrinos):
    exposed = True

class bg_bgmiss_bgcross_with_nu(has_neutrinos):
    exposed = True

class bg_bgmiss_bgcross_nosig_with_nu(has_neutrinos):
    exposed = True

class bg_bgmiss_bgcross_with_nu_kl1(has_neutrinos):
    exposed = True

class bg_bgmiss_bgcross_with_nu_kl2p45(has_neutrinos):
    exposed = True

class bg_bgmiss_bgcross_with_nu_kl5(has_neutrinos):
    exposed = True

class bg_bgmiss_bgcross_with_nu_kl0(has_neutrinos):
    exposed = True

class bg_bgmiss_bgcross_with_nu_kl0_lr_adjust(has_neutrinos):
    exposed = True

class ce_with_nu_kl1(has_neutrinos):
    exposed = True

class ce_with_nu_kl1_v2(has_neutrinos):
    exposed = True

class ce_with_nu_kl1_v3(has_neutrinos):
    exposed = True

class ce_with_nu_kl1_v4(has_neutrinos):
    exposed = True

class ce_with_nu_kl2p45(has_neutrinos):
    exposed = True

class ce_with_nu_kl5(has_neutrinos):
    exposed = True

class ce_with_nu_kl0(has_neutrinos):
    exposed = True

class ce_diag1111_kl0_kl1(has_neutrinos):
    exposed = True

class ce_diag1111_kl0_kl1_uneven_importance(has_neutrinos):
    exposed = True

class wm_all1s_kl0_kl1(has_neutrinos):
    exposed = True

class wm_diag11p5p5_kl0_kl1(has_neutrinos):
    exposed = True

class wm_diag11p5p5_kl0_kl1_uneven_importance(has_neutrinos):
    exposed = True

class ce_diag1111_kl0_kl1_fixed_v2(has_neutrinos):
    exposed = True

class ce_diag111_kl1_fixed_v1(has_neutrinos):
    exposed = True

class ce_diag1111_kl0_kl1_uneven(has_neutrinos):
    exposed = True

class diag1155_kl0_kl1(has_neutrinos):
    exposed = True

class diag11p5p5_kl0_kl1(has_neutrinos):
    exposed = True

class diag111_diag11_kl0_kl1_test(has_neutrinos):
    exposed = True

class diag111_diag11_kl0_kl1_test_2(has_neutrinos):
    exposed = True


