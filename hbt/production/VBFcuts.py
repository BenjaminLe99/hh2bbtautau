from columnflow.production import Producer, producer
from columnflow.util import maybe_import, dev_sandbox


np = maybe_import("numpy")
ak = maybe_import("awkward")



@producer(
    uses={
        # nano columns
        "event",
        "GenPart.*",
    },
    sandbox=dev_sandbox("bash::$HBT_BASE/sandboxes/venv_columnar.sh"),
)
def gen_HH_decay_product_VBF_sel(self: Producer, events: ak.Array, genVBFpartonIndices, **kwargs) -> ak.Array:
    """
    Creates a new ragged column "gen_top_decay" with one element per hard top quark. Each element is
    a GenParticleArray with five or more objects in a distinct order: top quark, bottom quark,
    W boson, down-type quark or charged lepton, up-type quark or neutrino, and any additional decay
    produces of the W boson (if any, then most likly photon radiations). Per event, the structure
    will be similar to:

    .. code-block:: python

        [[t1, b1, W1, q1/l, q2/n(, additional_w_decay_products)], [t2, ...], ...]
    """

    # selection mask for the VBF events
    def VBF_sel(events: ak.Array):

        # filter requirements for interesting partons
        abs_id = abs(events.GenPart.pdgId)
        mask = (
            (events.GenPart.status != 21) &
            (events.GenPart.hasFlags("isHardProcess")) &
            (abs_id < 7) &
            ((abs(events.GenPart.distinctParent.pdgId) == 24) | (abs(events.GenPart.distinctParent.pdgId) == 23))
        )

        mask = ak.fill_none(mask, False)
        mask_VBF_events = (np.sum(mask, axis=1) == 0)

        # VBF_partons = events.GenPart[genVBFpartonIndices]

        # mask_row_0_eta = np.where(abs(VBF_partons.eta[:, 0]) < 4.7, True, False)
        # mask_row_0_pt = np.where(VBF_partons.pt[:, 0] > 30.0, True, False)
        # mask_row_0 = np.logical_and(mask_row_0_pt, mask_row_0_eta)
        # mask_row_1_eta = np.where(abs(VBF_partons.eta[:, 1]), True, False)
        # mask_row_1_pt = np.where(VBF_partons.pt[:, 1] > 30.0, True, False)
        # mask_row_1 = np.logical_and(mask_row_1_pt, mask_row_1_eta)

        # kinemtaic_mask = np.logical_and(mask_row_0, mask_row_1)

        # if kinemtaic:
        #     mask_VBF_events = np.logical_and(mask_VBF_events, kinemtaic_mask)

        return mask_VBF_events

    return VBF_sel(events)