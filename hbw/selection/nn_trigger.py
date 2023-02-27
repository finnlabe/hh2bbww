# coding: utf-8

"""
NN trigger methods for HHtobbWW.
At the moment, just some tests
"""

from collections import defaultdict
from typing import Tuple

from columnflow.util import maybe_import, dev_sandbox

from columnflow.columnar_util import set_ak_column

from columnflow.selection import Selector, SelectionResult, selector
from columnflow.production import Producer, producer

np = maybe_import("numpy")
ak = maybe_import("awkward")
keras = maybe_import("keras")
pickle = maybe_import("pickle")

# this producer will run the DNN inference and will produce the L1NNscore output value
@producer(
    uses={
        "L1Jet.pt", "L1Jet.eta", "L1Jet.phi",
        "L1Mu.pt", "L1Mu.eta", "L1Mu.phi",
        "L1EG.pt", "L1EG.eta", "L1EG.phi",
        "L1EtSum.pt", "L1EtSum.phi", "L1EtSum.etSumType",
    },
    produces={"L1NNscore"},
    sandbox=dev_sandbox("bash::$HBW_BASE/sandboxes/venv_ml_keras.sh")
)
def NN_trigger_inference(
    self: Producer,
    events: ak.Array,
    **kwargs,
) -> ak.Array:

    # first, importing the network and all info
    # this will be three files:
    # - the model itself
    # - the standard scaler object
    # - some custom objects (optinal)

    # preparing the input vector
    pad_nums = {"L1EG": 2, "L1Mu": 2, "L1Jet": 4}
    variables = {"pt", "eta", "phi"}

    pad_arrs = []
    for obj in pad_nums:
        pad_arr = ak.pad_none(events[obj], pad_nums[obj], clip=True)

        for i in range(pad_nums[obj]):
            for var in variables:
                pad_arrs += [ak.to_numpy(ak.fill_none(pad_arr[var][:,i],0))]
    
    # now, we are only missing the MET inputs
    # for these, we need to read the etSumType to get MET
    kMissingEt = 2
    missing_et_mask = events.L1EtSum.etSumType == kMissingEt

    MET_pt = events.L1EtSum.pt[missing_et_mask]
    MET_phi = events.L1EtSum.phi[missing_et_mask]

    inputs_MET = np.concatenate( (ak.to_numpy(MET_pt).reshape(len(events), 1), ak.to_numpy(MET_phi).reshape(len(events), 1)), axis = 1)
    
    # concatenating everything...
    inputs = np.concatenate( ( inputs_MET, ak.to_numpy(np.stack(pad_arrs)).T ), axis = 1 )
    
    # scaling the input vector
    scaled_inputs = self.scaler.transform(inputs)

    # running the inference
    predictions = self.model.predict(scaled_inputs).flatten()

    # returning the inference results
    events = set_ak_column(events, "L1NNscore", predictions)

    return events

# the following code assures that the required external files are loaded
@NN_trigger_inference.requires
def NN_trigger_inference_requires(self: Producer, reqs: dict) -> None:
    if "external_files" in reqs: return

    from columnflow.tasks.external import BundleExternalFiles
    reqs["external_files"] = BundleExternalFiles.req(self.task)

# set up the producer
@NN_trigger_inference.setup
def NN_trigger_inference_setup(self: Producer, reqs: dict, inputs: dict) -> None:
    bundle = reqs["external_files"]

    # loading the model
    self.model = keras.models.load_model(bundle.files.L1NN_network.path) # here, custom objects might be added later

    # loading the scaler
    with open(bundle.files.L1NN_scaler.path, 'rb') as inp: self.scaler = pickle.load(inp)


# this selector uses the L1NNscore column to make a L1 trigger cut
@selector(
    uses={"L1NNscore", NN_trigger_inference},
    produces={"L1NNscore", NN_trigger_inference},
    exposed=True,
    sandbox=dev_sandbox("bash::$HBW_BASE/sandboxes/venv_ml_keras.sh"),
)
def NN_trigger_selection(
    self: Selector,
    events: ak.Array,
    stats: defaultdict,
    **kwargs,
) -> Tuple[ak.Array, SelectionResult]:

    # this selector will perform a selection on the L1NNscore

    # add L1NN score
    events = self[NN_trigger_inference](events, **kwargs)

    threshold = 0.
    greater = True

    # defining some individual masks
    if(greater): L1NNscoremask = events.L1NNscore >= threshold
    else: L1NNscoremask = events.L1NNscore <= threshold

    return events, SelectionResult(
        steps={"L1NNcut":L1NNscoremask}
    )