import numpy as np

from benchml.kernels import KernelDot
from benchml.pipeline import Macro, Transform
from benchml.plugins.plugin_check import deepchem, check_deepchem_available
from benchml.utils import get_smiles

class ConvMol(Transform):
    default_args = {
        "use_chirality": False
    }
    allow_stream = {"X"}
    stream_samples = ("X",)
    precompute = True

    def check_available(self, *args, **kwargs):
        return check_deepchem_available(self, *args, **kwargs)  

    def _setup(self):
        self.use_chirality = self.args["use_chirality"]

    def _map(self, inputs, stream):
        configs = inputs["configs"]
        cms=deepchem.feat.ConvMolFeaturizer().featurize([get_smiles(c) for c in configs],use_chirality=self.use_chirality)
        stream.put("X", cms)

