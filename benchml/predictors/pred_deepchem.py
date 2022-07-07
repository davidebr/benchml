from abc import ABC

import numpy as np
import warnings

from benchml.pipeline import FitTransform

try:
    import deepchem as dc 
    import deepchem.models as dcmodels

except ImportError:
    dc = None


def check_deepchem_available(obj, require=False):
    if dc  is None:
        if require:
            raise ImportError("%s requires deepchem" % obj.__name__)
        return False
    return True


class DeepchemTransform(FitTransform, ABC):
    def check_available(self, *args, **kwargs):
        return check_deepchem_available(self, *args, **kwargs)


class GCNNBinaryClassifier(DeepchemTransform):
    default_args = { 
        'graph_conv_layers': [64,64],
        'dense_layer_size':128,
        'batch_size':100,
        'nb_epoch':5 } 
    req_inputs = ("X", "y")
    allow_params = {
        "model",
    }
    allow_stream = {"y", "z"}

    def _fit(self, inputs, stream, params):
        y_train = inputs["y"]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = dcmodels.GraphConvModel(1, mode='classification',  # 1 means binary classifier
                                  batch_size=self.args['batch_size'], 
                                  dense_layer_size=self.args['dense_layer_size'],
                                  graph_conv_layers=self.args['graph_conv_layers']) 
            # now create a disk dataset
            dds=dc.data.DiskDataset.from_numpy(inputs["X"],y_train) 
            loss=model.fit(dds,nb_epoch=self.args['nb_epoch'])
            z = model.predict(dds)[:,0,:]  # ndata,nclasses,[neg,pos]
        y = [1 if r[1]>r[0] else 0 for r in z] # assume 0 1 labels... may not be true 
        params.put("model", model)
        stream.put("y", y)
        stream.put("z", [ v[1] for v in z]) # convert in a proba

    def _map(self, inputs, stream):
        model = self.params().get("model")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dds=dc.data.DiskDataset.from_numpy(inputs["X"]) 
            z = model.predict(dds)[:,0,:]
        y = [1 if r[1]>r[0] else 0 for r in z] # assume 0 1 labels... may not be true 
        stream.put("y", y)
        stream.put("z", [ v[1] for v in z]) # convert in a proba

