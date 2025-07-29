import json

import numpy as np

class FederatedModel(json.JSONEncoder):
    
    def default(self, o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        return json.JSONEncoder.default(self, o)
