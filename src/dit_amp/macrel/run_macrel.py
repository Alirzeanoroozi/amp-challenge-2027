import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from macrel.macrel_features import get_sequence_features
import gzip
import onnxruntime as rt
from pathlib import Path

# paths relative to this script's directory
_script_dir = Path(__file__).resolve().parent
model1_path = _script_dir / "AMP.onnx.gz"
model2_path = _script_dir / "Hemo.onnx.gz"

sess_options = rt.SessionOptions()
sess_options.intra_op_num_threads = 4  # avoid pthread_setaffinity_np errors on HPC
sess_options.inter_op_num_threads = 4

with gzip.open(model1_path, 'rb') as f:
    model1 = rt.InferenceSession(
        f.read(), sess_options=sess_options, providers=["CPUExecutionProvider"]
    )

with gzip.open(model2_path, 'rb') as f:
    model2 = rt.InferenceSession(
        f.read(), sess_options=sess_options, providers=["CPUExecutionProvider"]
    )

def predict(features):
    [amp_prob] = model1.run(['output_probability'], {'input_features': features.astype(np.float32)})
    [hemo_prob] = model2.run(['output_probability'], {'input_features': features.astype(np.float32)})
    return amp_prob[0]['AMP'], hemo_prob[0]['Hemo']

def run_macrel(seq):
    features = np.expand_dims(get_sequence_features(seq), axis=0)
    amp_prob, hemo_prob = predict(features)
    return amp_prob, hemo_prob
