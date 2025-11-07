# adapters/ml_infer.py
import numpy as np
import cv2

class TFLitePoopResidueClassifier:
    def __init__(self, model_path: str):
        import tflite_runtime.interpreter as tflite
        self.interp = tflite.Interpreter(model_path=model_path, num_threads=2)
        self.interp.allocate_tensors()
        self.in_idx = self.interp.get_input_details()[0]['index']
        self.out_idx = self.interp.get_output_details()[0]['index']

    def _prep(self, img_bgr: np.ndarray) -> np.ndarray:
        x = cv2.resize(img_bgr, (224,224))
        x = x[:, :, ::-1].astype(np.float32) / 255.0  # BGR->RGB
        x = np.expand_dims(x, 0)
        return x

    def predict(self, roi_bgr: np.ndarray):
        from domain.models import PoopResidueScore
        x = self._prep(roi_bgr)
        self.interp.set_tensor(self.in_idx, x)
        self.interp.invoke()
        y = self.interp.get_tensor(self.out_idx)[0]  # shape (1,)
        return PoopResidueScore(prob_poop=float(y))

class TFLiteCoprophagyClassifier:
    def __init__(self, model_path: str):
        import tflite_runtime.interpreter as tflite
        self.interp = tflite.Interpreter(model_path=model_path, num_threads=2)
        self.interp.allocate_tensors()
        self.in_idx = self.interp.get_input_details()[0]['index']
        self.out_idx = self.interp.get_output_details()[0]['index']

    def _prep_pair(self, before_bgr, after_bgr):
        b = cv2.resize(before_bgr, (224,224))[:, :, ::-1].astype(np.float32)/255.0
        a = cv2.resize(after_bgr,  (224,224))[:, :, ::-1].astype(np.float32)/255.0
        x = np.concatenate([b, a], axis=2)  # 224×224×6
        x = np.expand_dims(x, 0)
        return x

    def predict_pair(self, before_bgr, after_bgr):
        from domain.models import CoprophagyScore
        x = self._prep_pair(before_bgr, after_bgr)
        self.interp.set_tensor(self.in_idx, x)
        self.interp.invoke()
        y = self.interp.get_tensor(self.out_idx)[0]
        return CoprophagyScore(prob_eaten=float(y))