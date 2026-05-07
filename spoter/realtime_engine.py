import torch
import numpy as np
import time
import json

try:
    from .normalization.body_normalization import normalize_single_dict as normalize_single_body_dict
    from .normalization.hand_normalization import normalize_single_dict as normalize_single_hand_dict
    from .normalization.body_normalization import BODY_IDENTIFIERS
    from .normalization.hand_normalization import HAND_IDENTIFIERS
except ImportError:
    print("⚠️ Warning: Could not import normalization modules. Check your folder structure.")

class SignLanguageEngine:
    def __init__(self, model_path, label_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 1. Load Model
        self.model = torch.jit.load(model_path, map_location=self.device)
        self.model.eval()

        # 2. Warm-up (ฉีดไนตรัสรอไว้เลย)
        print("🔥 Warming up model...")
        dummy = torch.randn(64, 108).to(self.device)
        for _ in range(5):
            _ = self.model(dummy)
        print("🏁 Ready to race!")
        
        # 3. Load Labels
        with open(label_path, 'r', encoding='utf-8') as f:
            self.label_map = json.load(f)

        # เตรียม Identifiers ไว้ล่วงหน้า
        self.hand_ids = [id + "_0" for id in HAND_IDENTIFIERS] + [id + "_1" for id in HAND_IDENTIFIERS]
        self.all_ids = BODY_IDENTIFIERS + self.hand_ids

    def _internal_preprocess(self, numpy_array: np.ndarray) -> torch.Tensor:
        # Step 1: Reshape
        data = numpy_array.reshape(64, 54, 2)
        
        # Step 2: Create Dict (แบบรวดเร็ว)
        depth_map_dict = {identifier: data[:, idx] for idx, identifier in enumerate(self.all_ids)}

        # Step 3: Normalize (Hand ก่อน Body)
        depth_map_dict = normalize_single_hand_dict(depth_map_dict)
        depth_map_dict = normalize_single_body_dict(depth_map_dict)

        # Step 4: Reconstruct Array (ใช้ stack แทนการวนลูป append)
        output_list = [depth_map_dict[id] for id in self.all_ids]
        output = np.stack(output_list, axis=1)

        # Step 5: Convert to Tensor & Shift
        res = torch.from_numpy(output).float().to(self.device)
        return (res - 0.5).view(64, 108)

    def run_inference(self, raw_data_64_108, overall_start):
        """
        รับข้อมูล 64x108 และส่งคืนผลลัพธ์พร้อมสถิติเวลา
        """
        # overall_start = time.perf_counter()
        
        # 1. Preprocess
        preprocessed_data = self._internal_preprocess(raw_data_64_108)
        
        # 2. Predict
        predict_start = time.perf_counter()
        with torch.no_grad():
            outputs = self.model(preprocessed_data)
            prediction = torch.argmax(outputs, dim=1).item()
            conf = torch.softmax(outputs, dim=1).max().item()
        predict_end = time.perf_counter()
        
        overall_total = (time.perf_counter() - overall_start) * 1000
        predict_only = (predict_end - predict_start) * 1000
        
        return {
            "label": self.label_map.get(str(prediction), "Unknown"),
            "confidence": f"{conf:.2%}",
            "predict_time_ms": round(predict_only, 2),
            "total_process_time_ms": round(overall_total, 2)
        }