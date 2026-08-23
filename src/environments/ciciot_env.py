"""8-class Tabular IoT Packet Features Environment (CIC-IoT Agent)."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Union
import numpy as np

from src.environments.base import BaseEnvironment, TaskQuery, TaskResult

CICIOT_CLASSES = [
    "DDoS-ICMP_Flood",
    "DDoS-UDP_Flood",
    "DDoS-TCP_Flood",
    "DDoS-SYN_Flood",
    "DDoS-PSHACK_Flood",
    "DDoS-RSTFINFlood",
    "DDoS-HTTP_Flood",
    "BenignTraffic",
]

CICIOT_CONTINUOUS_FEATURES = [
    "flow_duration",
    "Header_Length",
    "Duration",
    "Rate",
    "Srate",
    "Drate",
    "ack_count",
    "syn_count",
    "fin_count",
    "urg_count",
    "rst_count",
    "Tot_sum",
    "Min",
    "Max",
    "AVG",
    "Std",
    "Tot_size",
    "IAT",
    "Number",
    "Magnitude",
    "Radius",
    "Covariance",
    "Variance",
    "Weight",
]

CICIOT_DISCRETE_FEATURES = [
    "Protocol_Type",
    "fin_flag_number",
    "syn_flag_number",
    "rst_flag_number",
    "psh_flag_number",
    "ack_flag_number",
    "ece_flag_number",
    "cwr_flag_number",
    "HTTP",
    "HTTPS",
    "DNS",
    "Telnet",
    "SMTP",
    "SSH",
    "IRC",
    "TCP",
    "UDP",
    "DHCP",
    "ARP",
    "ICMP",
    "IPv",
    "LLC",
]

ALL_FEATURE_NAMES = CICIOT_CONTINUOUS_FEATURES + CICIOT_DISCRETE_FEATURES


def canonical_label(name: str) -> str:
    """Normalize label for robust invariant comparison."""
    if not name:
        return ""
    clean = (
        name.lower()
        .replace("_", "")
        .replace("-", "")
        .replace(" ", "")
        .replace("[", "")
        .replace("]", "")
        .replace("'", "")
        .replace('"', "")
        .strip()
    )
    return clean


class CICIOTEnvironment(BaseEnvironment):
    """8-class Tabular IoT Network Traffic Classification Environment."""

    def __init__(
        self,
        classes: Optional[List[str]] = None,
        continuous_features: Optional[List[str]] = None,
        discrete_features: Optional[List[str]] = None,
    ) -> None:
        self.classes = classes or list(CICIOT_CLASSES)
        self.continuous_features = continuous_features or list(CICIOT_CONTINUOUS_FEATURES)
        self.discrete_features = discrete_features or list(CICIOT_DISCRETE_FEATURES)
        self.all_features = self.continuous_features + self.discrete_features

    def generate_synthetic_features_for_class(
        self,
        traffic_type: str,
        rng: Optional[np.random.RandomState] = None,
    ) -> Dict[str, Any]:
        """Generate a realistic feature dictionary for a specific traffic class."""
        if rng is None:
            rng = np.random.RandomState()

        feat: Dict[str, Any] = {}

        for df in self.discrete_features:
            feat[df] = 0
        feat["IPv"] = 1
        feat["LLC"] = 1

        if traffic_type == "DDoS-ICMP_Flood":
            feat["ICMP"] = 1
            feat["Protocol_Type"] = 1
            rate = float(rng.uniform(800.0, 3000.0))
            feat["Rate"] = rate
            feat["Srate"] = rate * 0.95
            feat["Drate"] = rate * 0.05
            feat["flow_duration"] = float(rng.uniform(0.01, 2.0))
            feat["Header_Length"] = int(rng.randint(20, 40))
            feat["Duration"] = 64
            feat["ack_count"] = 0
            feat["syn_count"] = 0
            feat["fin_count"] = 0
            feat["urg_count"] = 0
            feat["rst_count"] = 0

        elif traffic_type == "DDoS-UDP_Flood":
            feat["UDP"] = 1
            feat["Protocol_Type"] = 17
            rate = float(rng.uniform(600.0, 2500.0))
            feat["Rate"] = rate
            feat["Srate"] = rate * 0.98
            feat["Drate"] = rate * 0.02
            feat["flow_duration"] = float(rng.uniform(0.05, 3.0))
            feat["Header_Length"] = int(rng.randint(20, 50))
            feat["Duration"] = 64
            feat["ack_count"] = 0
            feat["syn_count"] = 0
            feat["fin_count"] = 0
            feat["urg_count"] = 0
            feat["rst_count"] = 0

        elif traffic_type == "DDoS-SYN_Flood":
            feat["TCP"] = 1
            feat["Protocol_Type"] = 6
            feat["syn_flag_number"] = 1
            rate = float(rng.uniform(500.0, 2000.0))
            feat["Rate"] = rate
            feat["Srate"] = rate * 0.99
            feat["Drate"] = 0.0
            feat["flow_duration"] = float(rng.uniform(0.01, 1.5))
            feat["Header_Length"] = int(rng.randint(32, 60))
            feat["Duration"] = 64
            feat["syn_count"] = int(rng.randint(50, 500))
            feat["ack_count"] = 0
            feat["fin_count"] = 0
            feat["urg_count"] = 0
            feat["rst_count"] = 0

        elif traffic_type == "DDoS-TCP_Flood":
            feat["TCP"] = 1
            feat["Protocol_Type"] = 6
            rate = float(rng.uniform(400.0, 1800.0))
            feat["Rate"] = rate
            feat["Srate"] = rate * 0.8
            feat["Drate"] = rate * 0.2
            feat["flow_duration"] = float(rng.uniform(0.1, 4.0))
            feat["Header_Length"] = int(rng.randint(40, 80))
            feat["Duration"] = 64
            feat["syn_count"] = int(rng.randint(5, 50))
            feat["ack_count"] = int(rng.randint(10, 100))
            feat["fin_count"] = 0
            feat["urg_count"] = 0
            feat["rst_count"] = 0

        elif traffic_type == "DDoS-PSHACK_Flood":
            feat["TCP"] = 1
            feat["Protocol_Type"] = 6
            feat["psh_flag_number"] = 1
            feat["ack_flag_number"] = 1
            rate = float(rng.uniform(500.0, 2200.0))
            feat["Rate"] = rate
            feat["Srate"] = rate * 0.9
            feat["Drate"] = rate * 0.1
            feat["flow_duration"] = float(rng.uniform(0.05, 2.5))
            feat["Header_Length"] = int(rng.randint(40, 80))
            feat["Duration"] = 64
            feat["ack_count"] = int(rng.randint(50, 400))
            feat["syn_count"] = 0
            feat["fin_count"] = 0
            feat["urg_count"] = 0
            feat["rst_count"] = 0

        elif traffic_type == "DDoS-RSTFINFlood":
            feat["TCP"] = 1
            feat["Protocol_Type"] = 6
            feat["rst_flag_number"] = 1
            feat["fin_flag_number"] = 1
            rate = float(rng.uniform(300.0, 1500.0))
            feat["Rate"] = rate
            feat["Srate"] = rate * 0.9
            feat["Drate"] = rate * 0.1
            feat["flow_duration"] = float(rng.uniform(0.02, 2.0))
            feat["Header_Length"] = int(rng.randint(32, 64))
            feat["Duration"] = 64
            feat["rst_count"] = int(rng.randint(30, 300))
            feat["fin_count"] = int(rng.randint(30, 300))
            feat["ack_count"] = 0
            feat["syn_count"] = 0
            feat["urg_count"] = 0

        elif traffic_type == "DDoS-HTTP_Flood":
            feat["TCP"] = 1
            feat["HTTP"] = 1
            feat["Protocol_Type"] = 6
            rate = float(rng.uniform(100.0, 800.0))
            feat["Rate"] = rate
            feat["Srate"] = rate * 0.6
            feat["Drate"] = rate * 0.4
            feat["flow_duration"] = float(rng.uniform(0.5, 10.0))
            feat["Header_Length"] = int(rng.randint(80, 250))
            feat["Duration"] = 128
            feat["ack_count"] = int(rng.randint(20, 200))
            feat["syn_count"] = int(rng.randint(5, 30))
            feat["fin_count"] = int(rng.randint(5, 30))
            feat["urg_count"] = 0
            feat["rst_count"] = 0

        else:  # BenignTraffic
            is_tcp = rng.choice([0, 1])
            feat["TCP"] = int(is_tcp)
            feat["UDP"] = 1 - int(is_tcp)
            feat["Protocol_Type"] = 6 if is_tcp else 17
            if is_tcp and rng.random() > 0.5:
                feat["HTTPS"] = 1
            rate = float(rng.uniform(0.5, 50.0))
            feat["Rate"] = rate
            feat["Srate"] = rate * 0.5
            feat["Drate"] = rate * 0.5
            feat["flow_duration"] = float(rng.uniform(1.0, 30.0))
            feat["Header_Length"] = int(rng.randint(40, 120))
            feat["Duration"] = int(rng.choice([64, 128]))
            feat["ack_count"] = int(rng.randint(1, 20))
            feat["syn_count"] = int(rng.randint(0, 5))
            feat["fin_count"] = int(rng.randint(0, 5))
            feat["urg_count"] = 0
            feat["rst_count"] = 0

        tot_size = float(rng.uniform(100.0, 5000.0))
        num_pkts = float(max(1.0, feat["Rate"] * feat["flow_duration"]))
        feat["Tot_sum"] = float(tot_size * 1.5)
        feat["Min"] = float(rng.uniform(0.0, 40.0))
        feat["Max"] = float(rng.uniform(500.0, 1500.0))
        feat["AVG"] = float(tot_size / max(1.0, num_pkts))
        feat["Std"] = float(rng.uniform(10.0, 100.0))
        feat["Tot_size"] = tot_size
        feat["IAT"] = float(1000.0 / max(0.1, feat["Rate"]))
        feat["Number"] = num_pkts
        feat["Magnitude"] = float(np.sqrt(feat["AVG"] ** 2 + feat["Rate"] ** 2))
        feat["Radius"] = float(feat["Std"] * 1.2)
        feat["Covariance"] = float(rng.uniform(0.0, 50.0))
        feat["Variance"] = float(feat["Std"] ** 2)
        feat["Weight"] = float(num_pkts * 1.1)

        return feat

    def generate_single_query(
        self,
        query_id: str,
        traffic_type: Optional[str] = None,
        rng: Optional[np.random.RandomState] = None,
    ) -> TaskQuery:
        """Generate a single CIC-IoT query with feature vector."""
        if rng is None:
            rng = np.random.RandomState()

        if traffic_type is None:
            traffic_type = str(rng.choice(self.classes))

        features = self.generate_synthetic_features_for_class(traffic_type, rng=rng)
        vector = [float(features.get(f, 0.0)) for f in self.all_features]

        return TaskQuery(
            query_id=query_id,
            query_vector=vector,
            raw_input=features,
            ground_truth=traffic_type,
            features=features,
            metadata={"traffic_type": traffic_type},
        )

    def sample_initial_memory(
        self, n_samples: int = 100, seed: Optional[int] = 42
    ) -> List[TaskQuery]:
        """Sample initial verified memory bank D_0 (N=100)."""
        rng = np.random.RandomState(seed)
        queries: List[TaskQuery] = []
        for i in range(n_samples):
            qid = f"init_ciciot_{i:04d}"
            queries.append(self.generate_single_query(qid, rng=rng))
        return queries

    def sample_stream(
        self, n_samples: int = 1000, seed: Optional[int] = 128
    ) -> List[TaskQuery]:
        """Sample test query stream (N=1000)."""
        rng = np.random.RandomState(seed)
        queries: List[TaskQuery] = []
        for i in range(n_samples):
            qid = f"stream_ciciot_{i:05d}"
            queries.append(self.generate_single_query(qid, rng=rng))
        return queries

    def evaluate(
        self, query: TaskQuery, prediction: Any, raw_output: str = ""
    ) -> TaskResult:
        """Evaluate predicted attack label against ground truth."""
        gt = str(query.ground_truth).strip()
        pred_str = ""

        if isinstance(prediction, str):
            pred_str = prediction.strip()
        elif prediction is not None:
            pred_str = str(prediction).strip()

        norm_gt = canonical_label(gt)
        norm_pred = canonical_label(pred_str)

        is_success = bool(norm_gt and norm_gt == norm_pred)
        score = 1.0 if is_success else 0.0
        error = 0.0 if is_success else 1.0

        return TaskResult(
            query_id=query.query_id,
            prediction=pred_str,
            ground_truth=gt,
            raw_output=raw_output,
            is_success=is_success,
            score=score,
            error=error,
            metadata={
                "normalized_gt": norm_gt,
                "normalized_pred": norm_pred,
            },
        )

    def compute_input_similarity(
        self,
        vec_a: Union[TaskQuery, np.ndarray, List[float]],
        vec_b: Union[TaskQuery, np.ndarray, List[float]],
    ) -> float:
        """Compute relative feature distance similarity per Appendix A.1 / A.3."""
        dict_a = self._extract_feature_dict(vec_a)
        dict_b = self._extract_feature_dict(vec_b)

        if not dict_a or not dict_b:
            return 0.0

        diff_sum = 0.0
        total_feats = len(self.all_features)

        for cf in self.continuous_features:
            val_a = float(dict_a.get(cf, 0.0))
            val_b = float(dict_b.get(cf, 0.0))
            denom = max(abs(val_a), abs(val_b))
            if denom > 1e-7:
                s_cont = abs(val_a - val_b) / denom
            else:
                s_cont = 0.0
            diff_sum += min(1.0, s_cont)

        for df in self.discrete_features:
            val_a = dict_a.get(df, 0)
            val_b = dict_b.get(df, 0)
            s_disc = 1.0 if val_a != val_b else 0.0
            diff_sum += s_disc

        sim_in = 1.0 - (diff_sum / total_feats)
        return float(np.clip(sim_in, 0.0, 1.0))

    def compute_output_similarity(self, out_a: Any, out_b: Any) -> float:
        """Compute output similarity between two agent trajectory outputs."""
        label_a = self._extract_label(out_a)
        label_b = self._extract_label(out_b)

        if not label_a or not label_b:
            return 0.0

        norm_a = canonical_label(label_a)
        norm_b = canonical_label(label_b)

        return 1.0 if norm_a == norm_b else 0.0

    def _extract_feature_dict(
        self, val: Union[TaskQuery, np.ndarray, List[float]]
    ) -> Dict[str, Any]:
        if isinstance(val, TaskQuery):
            if val.features:
                return val.features
            if val.query_vector and len(val.query_vector) == len(self.all_features):
                return {f: val.query_vector[i] for i, f in enumerate(self.all_features)}
            return {}

        arr = np.asarray(val, dtype=np.float64)
        if arr.shape[0] == len(self.all_features):
            return {f: float(arr[i]) for i, f in enumerate(self.all_features)}
        return {}

    @staticmethod
    def _extract_label(out: Any) -> str:
        if isinstance(out, str):
            match = re.search(r"ANSWER:\s*([^\n\r]+)", out, re.IGNORECASE)
            if match:
                return match.group(1).strip()
            return out.strip()
        return str(out)
