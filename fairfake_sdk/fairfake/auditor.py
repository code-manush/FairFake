"""
FairFake Standalone Auditor Engine (Task 5.1)
"""

import os
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from dataclasses import dataclass, field
import pandas as pd
import numpy as np


STANDARD_ATTRIBUTES = [
    "Male", "Female", "Young", "Senior", "Asian", "White", "Black", "Dark_Skin",
    "Shiny_Skin", "Bald", "Wavy_Hair", "Receding_Hairline", "Bangs",
    "Black_Hair", "Blond_Hair", "Brown_Hair", "No_Beard", "Mustache",
    "Goatee", "Oval_Face", "Square_Face", "Double_Chin", "Chubby",
    "Obstructed_Forehead", "Fully_Visible_Forehead", "Mouth_Closed",
    "Smiling", "Big_Lips", "Big_Nose", "Pointy_Nose", "Heavy_Makeup",
    "Wearing_Hat", "Wearing_Lipstick", "Eyeglasses", "Attractive"
]


def safe_div(n: float, d: float, fallback: float = 0.0) -> float:
    return float(n / d) if d != 0 else fallback


@dataclass
class CohortMetricRecord:
    attribute: str
    sample_count: int
    accuracy: float
    error_rate: float
    pdrp: float
    ddrp: float
    rp: float
    crp: float


@dataclass
class AuditResult:
    global_accuracy: float
    total_samples: int
    cohort_metrics: Dict[str, CohortMetricRecord]
    high_bias_cohorts: List[str]
    max_abs_crp: float
    max_pdrp: float
    max_ddrp: float
    dataset_source: str


class Auditor:
    """
    Modular fairness auditing engine for deepfake detectors.
    Accepts pandas DataFrames, CSV paths, or custom prediction functions.
    """

    def __init__(
        self,
        dataset: Optional[Union[str, pd.DataFrame]] = None,
        model: Optional[Any] = None,
        attributes: Optional[List[str]] = None
    ):
        self.model = model
        self.attributes = attributes or STANDARD_ATTRIBUTES

        if isinstance(dataset, pd.DataFrame):
            self.df = dataset.copy()
            self.source = "DataFrame"
        elif isinstance(dataset, str):
            if not os.path.exists(dataset):
                raise FileNotFoundError(f"Dataset path not found: {dataset}")
            self.df = pd.read_csv(dataset)
            self.source = dataset
        else:
            self.df = None
            self.source = "None"

    def run_audit(
        self,
        threshold: float = 0.5,
        dataset: Optional[Union[str, pd.DataFrame]] = None,
        calibration_map: Optional[Dict[str, float]] = None
    ) -> Any:
        from .report import FairnessReport

        df = self.df if dataset is None else (
            dataset.copy() if isinstance(dataset, pd.DataFrame) else pd.read_csv(dataset)
        )
        if df is None:
            raise ValueError("No dataset provided for audit.")

        # Standardize labels and predictions
        prob_col = "predicted_prob" if "predicted_prob" in df.columns else "fake_probability"
        label_col = "true_label" if "true_label" in df.columns else "label"

        if prob_col not in df.columns or label_col not in df.columns:
            raise ValueError(f"Dataset must contain probability and label columns ('{prob_col}', '{label_col}')")

        # Map labels to 0 (REAL) / 1 (FAKE)
        raw_lbl = df[label_col]
        if raw_lbl.dtype == object or isinstance(raw_lbl.iloc[0], str):
            y_true = raw_lbl.map({"REAL": 0, "FAKE": 1, "0": 0, "1": 1}).fillna(0).astype(int).values
        else:
            y_true = raw_lbl.astype(int).values

        y_prob = df[prob_col].values.astype(float)
        
        # Apply cohort-calibrated thresholds if calibration_map is provided
        if calibration_map:
            cal_thresholds = []
            for _, row in df.iterrows():
                active_t = [
                    calibration_map[attr] for attr in self.attributes
                    if attr in calibration_map and attr in row and row[attr] == 1
                ]
                cal_thresholds.append(float(np.mean(active_t)) if active_t else threshold)
            y_pred = (y_prob >= np.array(cal_thresholds)).astype(int)
        else:
            y_pred = (y_prob >= threshold).astype(int)

        is_error = (y_pred != y_true).astype(int)

        df_eval = df.copy()
        df_eval["_pred"] = y_pred
        df_eval["_error"] = is_error
        df_eval["_true"] = y_true

        global_acc = float((y_pred == y_true).mean())

        # If Dark_Skin missing but Black present, duplicate
        if "Dark_Skin" not in df_eval.columns and "Black" in df_eval.columns:
            df_eval["Dark_Skin"] = df_eval["Black"]

        cohort_records: Dict[str, CohortMetricRecord] = {}
        high_bias = []
        crp_vals = []
        pdrp_vals = []
        ddrp_vals = []

        for attr in self.attributes:
            if attr not in df_eval.columns:
                continue

            with_attr = df_eval[df_eval[attr] == 1]
            without_attr = df_eval[df_eval[attr] == 0]
            cnt = len(with_attr)
            if cnt == 0:
                continue

            err_w = float(with_attr["_error"].mean())
            acc_w = float(1.0 - err_w)
            err_wo = float(without_attr["_error"].mean()) if len(without_attr) > 0 else err_w

            rp_val = max(-3.0, min(3.0, safe_div(err_w, err_wo, 1.0) - 1.0))

            # Matched control group sampling for CRP
            n_ctrl = min(cnt, len(without_attr))
            if n_ctrl > 5:
                cw = with_attr.sample(n=n_ctrl, random_state=42)
                cwo = without_attr.sample(n=n_ctrl, random_state=42)
                err_cw = float(cw["_error"].mean())
                err_cwo = float(cwo["_error"].mean())
                rp_ctrl = max(-3.0, min(3.0, safe_div(err_cw, err_cwo, 1.0) - 1.0))
                crp_val = float(rp_val - rp_ctrl)
            else:
                crp_val = float(rp_val)

            # PDRP & DDRP
            gen_w = with_attr[with_attr["_true"] == 0]
            gen_wo = without_attr[without_attr["_true"] == 0]
            fpr_w = float(gen_w["_pred"].mean()) if len(gen_w) > 0 else 0.0
            fpr_wo = float(gen_wo["_pred"].mean()) if len(gen_wo) > 0 else 0.0
            pdrp_val = safe_div(fpr_w, fpr_wo, 1.0)

            fake_w = with_attr[with_attr["_true"] == 1]
            fake_wo = without_attr[without_attr["_true"] == 1]
            fnr_w = float((fake_w["_pred"] == 0).mean()) if len(fake_w) > 0 else 0.0
            fnr_wo = float((fake_wo["_pred"] == 0).mean()) if len(fake_wo) > 0 else 0.0
            ddrp_val = safe_div(fnr_w, fnr_wo, 1.0)

            record = CohortMetricRecord(
                attribute=attr,
                sample_count=cnt,
                accuracy=round(acc_w, 4),
                error_rate=round(err_w, 4),
                pdrp=round(pdrp_val, 4),
                ddrp=round(ddrp_val, 4),
                rp=round(rp_val, 4),
                crp=round(crp_val, 4)
            )
            cohort_records[attr] = record

            crp_vals.append(abs(crp_val))
            pdrp_vals.append(pdrp_val)
            ddrp_vals.append(ddrp_val)

            if abs(crp_val) > 0.15 or pdrp_val > 1.4 or ddrp_val > 1.4:
                high_bias.append(attr)

        res = AuditResult(
            global_accuracy=round(global_acc, 4),
            total_samples=len(df_eval),
            cohort_metrics=cohort_records,
            high_bias_cohorts=high_bias,
            max_abs_crp=round(max(crp_vals), 4) if crp_vals else 0.0,
            max_pdrp=round(max(pdrp_vals), 4) if pdrp_vals else 1.0,
            max_ddrp=round(max(ddrp_vals), 4) if ddrp_vals else 1.0,
            dataset_source=self.source
        )

        return FairnessReport(result=res)
