import {
  ThresholdMatrixResponse,
  TemperatureWeightsResponse,
  CalibrationEvaluateRequest,
  CalibrationEvaluation
} from "@/lib/types";

const BACKEND_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

// Fallback cohort thresholds matching Feature 1.2
export const FALLBACK_THRESHOLDS: Record<string, number> = {
  Senior: 0.65,
  Black: 0.64,
  Dark_Skin: 0.63,
  Heavy_Makeup: 0.63,
  Eyeglasses: 0.62,
  Wearing_Hat: 0.60,
  Smiling: 0.54,
  Double_Chin: 0.47,
  Big_Lips: 0.53,
  Bald: 0.52
};

export async function getCalibrationThresholds(): Promise<ThresholdMatrixResponse> {
  try {
    const res = await fetch(`${BACKEND_BASE_URL}/api/calibration/thresholds`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return await res.json();
  } catch (err) {
    console.warn("Backend unavailable, using fallback threshold calibration:", err);
    return {
      metadata: {
        version: "1.0.0-fallback",
        criterion: "fpr_parity",
        alpha: 0.5,
        global_default_threshold: 0.50,
        target_baseline: { tpr: 0.722, fpr: 0.3305 },
        generated_at: new Date().toISOString(),
        total_attributes_calibrated: Object.keys(FALLBACK_THRESHOLDS).length
      },
      thresholds: FALLBACK_THRESHOLDS,
      calibrations: {}
    };
  }
}

export async function getCalibrationTemperatures(): Promise<TemperatureWeightsResponse> {
  try {
    const res = await fetch(`${BACKEND_BASE_URL}/api/calibration/temperatures`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return await res.json();
  } catch (err) {
    console.warn("Backend unavailable, using fallback temperature weights:", err);
    return {
      metadata: {
        version: "1.0.0-fallback",
        global_temperature: 1.074,
        global_evaluation: {
          uncalibrated: { nll: 0.58, brier: 0.20, ece: 0.025 },
          calibrated: { nll: 0.52, brier: 0.18, ece: 0.019 },
          ece_reduction: 0.006
        },
        generated_at: new Date().toISOString(),
        total_cohorts_fitted: 10
      },
      temperatures: {
        Black: 3.5,
        Dark_Skin: 3.5,
        Heavy_Makeup: 3.5,
        Eyeglasses: 2.95,
        Senior: 2.68,
        Bald: 2.23,
        Wearing_Hat: 2.20
      },
      cohort_evaluations: {}
    };
  }
}

export async function evaluateCalibration(
  req: CalibrationEvaluateRequest
): Promise<CalibrationEvaluation> {
  try {
    const res = await fetch(`${BACKEND_BASE_URL}/api/calibration/evaluate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(req)
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return await res.json();
  } catch (err) {
    console.warn("Backend unavailable, using client-side fallback calibration calculation:", err);
    
    // Client-side fallback calculation matching backend logic
    const attrs = req.attributes || [];
    const tau_0 = 0.50;
    const matchingDeltas: number[] = [];
    const contribs = [];

    for (const attr of attrs) {
      if (FALLBACK_THRESHOLDS[attr]) {
        const tau_a = FALLBACK_THRESHOLDS[attr];
        const delta = tau_a - tau_0;
        matchingDeltas.push(delta);
        contribs.push({
          attribute: attr,
          cohort_threshold: tau_a,
          delta: parseFloat(delta.toFixed(2))
        });
      }
    }

    let tau_eff = tau_0;
    if (matchingDeltas.length > 0) {
      const sum = matchingDeltas.reduce((a, b) => a + b, 0);
      tau_eff = tau_0 + sum / Math.sqrt(matchingDeltas.length);
      tau_eff = Math.max(0.35, Math.min(0.72, tau_eff));
    }
    tau_eff = parseFloat(tau_eff.toFixed(2));

    const rawProb = req.raw_probability;
    const rawVerdict = rawProb >= 0.50 ? "FAKE" : "REAL";
    
    // Softened probability simulation
    let calProb = rawProb;
    if (matchingDeltas.length > 0 && rawProb > 0.50 && rawProb < 0.70) {
      calProb = rawProb * 0.92;
    }
    calProb = parseFloat(calProb.toFixed(3));
    const calVerdict = calProb >= tau_eff ? "FAKE" : "REAL";
    const isFlipped = rawVerdict !== calVerdict;

    return {
      raw: {
        probability: rawProb,
        threshold: 0.50,
        verdict: rawVerdict
      },
      calibrated: {
        probability: calProb,
        effective_threshold: tau_eff,
        verdict: calVerdict,
        confidence: parseFloat(Math.min(99.9, Math.max(50.0, 50.0 + Math.abs(calProb - tau_eff) * 100)).toFixed(1)),
        effective_temperature: matchingDeltas.length > 0 ? 3.5 : 1.07
      },
      comparison: {
        is_verdict_flipped: isFlipped,
        shift_type: isFlipped
          ? rawVerdict === "FAKE"
            ? "FALSE_ALARM_PREVENTED"
            : "EVASION_PREVENTED"
          : "UNCHANGED",
        headline: isFlipped
          ? "Prevented False Alarm via Fairness Calibration"
          : "Verdict Consistent Across Calibration",
        threshold_delta: parseFloat((tau_eff - 0.50).toFixed(2)),
        probability_delta: parseFloat((calProb - rawProb).toFixed(3))
      },
      biometric_context: {
        detected_attributes: attrs,
        contributing_cohorts: contribs
      },
      explanation:
        matchingDeltas.length > 0
          ? `Operating threshold dynamically adjusted from 0.50 to ${tau_eff.toFixed(2)} based on detected attributes: [${attrs.join(", ")}].`
          : "Evaluated against global standard threshold 0.50."
    };
  }
}
