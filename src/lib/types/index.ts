export type DetectionVerdict = "REAL" | "FAKE" | "LIKELY FAKE" | "UNCERTAIN";
export type BiasSeverity = "low" | "moderate" | "high";
export type MediaType = "image" | "video";
export type VerdictShiftType = "FALSE_ALARM_PREVENTED" | "EVASION_PREVENTED" | "UNCHANGED";

export type CalibratedDecision = {
  probability: number;
  effective_threshold: number;
  verdict: DetectionVerdict;
  confidence: number;
  effective_temperature: number;
};

export type CalibrationComparison = {
  is_verdict_flipped: boolean;
  shift_type: VerdictShiftType;
  headline: string;
  threshold_delta: number;
  probability_delta: number;
};

export type ContributingCohort = {
  attribute: string;
  cohort_threshold: number;
  delta: number;
};

export type BiometricCalibrationContext = {
  detected_attributes: string[];
  contributing_cohorts: ContributingCohort[];
};

export type CalibrationEvaluation = {
  raw: {
    probability: number;
    threshold: number;
    verdict: DetectionVerdict;
  };
  calibrated: CalibratedDecision;
  comparison: CalibrationComparison;
  biometric_context: BiometricCalibrationContext;
  explanation: string;
};

export type ImageResult = {
  fake_probability: number;
  verdict: DetectionVerdict;
  gradcam_base64: string;
  calibrated?: CalibratedDecision;
  comparison?: CalibrationComparison;
  explanation?: string;
};

export type VideoFrameResult = {
  second: number;
  frame_index: number;
  fake_probability: number;
  gradcam_base64: string;
};

export type FaceResult = {
  boundingBox: [number, number, number, number];
  attributes: Record<string, string | boolean>;
};

export type DetectionResult = {
  id: string; // client-side generated for reference
  mediaType: MediaType;
  
  // Image properties
  total_images_analyzed?: number;
  image_authenticity_score?: number;
  results?: ImageResult[];
  
  // Video properties
  duration_seconds?: number;
  total_frames_analyzed?: number;
  anomaly_seconds?: number[];
  max_fake_probability?: number;
  video_authenticity_score?: number;
  verdict?: DetectionVerdict;
  frame_results?: VideoFrameResult[];
  
  faces: FaceResult[];
  calibration?: CalibrationEvaluation;
  spectral?: SpectralMetrics;
  spatio_temporal?: SpatioTemporalMetrics;
  biological_signals?: BiologicalSignals;
  per_second_timeline?: SecondTimelineEntry[];
};

export type AttributeBiasResult = {
  attribute: string;           // e.g. "Wearing_Glasses"
  model: string;               // e.g. "Xception" | "EfficientNetB0"
  errorWithAttribute: number;  // 0-1
  errorWithoutAttribute: number;
  rp: number;                  // relative performance, can be negative
  crp: number;                 // corrected relative performance
  pdrp: number;                // pristine data relative performance
  ddrp: number;                // deepfake data relative performance
  severity: BiasSeverity;
  sampleCount: { withAttribute: number; withoutAttribute: number };
  exampleMisclassifiedIds: string[]; // point at mock face thumbnails
};

export type AuditSummary = {
  model: string;
  totalImages: number;
  overallAccuracy: number;
  overallFPR: number;
  overallFNR: number;
  highBiasAttributeCount: number;
};

export type ThresholdMatrixResponse = {
  metadata: {
    version: string;
    criterion: string;
    alpha: number;
    global_default_threshold: number;
    target_baseline: {
      tpr: number;
      fpr: number;
    };
    generated_at: string;
    total_attributes_calibrated: number;
  };
  thresholds: Record<string, number>;
  calibrations: Record<string, {
    attribute: string;
    sample_count_with: number;
    uncalibrated_threshold: number;
    calibrated_threshold: number;
    uncalibrated_metrics: {
      tpr: number;
      fpr: number;
      fnr: number;
      accuracy: number;
      pdrp: number;
      ddrp: number;
    };
    calibrated_metrics: {
      tpr: number;
      fpr: number;
      fnr: number;
      accuracy: number;
      pdrp: number;
      ddrp: number;
    };
    improvements: {
      delta_pdrp: number;
      delta_ddrp: number;
      delta_accuracy: number;
    };
    loss: number;
  }>;
};

export type TemperatureWeightsResponse = {
  metadata: {
    version: string;
    global_temperature: number;
    global_evaluation: {
      uncalibrated: { nll: number; brier: number; ece: number };
      calibrated: { nll: number; brier: number; ece: number };
      ece_reduction: number;
    };
    generated_at: string;
    total_cohorts_fitted: number;
  };
  temperatures: Record<string, number>;
  cohort_evaluations: Record<string, any>;
};

export type CalibrationEvaluateRequest = {
  raw_probability: number;
  attributes?: string[];
  attributes_dict?: Record<string, any>;
  strategy?: "weighted_residual" | "max_margin" | "mean";
};

// ---------------------------------------------------------------------------
// Phase 4: Spatio-Temporal & Multimodal Video Forensics Types
// ---------------------------------------------------------------------------

export type SpectralMetrics = {
  checkerboard_score: number;
  high_freq_energy_ratio: number;
  spectral_verdict: "SYNTHETIC_UPSAMPLING" | "NATURAL_SPECTRUM";
  radial_profile: number[];
  radial_frequencies: number[];
  spectral_heatmap_b64: string;
};

export type SpatioTemporalMetrics = {
  flicker_score: number;
  motion_continuity: string;
  mean_optical_flow_mag: number;
};

export type BiologicalSignals = {
  blinks: {
    blink_count: number;
    blinks_per_minute: number;
    blink_status: string;
    biological_risk: "LOW" | "MEDIUM" | "HIGH";
  };
  rppg_pulse: {
    bvp_correlation: number;
    estimated_heart_rate_bpm: number;
    rppg_verdict: string;
    pulse_synchrony_score: number;
  };
};

export type SecondTimelineEntry = {
  second: number;
  max_fake_probability: number;
  mean_fake_probability: number;
  risk_level: "LOW" | "ELEVATED" | "CRITICAL";
  detected_anomalies: string[];
};

export type LipSyncAuditResponse = {
  sync_correlation: number;
  zero_lag_correlation: number;
  temporal_offset_ms: number;
  multimodal_verdict: string;
  sync_confidence: number;
};
