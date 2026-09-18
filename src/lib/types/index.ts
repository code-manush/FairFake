export type DetectionVerdict = "REAL" | "FAKE" | "LIKELY FAKE" | "UNCERTAIN";
export type BiasSeverity = "low" | "moderate" | "high";
export type MediaType = "image" | "video";

export type ImageResult = {
  fake_probability: number;
  verdict: DetectionVerdict;
  gradcam_base64: string;
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
};

export type AttributeBiasResult = {
  attribute: string;           // e.g. "Wearing_Glasses"
  model: string;               // e.g. "Xception" | "EfficientNetB0"
  errorWithAttribute: number;  // 0-1
  errorWithoutAttribute: number;
  rp: number;                  // relative performance, can be negative
  crp: number;                 // corrected relative performance
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
