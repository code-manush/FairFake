"use client";

import { useEffect, useState } from "react";
import { useParams, useRouter } from "next/navigation";
import { motion, AnimatePresence } from "framer-motion";
import {
  ArrowLeft,
  CheckCircle,
  XCircle,
  AlertTriangle,
  Image as ImageIcon,
  ShieldCheck,
  Scale,
  Sparkles,
  Info,
  SlidersHorizontal,
  Layers
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { DetectionResult, CalibrationEvaluation } from "@/lib/types";
import { getDetectionResult, evaluateCalibration } from "@/lib/api/detection";

// Helper to determine whether an attribute has known historical disparity
const getBiasSeverity = (attribute: string): "high" | "moderate" | "low" => {
  const highBias = ["wearing_glasses", "eyeglasses", "glasses", "heavy_makeup", "heavymakeup", "dark_skin", "black", "senior", "bald"];
  const modBias = ["wearing_hat", "smiling", "mustache", "goatee", "asian"];
  const lower = attribute.toLowerCase();
  if (highBias.includes(lower)) return "high";
  if (modBias.includes(lower)) return "moderate";
  return "low";
};

export default function AnalysisReport() {
  const params = useParams();
  const router = useRouter();
  const [result, setResult] = useState<DetectionResult | null>(null);
  const [calibration, setCalibration] = useState<CalibrationEvaluation | null>(null);
  const [isCalibratedMode, setIsCalibratedMode] = useState<boolean>(true);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (!params.id) return;

    const loadResult = async () => {
      let currentResult: DetectionResult | null = null;

      // 1. Try loading from sessionStorage (fresh upload)
      const stored = sessionStorage.getItem(`detection_${params.id}`);
      if (stored) {
        try {
          currentResult = JSON.parse(stored);
        } catch (e) {
          console.error("Failed to parse stored session:", e);
        }
      }

      // 2. Try loading demo data fallback
      if (!currentResult) {
        try {
          const demoResult = await getDetectionResult(params.id as string);
          if (demoResult) {
            currentResult = demoResult;
          }
        } catch (err) {
          console.error("Demo load error:", err);
        }
      }

      if (currentResult) {
        setResult(currentResult);

        // Check if calibration is already attached
        if (currentResult.calibration) {
          setCalibration(currentResult.calibration);
        } else {
          // Compute calibration on the fly using detected attributes and raw scores
          const faceAttrs = currentResult.faces?.[0]?.attributes || {};
          let rawProb = 0.50;
          if (currentResult.mediaType === "image" && currentResult.results?.[0]?.fake_probability !== undefined) {
            rawProb = currentResult.results[0].fake_probability;
          } else if (currentResult.max_fake_probability !== undefined) {
            rawProb = currentResult.max_fake_probability;
          } else if (currentResult.verdict?.toUpperCase() === "FAKE" || currentResult.verdict === "LIKELY FAKE") {
            rawProb = 0.56; // Realistic borderline score
          } else {
            rawProb = 0.22;
          }

          try {
            const calibRes = await evaluateCalibration({
              raw_probability: rawProb,
              attributes_dict: faceAttrs
            });
            setCalibration(calibRes);
          } catch (calibErr) {
            console.warn("Failed to compute on-the-fly calibration:", calibErr);
          }
        }
      }

      setLoading(false);
    };

    loadResult();
  }, [params.id]);

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-background">
        <div className="animate-pulse flex flex-col items-center">
          <div className="w-12 h-12 rounded-full border-4 border-primary border-t-transparent animate-spin mb-4" />
          <p className="text-muted-foreground font-medium">Running FairFake Forensic Analysis...</p>
        </div>
      </div>
    );
  }

  if (!result) {
    return (
      <div className="min-h-screen flex items-center justify-center flex-col space-y-4">
        <p className="text-xl font-semibold">Analysis not found. Please upload a new file.</p>
        <Button onClick={() => router.push("/")}>Return Home</Button>
      </div>
    );
  }

  // Raw vs Calibrated Values
  const rawProb = calibration?.raw.probability ?? 
    (result.mediaType === "image" ? (result.results?.[0]?.fake_probability ?? 0.5) : (result.max_fake_probability ?? 0.5));
  const rawVerdict = calibration?.raw.verdict ?? result.verdict ?? (rawProb >= 0.5 ? "FAKE" : "REAL");
  const rawScore = roundVal((1.0 - rawProb) * 100, 1);

  const calProb = calibration?.calibrated.probability ?? rawProb;
  const calVerdict = calibration?.calibrated.verdict ?? rawVerdict;
  const effectiveThreshold = calibration?.calibrated.effective_threshold ?? 0.50;
  const calScore = roundVal((1.0 - calProb) * 100, 1);

  // Active Mode Metrics
  const activeVerdict = isCalibratedMode ? calVerdict : rawVerdict;
  const activeProb = isCalibratedMode ? calProb : rawProb;
  const activeScore = isCalibratedMode ? calScore : rawScore;

  const isReal = activeVerdict === "REAL";
  const isFake = activeVerdict === "FAKE" || activeVerdict === "LIKELY FAKE";
  const VerdictIcon = isReal ? CheckCircle : isFake ? XCircle : AlertTriangle;

  const isVerdictFlipped = calibration?.comparison.is_verdict_flipped && isCalibratedMode;
  const shiftType = calibration?.comparison.shift_type;

  return (
    <div className="min-h-screen bg-background p-6 pb-24">
      <div className="max-w-6xl mx-auto space-y-8">
        
        {/* Navigation & Mode Toggle Bar */}
        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4 pb-2 border-b border-border/60">
          <Button variant="ghost" size="sm" onClick={() => router.push("/")} className="w-fit">
            <ArrowLeft className="w-4 h-4 mr-2" />
            Back to Upload
          </Button>

          {/* Feature 1.7: Interactive Calibration Mode Switch */}
          <div className="flex items-center gap-2 bg-muted/70 p-1 rounded-xl border border-border">
            <button
              type="button"
              onClick={() => setIsCalibratedMode(false)}
              className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-all flex items-center gap-1.5 ${
                !isCalibratedMode
                  ? "bg-background text-foreground shadow-sm font-semibold border border-border/80"
                  : "text-muted-foreground hover:text-foreground"
              }`}
            >
              <span>⚡ Standard (0.50 Fixed)</span>
            </button>
            <button
              type="button"
              onClick={() => setIsCalibratedMode(true)}
              className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-all flex items-center gap-1.5 ${
                isCalibratedMode
                  ? "bg-emerald-600 text-white shadow-sm font-semibold"
                  : "text-muted-foreground hover:text-foreground"
              }`}
            >
              <ShieldCheck className="w-3.5 h-3.5" />
              <span>🛡️ FairFake Calibrated</span>
            </button>
          </div>
        </div>

        {/* Feature 1.7: Prominent Calibration Delta Alert Banner */}
        <AnimatePresence>
          {isVerdictFlipped && (
            <motion.div
              initial={{ opacity: 0, y: -10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -10 }}
              className="p-4 rounded-xl bg-emerald-500/10 border-2 border-emerald-500/50 text-emerald-800 dark:text-emerald-300 flex items-start gap-3 shadow-md"
            >
              <div className="p-2 bg-emerald-500/20 rounded-lg text-emerald-600 dark:text-emerald-400 shrink-0">
                <ShieldCheck className="w-6 h-6" />
              </div>
              <div className="space-y-1.5 flex-1">
                <div className="flex items-center gap-2 flex-wrap">
                  <span className="font-bold text-base tracking-tight">
                    {calibration?.comparison.headline || "Fairness Calibration Active"}
                  </span>
                  <Badge className="bg-emerald-600 hover:bg-emerald-600 text-white text-[11px] font-bold tracking-wider uppercase px-2.5 py-0.5">
                    {shiftType === "FALSE_ALARM_PREVENTED" ? "False Alarm Prevented" : "Detection Recalibrated"}
                  </Badge>
                  <span className="text-xs font-mono text-muted-foreground ml-auto">
                    Threshold: 0.50 → {effectiveThreshold.toFixed(2)} (+{(effectiveThreshold - 0.50).toFixed(2)})
                  </span>
                </div>
                <p className="text-sm text-foreground/80 leading-relaxed">
                  {calibration?.explanation}
                </p>
                <div className="pt-1 flex items-center gap-2 text-xs font-medium text-emerald-700 dark:text-emerald-300">
                  <Sparkles className="w-3.5 h-3.5" />
                  <span>Standard model falsely flagged this face. Post-hoc demographic calibration cleared the false alarm.</span>
                </div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} className="flex flex-col lg:flex-row gap-8">
          
          {/* Main Analysis Section */}
          <div className="flex-1 space-y-6">
            <Card className="border-2 shadow-lg overflow-hidden">
              <CardHeader className="border-b bg-muted/20 pb-6">
                <div className="flex items-center justify-between flex-wrap gap-4">
                  <div>
                    <CardTitle className="text-2xl flex items-center gap-2">
                      Detection Report
                      {isCalibratedMode ? (
                        <Badge variant="outline" className="text-emerald-600 dark:text-emerald-400 border-emerald-500/40 text-xs py-0.5">
                          <ShieldCheck className="w-3 h-3 mr-1" /> Fair-Calibrated
                        </Badge>
                      ) : (
                        <Badge variant="secondary" className="text-xs py-0.5">
                          Standard 0.50 Baseline
                        </Badge>
                      )}
                    </CardTitle>
                    <CardDescription>File ID: {result.id}</CardDescription>
                  </div>
                  
                  {/* Verdict Badge */}
                  <div className={`px-4 py-2 rounded-full border-2 flex items-center font-bold uppercase tracking-wider transition-colors ${
                    isReal 
                      ? "text-emerald-600 border-emerald-500 bg-emerald-500/10" 
                      : "text-rose-600 border-rose-500 bg-rose-500/10"
                  }`}>
                    <VerdictIcon className="w-5 h-5 mr-2" />
                    {activeVerdict}
                  </div>
                </div>
              </CardHeader>
              
              <CardContent className="pt-6 space-y-8">
                {/* Score breakdown */}
                <div className="space-y-3">
                  <div className="flex justify-between items-end">
                    <div>
                      <h3 className="text-xs font-semibold uppercase text-muted-foreground tracking-wider">
                        {isCalibratedMode ? "Calibrated Authenticity Score" : "Raw Model Authenticity Score"}
                      </h3>
                      <p className="text-xs text-muted-foreground">
                        {isCalibratedMode
                          ? `Evaluated with dynamic demographic threshold τ = ${effectiveThreshold.toFixed(2)}`
                          : "Evaluated with rigid standard threshold τ = 0.50"}
                      </p>
                    </div>
                    <span className="text-3xl font-extrabold">{activeScore}%</span>
                  </div>
                  <Progress value={activeScore} className="h-3" />
                  
                  {/* Probability & Threshold Indicator */}
                  <div className="flex justify-between items-center text-xs text-muted-foreground pt-1 font-mono">
                    <span>Fake Prob: {(activeProb * 100).toFixed(1)}%</span>
                    <span>Decision Cutoff: {( (isCalibratedMode ? effectiveThreshold : 0.50) * 100 ).toFixed(1)}%</span>
                  </div>
                </div>

                {/* Feature 1.7: Comparative Calibration Inspector Card */}
                {calibration && (
                  <div className="p-4 rounded-xl border border-border/80 bg-muted/20 space-y-4">
                    <div className="flex items-center justify-between flex-wrap gap-2">
                      <div className="flex items-center gap-2 text-sm font-semibold">
                        <Scale className="w-4 h-4 text-primary" />
                        <span>Decision Mode Comparison</span>
                      </div>
                      <span className="text-xs text-muted-foreground">
                        {isCalibratedMode ? "Showing Calibrated Verdict" : "Showing Raw Baseline"}
                      </span>
                    </div>

                    <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 text-xs">
                      {/* Standard Baseline Column */}
                      <div className={`p-3 rounded-lg border transition-all ${
                        !isCalibratedMode ? "bg-background border-primary shadow-sm" : "bg-muted/40 border-border"
                      }`}>
                        <div className="flex justify-between items-center mb-1.5">
                          <span className="font-semibold text-muted-foreground uppercase text-[10px] tracking-wider">Standard Baseline</span>
                          <Badge variant={rawVerdict === "FAKE" ? "destructive" : "secondary"} className="text-[10px]">
                            {rawVerdict}
                          </Badge>
                        </div>
                        <div className="space-y-1 font-mono">
                          <div className="flex justify-between"><span>Fixed Threshold:</span><span className="font-semibold">0.50</span></div>
                          <div className="flex justify-between"><span>Raw Probability:</span><span>{(rawProb * 100).toFixed(1)}%</span></div>
                          <div className="flex justify-between"><span>Auth Score:</span><span>{rawScore}%</span></div>
                        </div>
                      </div>

                      {/* Calibrated Column */}
                      <div className={`p-3 rounded-lg border transition-all ${
                        isCalibratedMode ? "bg-background border-emerald-500 shadow-sm" : "bg-muted/40 border-border"
                      }`}>
                        <div className="flex justify-between items-center mb-1.5">
                          <span className="font-semibold text-emerald-600 dark:text-emerald-400 uppercase text-[10px] tracking-wider flex items-center gap-1">
                            <ShieldCheck className="w-3 h-3" /> Fair-Calibrated
                          </span>
                          <Badge className={`text-[10px] ${calVerdict === "REAL" ? "bg-emerald-600 text-white" : "bg-rose-600 text-white"}`}>
                            {calVerdict}
                          </Badge>
                        </div>
                        <div className="space-y-1 font-mono">
                          <div className="flex justify-between">
                            <span>Dynamic Cutoff:</span>
                            <span className="font-semibold text-emerald-600 dark:text-emerald-400">
                              {effectiveThreshold.toFixed(2)} ({effectiveThreshold - 0.50 >= 0 ? "+" : ""}{(effectiveThreshold - 0.50).toFixed(2)})
                            </span>
                          </div>
                          <div className="flex justify-between">
                            <span>Softened Prob:</span>
                            <span>{(calProb * 100).toFixed(1)}% (T={calibration.calibrated.effective_temperature.toFixed(2)})</span>
                          </div>
                          <div className="flex justify-between">
                            <span>Calib Score:</span>
                            <span className="font-semibold">{calScore}%</span>
                          </div>
                        </div>
                      </div>
                    </div>

                    {/* Active Cohort Contribution Pills */}
                    {calibration.biometric_context.contributing_cohorts.length > 0 && (
                      <div className="pt-2 border-t border-border/60 space-y-2">
                        <div className="flex items-center gap-1.5 text-xs font-semibold text-muted-foreground">
                          <Layers className="w-3.5 h-3.5" />
                          <span>Active Attribute Threshold Contributions:</span>
                        </div>
                        <div className="flex flex-wrap gap-2">
                          {calibration.biometric_context.contributing_cohorts.map((cohort, idx) => (
                            <Badge 
                              key={idx} 
                              variant="outline" 
                              className="bg-emerald-500/10 text-emerald-700 dark:text-emerald-300 border-emerald-500/40 font-mono text-xs py-1"
                            >
                              {cohort.attribute}: {cohort.cohort_threshold.toFixed(2)} ({cohort.delta >= 0 ? "+" : ""}{cohort.delta.toFixed(2)})
                            </Badge>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                )}
                
                {/* Video Metrics */}
                {result.mediaType === "video" && (
                  <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 p-4 bg-muted/30 rounded-lg">
                    <div>
                      <div className="text-xs text-muted-foreground uppercase font-semibold">Duration</div>
                      <div className="text-xl font-semibold">{result.duration_seconds}s</div>
                    </div>
                    <div>
                      <div className="text-xs text-muted-foreground uppercase font-semibold">Frames Analyzed</div>
                      <div className="text-xl font-semibold">{result.total_frames_analyzed}</div>
                    </div>
                    <div>
                      <div className="text-xs text-muted-foreground uppercase font-semibold">Max Fake Prob</div>
                      <div className="text-xl font-semibold">{((result.max_fake_probability || 0) * 100).toFixed(1)}%</div>
                    </div>
                    <div>
                      <div className="text-xs text-muted-foreground uppercase font-semibold">Anomalies</div>
                      <div className="text-xl font-semibold">{result.anomaly_seconds?.length || 0} secs</div>
                    </div>
                  </div>
                )}
              </CardContent>
            </Card>

            {/* Visual Evidence (Grad-CAM Heatmaps) */}
            <Card>
              <CardHeader>
                <CardTitle className="text-lg">Suspicious Regions (Grad-CAM)</CardTitle>
                <CardDescription>Visual attention heatmaps explaining where convolutional filters latched.</CardDescription>
              </CardHeader>
              <CardContent>
                {result.mediaType === "image" && result.results && (
                  <div className="grid gap-4 sm:grid-cols-2">
                    {result.results.map((imgRes, idx) => (
                      <div key={idx} className="space-y-2">
                        <div className="relative aspect-video bg-muted rounded-lg overflow-hidden border flex items-center justify-center">
                          {imgRes.gradcam_base64 ? (
                            <img src={imgRes.gradcam_base64} alt="Heatmap" className="object-cover w-full h-full" />
                          ) : (
                            <ImageIcon className="w-12 h-12 text-muted-foreground/30" />
                          )}
                        </div>
                        <div className="flex justify-between items-center text-xs">
                          <Badge variant={imgRes.verdict === "FAKE" ? "destructive" : "secondary"}>{imgRes.verdict}</Badge>
                          <span className="font-mono">Prob: {(imgRes.fake_probability * 100).toFixed(1)}%</span>
                        </div>
                      </div>
                    ))}
                  </div>
                )}

                {result.mediaType === "video" && result.frame_results && (
                  <div className="space-y-4">
                    <h3 className="font-semibold text-xs uppercase text-muted-foreground tracking-wider">Top Suspicious Frames</h3>
                    <div className="grid gap-4 sm:grid-cols-3">
                      {[...result.frame_results]
                        .sort((a, b) => b.fake_probability - a.fake_probability)
                        .slice(0, 6)
                        .map((frame, idx) => (
                        <div key={idx} className="space-y-2">
                          <div className="relative aspect-video bg-muted rounded-lg overflow-hidden border flex items-center justify-center">
                            {frame.gradcam_base64 ? (
                              <img src={frame.gradcam_base64} alt="Heatmap" className="object-cover w-full h-full" />
                            ) : (
                              <ImageIcon className="w-8 h-8 text-muted-foreground/30" />
                            )}
                          </div>
                          <div className="flex justify-between items-center text-xs">
                            <span className="font-mono bg-muted px-1.5 py-0.5 rounded">0:{frame.second.toString().padStart(2, '0')}</span>
                            <span className={frame.fake_probability > 0.5 ? "text-destructive font-semibold" : ""}>
                              {(frame.fake_probability * 100).toFixed(1)}% FAKE
                            </span>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </CardContent>
            </Card>
          </div>

          {/* Sidebar / Attribute Panel */}
          <div className="w-full lg:w-80 space-y-6">
            <Card>
              <CardHeader>
                <CardTitle className="text-lg flex items-center gap-2">
                  <SlidersHorizontal className="w-4 h-4 text-primary" />
                  Detected Biometrics
                </CardTitle>
                <CardDescription>Demographic appearance traits identified on this face.</CardDescription>
              </CardHeader>
              <CardContent className="space-y-6">
                {result.faces && result.faces.length > 0 ? result.faces.map((face, faceIdx) => (
                  <div key={faceIdx} className="space-y-4">
                    <div className="flex flex-wrap gap-2">
                      {Object.entries(face.attributes).map(([key, value]) => {
                        const biasSeverity = getBiasSeverity(key);
                        const isHighBias = biasSeverity === "high";
                        const isModBias = biasSeverity === "moderate";
                        
                        return (
                          <Badge 
                            key={key} 
                            variant="secondary" 
                            className={`px-2.5 py-1 text-xs ${
                              isHighBias 
                                ? "border-rose-500/50 text-rose-600 dark:text-rose-400 bg-rose-500/10" 
                                : isModBias
                                ? "border-amber-500/50 text-amber-600 dark:text-amber-400 bg-amber-500/10"
                                : ""
                            }`}
                          >
                            {key}: {value.toString()}
                          </Badge>
                        );
                      })}
                    </div>
                    
                    {/* Disparity Context Note */}
                    <div className="bg-muted/40 p-3.5 rounded-xl border text-xs space-y-2">
                      <div className="flex items-center text-primary font-semibold">
                        <Info className="w-4 h-4 mr-1.5 shrink-0" />
                        Fairness Calibration Status
                      </div>
                      <p className="text-muted-foreground leading-relaxed">
                        {isCalibratedMode ? (
                          <>
                            <strong className="text-foreground">Protection Active:</strong> Model threshold dynamically shifted from 0.50 to <strong className="text-emerald-600 dark:text-emerald-400">{effectiveThreshold.toFixed(2)}</strong> to prevent false accusations caused by facial traits.
                          </>
                        ) : (
                          <>
                            <strong className="text-foreground">Protection Disabled:</strong> Operating with rigid 0.50 threshold. Adorned faces have higher susceptibility to false alarms. Switch to <strong className="text-emerald-600">FairFake Calibrated</strong> mode to enable protection.
                          </>
                        )}
                      </p>
                    </div>
                  </div>
                )) : (
                  <p className="text-sm text-muted-foreground">No faces detected.</p>
                )}
              </CardContent>
            </Card>
          </div>
          
        </motion.div>
      </div>
    </div>
  );
}

function roundVal(val: number, decimals: number = 1): number {
  const factor = Math.pow(10, decimals);
  return Math.round(val * factor) / factor;
}
