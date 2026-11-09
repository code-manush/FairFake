"use client";

import { useState, useMemo } from "react";
import { motion } from "framer-motion";
import { 
  Sliders, 
  ShieldCheck, 
  Scale, 
  AlertTriangle, 
  CheckCircle2, 
  TrendingDown, 
  TrendingUp, 
  Info,
  Sparkles,
  Zap
} from "lucide-react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { FALLBACK_THRESHOLDS } from "@/lib/api/calibration";

interface CohortProfile {
  name: string;
  label: string;
  description: string;
  nominalRealLogit: number;
  nominalFakeLogit: number;
  calibratedTau: number;
  sampleTotal: number;
}

const COHORT_PROFILES: Record<string, CohortProfile> = {
  Dark_Skin: {
    name: "Dark_Skin",
    label: "Dark Skin Complexion",
    description: "Underrepresented skin pigmentation exhibiting elevated cosmetic and shading false alarms.",
    nominalRealLogit: -0.55,
    nominalFakeLogit: 0.57,
    calibratedTau: 0.63,
    sampleTotal: 720
  },
  Heavy_Makeup: {
    name: "Heavy_Makeup",
    label: "Heavy Cosmetics / Makeup",
    description: "Cosmetic blending contours mirror deepfake facial donor inpainting seams.",
    nominalRealLogit: -0.60,
    nominalFakeLogit: 0.67,
    calibratedTau: 0.63,
    sampleTotal: 1080
  },
  Eyeglasses: {
    name: "Eyeglasses",
    label: "Eyeglasses / Spectacles",
    description: "Frame edges and specular reflections induce spurious artifact alarms.",
    nominalRealLogit: -0.70,
    nominalFakeLogit: 0.70,
    calibratedTau: 0.62,
    sampleTotal: 680
  },
  Senior: {
    name: "Senior",
    label: "Senior Age Cohort (55+)",
    description: "Natural facial wrinkles and skin creases trigger premature forgery detections.",
    nominalRealLogit: -0.75,
    nominalFakeLogit: 0.75,
    calibratedTau: 0.65,
    sampleTotal: 560
  },
  Black: {
    name: "Black",
    label: "Black Demographic",
    description: "Subject to acute historical training data imbalance in FF++ and DFD benchmarks.",
    nominalRealLogit: -0.55,
    nominalFakeLogit: 0.57,
    calibratedTau: 0.64,
    sampleTotal: 480
  },
  Wearing_Hat: {
    name: "Wearing_Hat",
    label: "Headwear / Hats",
    description: "Forehead obstruction and shadows alter standard facial framing proportions.",
    nominalRealLogit: -0.85,
    nominalFakeLogit: 0.85,
    calibratedTau: 0.60,
    sampleTotal: 480
  },
  Bald: {
    name: "Bald",
    label: "Bald / Receding Hair",
    description: "High DDRP evasion vulnerability: manipulated head swaps blend into smooth scalp.",
    nominalRealLogit: -1.25,
    nominalFakeLogit: 0.80,
    calibratedTau: 0.52,
    sampleTotal: 360
  },
  Smiling: {
    name: "Smiling",
    label: "Smiling Expression",
    description: "Smile creases along cheeks and mouth corners introduce gradient anomalies.",
    nominalRealLogit: -1.10,
    nominalFakeLogit: 1.20,
    calibratedTau: 0.54,
    sampleTotal: 2080
  },
  Global_Baseline: {
    name: "Global_Baseline",
    label: "Global Baseline (Full Population)",
    description: "Overall cross-population reference distribution without attribute conditioning.",
    nominalRealLogit: -1.45,
    nominalFakeLogit: 1.45,
    calibratedTau: 0.50,
    sampleTotal: 4000
  }
};

// Gaussian cumulative distribution approximation
function normCdf(x: number): number {
  const t = 1.0 / (1.0 + 0.2316419 * Math.abs(x));
  const d = 0.3989423 * Math.exp(-x * x / 2);
  let p = d * t * (0.3193815 + t * (-0.3565638 + t * (1.781478 + t * (-1.821256 + t * 1.330274))));
  if (x > 0) p = 1.0 - p;
  return p;
}

function calculateCohortMetrics(profile: CohortProfile, tau: number) {
  const eps = 1e-4;
  const clampedTau = Math.max(eps, Math.min(1 - eps, tau));
  const tauLogit = Math.log(clampedTau / (1 - clampedTau));
  const sigma = 1.0;

  // Real faces: predicted fake if logit >= tauLogit
  const zReal = (tauLogit - profile.nominalRealLogit) / sigma;
  const fpr = 1.0 - normCdf(zReal);
  const tnr = 1.0 - fpr;

  // Fake faces: predicted fake if logit >= tauLogit
  const zFake = (tauLogit - profile.nominalFakeLogit) / sigma;
  const tpr = 1.0 - normCdf(zFake);
  const fnr = 1.0 - tpr;

  const realCount = Math.round(profile.sampleTotal / 2);
  const fakeCount = Math.round(profile.sampleTotal / 2);

  const tp = Math.round(tpr * fakeCount);
  const fp = Math.round(fpr * realCount);
  const tn = Math.round(tnr * realCount);
  const fn = Math.round(fnr * fakeCount);

  const accuracy = (tp + tn) / profile.sampleTotal;

  // Global reference for disparity calculations
  const globalRef = COHORT_PROFILES.Global_Baseline;
  const zGlobalReal = (tauLogit - globalRef.nominalRealLogit) / sigma;
  const globalFpr = 1.0 - normCdf(zGlobalReal);
  const zGlobalFake = (tauLogit - globalRef.nominalFakeLogit) / sigma;
  const globalFnr = 1.0 - (1.0 - normCdf(zGlobalFake));

  const pdrp = globalFpr > 0 ? (fpr / globalFpr) - 1.0 : 0.0;
  const ddrp = globalFnr > 0 ? (fnr / globalFnr) - 1.0 : 0.0;
  const rp = ((1.0 - accuracy) / 0.15) - 1.0;

  return {
    tau,
    fpr,
    tpr,
    fnr,
    tnr,
    accuracy,
    tp,
    fp,
    tn,
    fn,
    pdrp,
    ddrp,
    rp,
    globalFpr,
    globalFnr
  };
}

export function ThresholdSimulator() {
  const [selectedKey, setSelectedKey] = useState<string>("Dark_Skin");
  const [threshold, setThreshold] = useState<number>(0.50);

  const profile = COHORT_PROFILES[selectedKey] || COHORT_PROFILES.Dark_Skin;

  // Compute metrics at active threshold
  const metrics = useMemo(() => {
    return calculateCohortMetrics(profile, threshold);
  }, [profile, threshold]);

  // Compute metrics at nominal 0.50 for comparison
  const baselineMetrics = useMemo(() => {
    return calculateCohortMetrics(profile, 0.50);
  }, [profile]);

  const deltaFpr = metrics.fpr - baselineMetrics.fpr;
  const deltaTpr = metrics.tpr - baselineMetrics.tpr;

  const isAtCalibrated = Math.abs(threshold - profile.calibratedTau) < 0.015;

  return (
    <Card className="border-2 shadow-lg overflow-hidden">
      <CardHeader className="border-b bg-muted/20 pb-5">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div>
            <div className="flex items-center gap-2">
              <Scale className="w-5 h-5 text-primary" />
              <CardTitle className="text-xl">Interactive Demographic Threshold Simulator</CardTitle>
              <Badge variant="outline" className="text-xs bg-primary/10 text-primary border-primary/30">
                What-If Analysis
              </Badge>
            </div>
            <CardDescription className="pt-1">
              Simulate decision boundary adjustments in real-time. Analyze the mathematical trade-off between false alarms (PDRP) and detection evasion (DDRP).
            </CardDescription>
          </div>

          {/* Cohort Selector */}
          <div className="w-full md:w-64">
            <Select value={selectedKey} onValueChange={(val) => {
              if (val) {
                setSelectedKey(val);
                setThreshold(COHORT_PROFILES[val]?.calibratedTau ?? 0.50);
              }
            }}>
              <SelectTrigger className="w-full">
                <SelectValue placeholder="Select demographic cohort" />
              </SelectTrigger>
              <SelectContent>
                {Object.entries(COHORT_PROFILES).map(([k, p]) => (
                  <SelectItem key={k} value={k}>
                    {p.label}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
        </div>
      </CardHeader>

      <CardContent className="pt-6 space-y-6">
        {/* Slider Controls */}
        <div className="p-4 rounded-xl bg-muted/30 border space-y-4">
          <div className="flex items-center justify-between flex-wrap gap-2">
            <div>
              <span className="text-xs font-semibold uppercase text-muted-foreground tracking-wider">
                Operating Decision Boundary (τ)
              </span>
              <div className="flex items-baseline gap-2 pt-0.5">
                <span className="text-3xl font-extrabold font-mono text-primary">
                  {threshold.toFixed(2)}
                </span>
                <span className="text-xs text-muted-foreground font-mono">
                  ({threshold - 0.50 >= 0 ? "+" : ""}{(threshold - 0.50).toFixed(2)} from standard 0.50)
                </span>
              </div>
            </div>

            {/* Quick Presets */}
            <div className="flex flex-wrap items-center gap-1.5">
              <Button
                variant={threshold === 0.50 ? "default" : "outline"}
                size="sm"
                className="text-xs h-8"
                onClick={() => setThreshold(0.50)}
              >
                <Zap className="w-3 h-3 mr-1" /> Fixed 0.50
              </Button>
              <Button
                variant={isAtCalibrated ? "default" : "outline"}
                size="sm"
                className={`text-xs h-8 ${isAtCalibrated ? "bg-emerald-600 hover:bg-emerald-700 text-white" : ""}`}
                onClick={() => setThreshold(profile.calibratedTau)}
              >
                <ShieldCheck className="w-3 h-3 mr-1" /> Fair Calibrated ({profile.calibratedTau.toFixed(2)})
              </Button>
              <Button
                variant={threshold === 0.40 ? "default" : "outline"}
                size="sm"
                className="text-xs h-8"
                onClick={() => setThreshold(0.40)}
              >
                Evasion Defense (0.40)
              </Button>
            </div>
          </div>

          {/* Interactive Range Input */}
          <div className="space-y-2">
            <input
              type="range"
              min="0.10"
              max="0.90"
              step="0.01"
              value={threshold}
              onChange={(e) => setThreshold(parseFloat(e.target.value))}
              className="w-full h-2 bg-muted rounded-lg appearance-none cursor-pointer accent-primary"
            />
            <div className="flex justify-between text-[11px] text-muted-foreground font-mono">
              <span>0.10 (Max Aggressive)</span>
              <span className="text-foreground font-semibold">0.50 (Standard)</span>
              <span className="text-emerald-600 dark:text-emerald-400 font-semibold">{profile.calibratedTau.toFixed(2)} (Fair-Optimal)</span>
              <span>0.90 (Max Conservative)</span>
            </div>
          </div>

          <p className="text-xs text-muted-foreground italic">
            {profile.description}
          </p>
        </div>

        {/* Dynamic Metric Gauges */}
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
          <div className="p-3.5 rounded-xl border bg-muted/20 space-y-1">
            <div className="text-[11px] uppercase font-semibold text-muted-foreground">False Alarm Rate (FPR)</div>
            <div className="text-2xl font-bold font-mono">{(metrics.fpr * 100).toFixed(1)}%</div>
            <div className="flex items-center text-xs gap-1">
              {deltaFpr < 0 ? (
                <span className="text-emerald-600 dark:text-emerald-400 flex items-center font-medium">
                  <TrendingDown className="w-3.5 h-3.5 mr-0.5" /> {(deltaFpr * 100).toFixed(1)}% vs 0.50
                </span>
              ) : deltaFpr > 0 ? (
                <span className="text-rose-600 flex items-center font-medium">
                  <TrendingUp className="w-3.5 h-3.5 mr-0.5" /> +{(deltaFpr * 100).toFixed(1)}% vs 0.50
                </span>
              ) : (
                <span className="text-muted-foreground">Baseline</span>
              )}
            </div>
          </div>

          <div className="p-3.5 rounded-xl border bg-muted/20 space-y-1">
            <div className="text-[11px] uppercase font-semibold text-muted-foreground">Fake Sensitivity (TPR)</div>
            <div className="text-2xl font-bold font-mono">{(metrics.tpr * 100).toFixed(1)}%</div>
            <div className="flex items-center text-xs gap-1">
              {deltaTpr >= 0 ? (
                <span className="text-emerald-600 dark:text-emerald-400 flex items-center font-medium">
                  <TrendingUp className="w-3.5 h-3.5 mr-0.5" /> +{(deltaTpr * 100).toFixed(1)}% vs 0.50
                </span>
              ) : (
                <span className="text-amber-600 flex items-center font-medium">
                  <TrendingDown className="w-3.5 h-3.5 mr-0.5" /> {(deltaTpr * 100).toFixed(1)}% vs 0.50
                </span>
              )}
            </div>
          </div>

          <div className="p-3.5 rounded-xl border bg-muted/20 space-y-1">
            <div className="text-[11px] uppercase font-semibold text-muted-foreground">False Alarm Bias (PDRP)</div>
            <div className={`text-2xl font-bold font-mono ${
              metrics.pdrp > 0.30 ? "text-rose-600" : metrics.pdrp < 0.10 ? "text-emerald-600 dark:text-emerald-400" : "text-amber-600"
            }`}>
              {metrics.pdrp >= 0 ? "+" : ""}{metrics.pdrp.toFixed(2)}
            </div>
            <div className="text-[11px] text-muted-foreground">
              {metrics.pdrp < 0.10 ? "Parity achieved" : "Error inflation"}
            </div>
          </div>

          <div className="p-3.5 rounded-xl border bg-muted/20 space-y-1">
            <div className="text-[11px] uppercase font-semibold text-muted-foreground">Evasion Bias (DDRP)</div>
            <div className={`text-2xl font-bold font-mono ${
              metrics.ddrp > 0.30 ? "text-rose-600" : metrics.ddrp < 0.15 ? "text-emerald-600 dark:text-emerald-400" : "text-amber-600"
            }`}>
              {metrics.ddrp >= 0 ? "+" : ""}{metrics.ddrp.toFixed(2)}
            </div>
            <div className="text-[11px] text-muted-foreground">
              {metrics.ddrp < 0.15 ? "Resistant to evasion" : "Vulnerability gap"}
            </div>
          </div>
        </div>

        {/* Real-Time Confusion Matrix Visualizer */}
        <div className="space-y-3">
          <div className="flex items-center justify-between">
            <h4 className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
              Simulated Confusion Matrix ({profile.label}, N = {profile.sampleTotal})
            </h4>
            <span className="text-xs font-mono text-muted-foreground">
              Accuracy: {(metrics.accuracy * 100).toFixed(1)}%
            </span>
          </div>

          <div className="grid grid-cols-2 gap-3 text-xs">
            {/* Top-Left: TN */}
            <div className="p-3.5 rounded-xl border bg-emerald-500/10 border-emerald-500/30 space-y-1">
              <div className="flex justify-between items-center text-emerald-700 dark:text-emerald-300 font-semibold">
                <span>True Negatives (TN)</span>
                <CheckCircle2 className="w-4 h-4" />
              </div>
              <div className="text-2xl font-extrabold font-mono text-emerald-800 dark:text-emerald-200">{metrics.tn}</div>
              <p className="text-[11px] text-muted-foreground">Genuine portraits correctly verified</p>
            </div>

            {/* Top-Right: FP */}
            <div className={`p-3.5 rounded-xl border space-y-1 transition-colors ${
              metrics.fp > baselineMetrics.fp ? "bg-rose-500/15 border-rose-500/40" : "bg-rose-500/10 border-rose-500/20"
            }`}>
              <div className="flex justify-between items-center text-rose-700 dark:text-rose-300 font-semibold">
                <span>False Positives (FP)</span>
                <AlertTriangle className="w-4 h-4" />
              </div>
              <div className="text-2xl font-extrabold font-mono text-rose-800 dark:text-rose-200">{metrics.fp}</div>
              <p className="text-[11px] text-muted-foreground">Innocent portraits falsely accused</p>
            </div>

            {/* Bottom-Left: FN */}
            <div className="p-3.5 rounded-xl border bg-amber-500/10 border-amber-500/20 space-y-1">
              <div className="flex justify-between items-center text-amber-700 dark:text-amber-300 font-semibold">
                <span>False Negatives (FN)</span>
                <AlertTriangle className="w-4 h-4" />
              </div>
              <div className="text-2xl font-extrabold font-mono text-amber-800 dark:text-amber-200">{metrics.fn}</div>
              <p className="text-[11px] text-muted-foreground">Deepfakes evading detection</p>
            </div>

            {/* Bottom-Right: TP */}
            <div className="p-3.5 rounded-xl border bg-emerald-500/10 border-emerald-500/30 space-y-1">
              <div className="flex justify-between items-center text-emerald-700 dark:text-emerald-300 font-semibold">
                <span>True Positives (TP)</span>
                <CheckCircle2 className="w-4 h-4" />
              </div>
              <div className="text-2xl font-extrabold font-mono text-emerald-800 dark:text-emerald-200">{metrics.tp}</div>
              <p className="text-[11px] text-muted-foreground">Deepfakes successfully intercepted</p>
            </div>
          </div>
        </div>

        {/* Live Operational Insight Banner */}
        <div className="p-3.5 rounded-xl bg-muted/40 border border-border flex items-start gap-3 text-xs leading-relaxed">
          <Info className="w-4 h-4 text-primary shrink-0 mt-0.5" />
          <div className="space-y-1">
            <span className="font-semibold text-foreground">Operational Trade-off Analysis:</span>
            <p className="text-muted-foreground">
              {threshold > 0.50 ? (
                <>
                  Operating at <strong className="text-foreground">τ = {threshold.toFixed(2)}</strong> lowers false alarm risk by <strong className="text-emerald-600 dark:text-emerald-400">{Math.abs(deltaFpr * 100).toFixed(1)}%</strong> on {profile.label}. PDRP disparity drops from {baselineMetrics.pdrp >= 0 ? "+" : ""}{baselineMetrics.pdrp.toFixed(2)} to {metrics.pdrp >= 0 ? "+" : ""}{metrics.pdrp.toFixed(2)}, clearing spurious makeup/shading false flags.
                </>
              ) : threshold < 0.50 ? (
                <>
                  Operating aggressively at <strong className="text-foreground">τ = {threshold.toFixed(2)}</strong> catches <strong className="text-emerald-600 dark:text-emerald-400">{(deltaTpr * 100).toFixed(1)}%</strong> more evasive deepfakes, but increases false accusations on innocent portraits by +{(deltaFpr * 100).toFixed(1)}%.
                </>
              ) : (
                <>
                  Standard nominal threshold <strong className="text-foreground">τ = 0.50</strong> incurs high false alarm inflation on {profile.label} (FPR: {(baselineMetrics.fpr * 100).toFixed(1)}%, PDRP: +{baselineMetrics.pdrp.toFixed(2)}). Click <em>Fair Calibrated</em> to equalize demographic risk.
                </>
              )}
            </p>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
