"use client";

import { useEffect, useState } from "react";
import { useParams, useRouter } from "next/navigation";
import { motion } from "framer-motion";
import { ArrowLeft, CheckCircle, XCircle, AlertTriangle, Image as ImageIcon } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { DetectionResult } from "@/lib/types";

// Mock helper to get bias severity note
const getBiasNote = (attribute: string) => {
  const biasedAttributes = ["Wearing_Glasses", "Heavy_Makeup", "Dark_Skin", "Bald"];
  if (biasedAttributes.includes(attribute)) {
    return "high";
  }
  return "low";
};

export default function AnalysisReport() {
  const params = useParams();
  const router = useRouter();
  const [result, setResult] = useState<DetectionResult | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (!params.id) return;
    
    // Load from sessionStorage where page.tsx saved it
    const stored = sessionStorage.getItem(`detection_${params.id}`);
    if (stored) {
      setResult(JSON.parse(stored));
    }
    setLoading(false);
  }, [params.id]);

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-background">
        <div className="animate-pulse flex flex-col items-center">
          <div className="w-12 h-12 rounded-full border-4 border-primary border-t-transparent animate-spin mb-4" />
          <p className="text-muted-foreground">Loading analysis...</p>
        </div>
      </div>
    );
  }

  if (!result) {
    return (
      <div className="min-h-screen flex items-center justify-center flex-col space-y-4">
        <p className="text-xl">Analysis not found. Please upload a new file.</p>
        <Button onClick={() => router.push("/")}>Return Home</Button>
      </div>
    );
  }

  const isReal = result.verdict === "REAL";
  const isFake = result.verdict === "FAKE" || result.verdict === "LIKELY FAKE";
  
  const VerdictIcon = isReal ? CheckCircle : isFake ? XCircle : AlertTriangle;
  const verdictColorClass = isReal ? "text-[var(--color-verdict-real)] border-[var(--color-verdict-real)]" : 
                            isFake ? "text-[var(--color-verdict-fake)] border-[var(--color-verdict-fake)]" : "text-[var(--color-verdict-uncertain)] border-[var(--color-verdict-uncertain)]";
  
  const authenticityScore = result.mediaType === "image" 
    ? result.image_authenticity_score 
    : result.video_authenticity_score;

  return (
    <div className="min-h-screen bg-background p-6 pb-24">
      <div className="max-w-6xl mx-auto space-y-8">
        <Button variant="ghost" className="mb-4" onClick={() => router.push("/")}>
          <ArrowLeft className="w-4 h-4 mr-2" />
          Back to Upload
        </Button>

        <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} className="flex flex-col lg:flex-row gap-8">
          
          {/* Main Analysis Section */}
          <div className="flex-1 space-y-6">
            <Card className="border-2 shadow-lg">
              <CardHeader className="border-b bg-muted/20 pb-6">
                <div className="flex items-center justify-between flex-wrap gap-4">
                  <div>
                    <CardTitle className="text-2xl">Detection Report</CardTitle>
                    <CardDescription>File ID: {result.id}</CardDescription>
                  </div>
                  <div className={`px-4 py-2 rounded-full border-2 flex items-center font-bold uppercase tracking-wider ${verdictColorClass}`}>
                    <VerdictIcon className="w-5 h-5 mr-2" />
                    {result.verdict || "UNKNOWN"}
                  </div>
                </div>
              </CardHeader>
              <CardContent className="pt-6 space-y-8">
                
                {/* Score breakdown */}
                <div className="space-y-3">
                  <div className="flex justify-between items-end">
                    <h3 className="text-sm font-semibold uppercase text-muted-foreground tracking-wider">Authenticity Score</h3>
                    <span className="text-3xl font-bold">{authenticityScore ?? 0}%</span>
                  </div>
                  <Progress value={authenticityScore ?? 0} className="h-3" />
                  <p className="text-sm text-muted-foreground pt-2">
                    {result.mediaType === 'image' 
                      ? `Based on Xception deepfake detection. 1 image analyzed.` 
                      : `Video analyzed. ${result.total_frames_analyzed} frames processed over ${result.duration_seconds}s.`}
                  </p>
                </div>
                
                {/* Video Metrics */}
                {result.mediaType === "video" && (
                  <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 p-4 bg-muted/30 rounded-lg">
                    <div>
                      <div className="text-sm text-muted-foreground uppercase">Duration</div>
                      <div className="text-xl font-semibold">{result.duration_seconds}s</div>
                    </div>
                    <div>
                      <div className="text-sm text-muted-foreground uppercase">Frames</div>
                      <div className="text-xl font-semibold">{result.total_frames_analyzed}</div>
                    </div>
                    <div>
                      <div className="text-sm text-muted-foreground uppercase">Max Fake Prob</div>
                      <div className="text-xl font-semibold">{((result.max_fake_probability || 0) * 100).toFixed(1)}%</div>
                    </div>
                    <div>
                      <div className="text-sm text-muted-foreground uppercase">Anomalies</div>
                      <div className="text-xl font-semibold">{result.anomaly_seconds?.length || 0} secs</div>
                    </div>
                  </div>
                )}
              </CardContent>
            </Card>

            {/* Visual Evidence (TruthLenss style Heatmaps) */}
            <Card>
              <CardHeader>
                <CardTitle>Suspicious Regions (Grad-CAM)</CardTitle>
                <CardDescription>Heatmaps indicate areas the model focused on to make its decision.</CardDescription>
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
                        <div className="flex justify-between items-center text-sm">
                          <Badge variant={imgRes.verdict === "FAKE" ? "destructive" : "secondary"}>{imgRes.verdict}</Badge>
                          <span>Prob: {(imgRes.fake_probability * 100).toFixed(1)}%</span>
                        </div>
                      </div>
                    ))}
                  </div>
                )}

                {result.mediaType === "video" && result.frame_results && (
                  <div className="space-y-4">
                    <h3 className="font-semibold text-sm uppercase text-muted-foreground">Top Suspicious Frames</h3>
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
                <CardTitle className="text-lg">Detected Attributes</CardTitle>
                <CardDescription>Demographic groups identified in this media.</CardDescription>
              </CardHeader>
              <CardContent className="space-y-6">
                {result.faces && result.faces.length > 0 ? result.faces.map((face, faceIdx) => (
                  <div key={faceIdx} className="space-y-4">
                    <div className="flex flex-wrap gap-2">
                      {Object.entries(face.attributes).map(([key, value]) => {
                        const biasSeverity = getBiasNote(key);
                        const isHighBias = biasSeverity === "high";
                        
                        return (
                          <Badge 
                            key={key} 
                            variant="secondary" 
                            className={`px-3 py-1 ${isHighBias ? 'border-[var(--color-bias-high)] border text-[var(--color-bias-high)] bg-transparent' : ''}`}
                          >
                            {key}: {value.toString()}
                          </Badge>
                        );
                      })}
                    </div>
                    
                    {/* Bias Warning Note */}
                    {Object.keys(face.attributes).some(k => getBiasNote(k) === "high") && (
                      <div className="bg-muted/40 p-3 rounded-md border-l-4 border-[var(--color-bias-high)] text-sm space-y-2">
                        <div className="flex items-center text-[var(--color-bias-high)] font-semibold">
                          <AlertTriangle className="w-4 h-4 mr-1.5" />
                          Fairness Warning
                        </div>
                        <p className="text-muted-foreground">
                          This detector has measured <span className="font-medium text-foreground">high bias</span> for some attributes detected here (e.g. Dark_Skin, Heavy_Makeup). See the <a href="/dashboard" className="underline hover:text-primary">Fairness Dashboard</a>.
                        </p>
                      </div>
                    )}
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
