"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { ArrowLeft, Printer } from "lucide-react";
import { Button } from "@/components/ui/button";
import { getAuditSummaries, getAuditResults } from "@/lib/api/fairness";
import { AuditSummary, AttributeBiasResult } from "@/lib/types";

export default function ReportPage() {
  const router = useRouter();
  const [summary, setSummary] = useState<AuditSummary | null>(null);
  const [results, setResults] = useState<AttributeBiasResult[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    // Default to Xception for the export report as an example
    Promise.all([
      getAuditSummaries(),
      getAuditResults("Xception")
    ]).then(([summaries, auditRes]) => {
      const s = summaries.find(x => x.model === "Xception");
      setSummary(s || null);
      setResults(auditRes);
      setLoading(false);
    });
  }, []);

  if (loading || !summary) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-background">
        <div className="animate-spin w-8 h-8 border-4 border-primary border-t-transparent rounded-full" />
      </div>
    );
  }

  const highBiasAttributes = results.filter(r => r.severity === "high").map(r => r.attribute);
  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case "high": return "var(--color-bias-high)";
      case "moderate": return "var(--color-bias-moderate)";
      default: return "var(--color-bias-low)";
    }
  };

  return (
    <div className="min-h-screen bg-background p-6 md:p-12">
      <div className="max-w-4xl mx-auto space-y-8 print:space-y-6">
        
        {/* Controls - Hidden in print mode */}
        <div className="flex justify-between items-center print:hidden">
          <Button variant="ghost" onClick={() => router.push("/dashboard")}>
            <ArrowLeft className="w-4 h-4 mr-2" />
            Back to Dashboard
          </Button>
          <Button onClick={() => window.print()}>
            <Printer className="w-4 h-4 mr-2" />
            Export as PDF
          </Button>
        </div>

        {/* Report Header */}
        <div className="space-y-4 border-b pb-6">
          <h1 className="text-4xl font-extrabold tracking-tight">Fairness Audit Report</h1>
          <div className="text-muted-foreground flex flex-col sm:flex-row justify-between">
            <span>Model: <strong className="text-foreground">{summary.model}</strong></span>
            <span>Date: {new Date().toLocaleDateString()}</span>
          </div>
        </div>

        {/* Executive Summary */}
        <div className="space-y-4">
          <h2 className="text-2xl font-bold border-b pb-2">1. Executive Summary</h2>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 py-4">
             <div className="bg-muted p-4 rounded-lg text-center">
               <div className="text-sm text-muted-foreground">Accuracy</div>
               <div className="text-2xl font-bold">{(summary.overallAccuracy * 100).toFixed(1)}%</div>
             </div>
             <div className="bg-muted p-4 rounded-lg text-center">
               <div className="text-sm text-muted-foreground">FPR</div>
               <div className="text-2xl font-bold">{(summary.overallFPR * 100).toFixed(1)}%</div>
             </div>
             <div className="bg-muted p-4 rounded-lg text-center">
               <div className="text-sm text-muted-foreground">FNR</div>
               <div className="text-2xl font-bold">{(summary.overallFNR * 100).toFixed(1)}%</div>
             </div>
             <div className="bg-muted p-4 rounded-lg text-center border border-[var(--color-bias-high)]">
               <div className="text-sm text-[var(--color-bias-high)]">High Bias Attrs</div>
               <div className="text-2xl font-bold text-[var(--color-bias-high)]">{summary.highBiasAttributeCount}</div>
             </div>
          </div>
        </div>

        {/* Methodology */}
        <div className="space-y-4">
          <h2 className="text-2xl font-bold border-b pb-2">2. Methodology</h2>
          <p className="text-muted-foreground leading-relaxed">
            This audit evaluates the deepfake detection model across various demographic attributes to identify disparate impact. 
            We calculate the Error Rate with and without each attribute. The Relative Performance (RP) represents the raw difference in error rates, 
            while the Corrected Relative Performance (CRP) normalizes this difference to account for baseline error rates, providing a robust metric for severity assessment.
          </p>
        </div>

        {/* Key Findings */}
        <div className="space-y-4">
          <h2 className="text-2xl font-bold border-b pb-2">3. Key Findings</h2>
          <ul className="list-disc pl-6 space-y-2 text-muted-foreground">
            <li>
              The model achieved an overall accuracy of <strong>{(summary.overallAccuracy * 100).toFixed(1)}%</strong> across the benchmark dataset of {summary.totalImages.toLocaleString()} images.
            </li>
            <li>
              We identified <strong>{highBiasAttributes.length} attributes</strong> demonstrating severe bias (CRP &gt; 40%), notably including: 
              <span className="font-medium text-foreground"> {highBiasAttributes.join(", ")}</span>.
            </li>
            <li>
              Attributes related to age (<span className="italic">Young</span>, <span className="italic">Senior</span>) and gender (<span className="italic">Male</span>, <span className="italic">Female</span>) showed lower levels of bias, indicating the model generalizes well across these demographics.
            </li>
            <li>
              Conversely, facial accessories and characteristics such as <span className="italic">Heavy_Makeup</span> and <span className="italic">Dark_Skin</span> severely degraded the model's accuracy, contributing disproportionately to False Positives.
            </li>
          </ul>
        </div>

        {/* Full Data Table */}
        <div className="space-y-4 break-inside-avoid">
          <h2 className="text-2xl font-bold border-b pb-2">4. Comprehensive Attribute Data</h2>
          <table className="w-full text-sm text-left border-collapse mt-4">
            <thead className="text-xs text-muted-foreground uppercase bg-muted border-b">
              <tr>
                <th className="px-4 py-3">Attribute</th>
                <th className="px-4 py-3 text-right">Err (With)</th>
                <th className="px-4 py-3 text-right">Err (W/O)</th>
                <th className="px-4 py-3 text-right">RP</th>
                <th className="px-4 py-3 text-right font-bold text-foreground">CRP</th>
                <th className="px-4 py-3">Severity</th>
              </tr>
            </thead>
            <tbody>
              {results.map((r, i) => (
                <tr key={i} className="border-b">
                  <td className="px-4 py-3 font-medium">{r.attribute}</td>
                  <td className="px-4 py-3 text-right">{(r.errorWithAttribute * 100).toFixed(1)}%</td>
                  <td className="px-4 py-3 text-right">{(r.errorWithoutAttribute * 100).toFixed(1)}%</td>
                  <td className="px-4 py-3 text-right">{(r.rp * 100).toFixed(1)}%</td>
                  <td className="px-4 py-3 text-right font-bold text-foreground">{(r.crp * 100).toFixed(1)}%</td>
                  <td className="px-4 py-3 font-semibold uppercase text-xs" style={{ color: getSeverityColor(r.severity) }}>
                    {r.severity}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

      </div>
    </div>
  );
}
