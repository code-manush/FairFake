import { AuditSummary, AttributeBiasResult } from "@/lib/types";

export async function getAuditSummaries(): Promise<AuditSummary[]> {
  const res = await fetch("http://localhost:8000/api/audit/results");
  if (!res.ok) throw new Error("Failed to fetch audit data");
  const data = await res.json();
  return data.summaries as AuditSummary[];
}

export async function getAuditResults(model?: string): Promise<AttributeBiasResult[]> {
  const res = await fetch("http://localhost:8000/api/audit/results");
  if (!res.ok) throw new Error("Failed to fetch audit data");
  const data = await res.json();
  let results = data.auditResults as AttributeBiasResult[];
  
  if (model) {
    results = results.filter((r) => r.model === model);
  }
  
  return results;
}

export async function getAvailableModels(): Promise<string[]> {
  const summaries = await getAuditSummaries();
  return summaries.map((s) => s.model);
}
