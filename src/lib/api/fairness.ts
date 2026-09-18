import { AuditSummary, AttributeBiasResult } from "@/lib/types";
import auditData from "@/lib/mock-data/audit-data.json";

export async function getAuditSummaries(): Promise<AuditSummary[]> {
  await new Promise((resolve) => setTimeout(resolve, 500));
  return auditData.summaries as AuditSummary[];
}

export async function getAuditResults(model?: string): Promise<AttributeBiasResult[]> {
  await new Promise((resolve) => setTimeout(resolve, 600));
  let results = auditData.auditResults as AttributeBiasResult[];
  
  if (model) {
    results = results.filter((r) => r.model === model);
  }
  
  return results;
}

export async function getAvailableModels(): Promise<string[]> {
  const summaries = await getAuditSummaries();
  return summaries.map((s) => s.model);
}
