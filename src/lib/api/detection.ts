import { DetectionResult } from "@/lib/types";
import demoData from "@/lib/mock-data/demo-files.json";

export async function getDetectionResult(id: string): Promise<DetectionResult | null> {
  // Simulate network delay
  await new Promise((resolve) => setTimeout(resolve, 800));
  
  const result = (demoData as unknown as DetectionResult[]).find((item) => item.id === id);
  return result || null;
}

export async function getAllDemoFiles(): Promise<DetectionResult[]> {
  await new Promise((resolve) => setTimeout(resolve, 300));
  return demoData as unknown as DetectionResult[];
}
