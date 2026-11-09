import { DetectionResult } from "@/lib/types";
import demoData from "@/lib/mock-data/demo-files.json";

export * from "./calibration";

export async function getDetectionResult(id: string): Promise<DetectionResult | null> {
  // Simulate network delay
  await new Promise((resolve) => setTimeout(resolve, 300));
  
  const result = (demoData as unknown as DetectionResult[]).find((item) => item.id === id);
  return result || null;
}

export async function getAllDemoFiles(): Promise<DetectionResult[]> {
  await new Promise((resolve) => setTimeout(resolve, 150));
  return demoData as unknown as DetectionResult[];
}
