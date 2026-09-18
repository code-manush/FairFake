"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { 
  ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine, Label, Cell 
} from "recharts";
import { ArrowLeft, ExternalLink, Download } from "lucide-react";

import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Badge } from "@/components/ui/badge";
import { ScrollArea } from "@/components/ui/scroll-area";
import { getAuditSummaries, getAuditResults, getAvailableModels } from "@/lib/api/fairness";
import { AuditSummary, AttributeBiasResult } from "@/lib/types";

export default function DashboardPage() {
  const router = useRouter();
  const [models, setModels] = useState<string[]>([]);
  const [activeModel, setActiveModel] = useState<string>("");
  
  const [summary, setSummary] = useState<AuditSummary | null>(null);
  const [results, setResults] = useState<AttributeBiasResult[]>([]);
  const [loading, setLoading] = useState(true);
  
  const [dataFilter, setDataFilter] = useState("both"); // pristine, fake, both
  const [selectedAttribute, setSelectedAttribute] = useState<AttributeBiasResult | null>(null);

  useEffect(() => {
    getAvailableModels().then(m => {
      setModels(m);
      if (m.length > 0) setActiveModel(m[0]);
    });
  }, []);

  useEffect(() => {
    if (!activeModel) return;
    setLoading(true);
    
    Promise.all([
      getAuditSummaries(),
      getAuditResults(activeModel)
    ]).then(([summaries, auditRes]) => {
      const s = summaries.find(x => x.model === activeModel);
      setSummary(s || null);
      setResults(auditRes);
      setSelectedAttribute(null);
      setLoading(false);
    });
  }, [activeModel]);

  // Scatter plot data formatting
  const scatterData = results.map(r => ({
    x: r.crp * 100, // format as percentage
    y: r.rp * 100,
    z: r.sampleCount.withAttribute, // circle size
    name: r.attribute,
    severity: r.severity
  }));

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case "high": return "var(--color-bias-high)";
      case "moderate": return "var(--color-bias-moderate)";
      default: return "var(--color-bias-low)";
    }
  };

  return (
    <div className="min-h-screen bg-background p-6 pb-24">
      <div className="max-w-7xl mx-auto space-y-6">
        
        {/* Header */}
        <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4">
          <div>
            <Button variant="ghost" className="mb-2 -ml-4" onClick={() => router.push("/")}>
              <ArrowLeft className="w-4 h-4 mr-2" />
              Back to Home
            </Button>
            <h1 className="text-3xl font-bold tracking-tight">Fairness Dashboard</h1>
            <p className="text-muted-foreground">Audit of deepfake detector performance across demographic attributes.</p>
          </div>
          
          <div className="flex items-center gap-4">
            <Button variant="outline" onClick={() => router.push("/dashboard/report")}>
              <Download className="w-4 h-4 mr-2" />
              Export Report
            </Button>
            <div className="w-48">
              <Select value={activeModel} onValueChange={(val) => setActiveModel(val || "")}>
                <SelectTrigger>
                  <SelectValue placeholder="Select a model" />
                </SelectTrigger>
                <SelectContent>
                  {models.map(m => (
                    <SelectItem key={m} value={m}>{m}</SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
          </div>
        </div>

        {loading || !summary ? (
          <div className="h-64 flex items-center justify-center border rounded-xl">
            <div className="animate-spin w-8 h-8 border-4 border-primary border-t-transparent rounded-full" />
          </div>
        ) : (
          <>
            {/* Top Summary Cards */}
            <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
              <Card>
                <CardHeader className="pb-2">
                  <CardTitle className="text-sm font-medium text-muted-foreground">Overall Accuracy</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="text-3xl font-bold">{(summary.overallAccuracy * 100).toFixed(1)}%</div>
                </CardContent>
              </Card>
              <Card>
                <CardHeader className="pb-2">
                  <CardTitle className="text-sm font-medium text-muted-foreground">False Positive Rate</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="text-3xl font-bold">{(summary.overallFPR * 100).toFixed(1)}%</div>
                </CardContent>
              </Card>
              <Card>
                <CardHeader className="pb-2">
                  <CardTitle className="text-sm font-medium text-muted-foreground">False Negative Rate</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="text-3xl font-bold">{(summary.overallFNR * 100).toFixed(1)}%</div>
                </CardContent>
              </Card>
              <Card className="border-[var(--color-bias-high)]">
                <CardHeader className="pb-2">
                  <CardTitle className="text-sm font-medium text-[var(--color-bias-high)]">High Bias Attributes</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="text-3xl font-bold text-[var(--color-bias-high)]">{summary.highBiasAttributeCount}</div>
                </CardContent>
              </Card>
            </div>

            <div className="grid lg:grid-cols-3 gap-6">
              {/* Scatter Plot */}
              <Card className="lg:col-span-2">
                <CardHeader className="flex flex-row items-center justify-between pb-2">
                  <div className="space-y-1">
                    <CardTitle>RP vs CRP by Attribute</CardTitle>
                    <CardDescription>
                      Corrected Relative Performance (CRP) vs Relative Performance (RP). 
                      Points further to the right indicate severe bias.
                    </CardDescription>
                  </div>
                  <Tabs value={dataFilter} onValueChange={setDataFilter}>
                    <TabsList>
                      <TabsTrigger value="pristine">Pristine</TabsTrigger>
                      <TabsTrigger value="fake">Fake</TabsTrigger>
                      <TabsTrigger value="both">Both</TabsTrigger>
                    </TabsList>
                  </Tabs>
                </CardHeader>
                <CardContent>
                  <div className="h-[400px] w-full mt-4">
                    <ResponsiveContainer width="100%" height="100%">
                      <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
                        <CartesianGrid strokeDasharray="3 3" opacity={0.2} />
                        <XAxis 
                          type="number" 
                          dataKey="x" 
                          name="CRP (%)" 
                          unit="%" 
                          domain={[0, 'dataMax + 10']}
                        >
                          <Label value="Corrected Relative Performance (CRP)" offset={-10} position="insideBottom" />
                        </XAxis>
                        <YAxis 
                          type="number" 
                          dataKey="y" 
                          name="RP (%)" 
                          unit="%" 
                          domain={['dataMin - 10', 'dataMax + 10']}
                        >
                          <Label value="Relative Performance (RP)" angle={-90} position="insideLeft" style={{ textAnchor: 'middle' }} />
                        </YAxis>
                        <Tooltip 
                          cursor={{ strokeDasharray: '3 3' }} 
                          content={({ active, payload }) => {
                            if (active && payload && payload.length) {
                              const data = payload[0].payload;
                              return (
                                <div className="bg-popover text-popover-foreground border p-3 rounded-lg shadow-lg">
                                  <p className="font-bold mb-1">{data.name}</p>
                                  <p className="text-sm">CRP: {data.x.toFixed(2)}%</p>
                                  <p className="text-sm">RP: {data.y.toFixed(2)}%</p>
                                  <p className="text-sm">Severity: <span style={{color: getSeverityColor(data.severity)}} className="uppercase font-semibold">{data.severity}</span></p>
                                </div>
                              );
                            }
                            return null;
                          }}
                        />
                        {/* Shading zones can be simulated using ReferenceAreas, but simple colored scatter works well */}
                        <Scatter 
                          name="Attributes" 
                          data={scatterData} 
                          onClick={(data: any) => {
                            const name = data?.name || data?.payload?.name;
                            const res = results.find(r => r.attribute === name);
                            if (res) setSelectedAttribute(res);
                          }}
                        >
                          {scatterData.map((entry, index) => (
                            <Cell key={`cell-${index}`} fill={getSeverityColor(entry.severity)} className="cursor-pointer hover:opacity-80 transition-opacity" />
                          ))}
                        </Scatter>
                        {/* Bisector line */}
                        <ReferenceLine x={20} stroke="var(--color-bias-moderate)" strokeDasharray="3 3" label={{ position: 'top', value: 'Moderate Threshold', fill: 'var(--color-bias-moderate)', fontSize: 12 }} />
                        <ReferenceLine x={40} stroke="var(--color-bias-high)" strokeDasharray="3 3" label={{ position: 'top', value: 'High Threshold', fill: 'var(--color-bias-high)', fontSize: 12 }} />
                      </ScatterChart>
                    </ResponsiveContainer>
                  </div>
                </CardContent>
              </Card>

              {/* Drill-down panel */}
              <Card className="flex flex-col h-[520px]">
                <CardHeader>
                  <CardTitle>Attribute Details</CardTitle>
                  <CardDescription>Select an attribute from the chart or table to view misclassifications.</CardDescription>
                </CardHeader>
                <CardContent className="flex-1 overflow-hidden flex flex-col">
                  {selectedAttribute ? (
                    <div className="space-y-6 h-full flex flex-col">
                      <div>
                        <div className="flex items-center justify-between mb-2">
                          <h3 className="text-2xl font-bold">{selectedAttribute.attribute}</h3>
                          <Badge 
                            variant="outline" 
                            className="uppercase"
                            style={{ 
                              color: getSeverityColor(selectedAttribute.severity),
                              borderColor: getSeverityColor(selectedAttribute.severity) 
                            }}
                          >
                            {selectedAttribute.severity} Bias
                          </Badge>
                        </div>
                        <div className="grid grid-cols-2 gap-2 text-sm mt-4">
                          <div className="bg-muted p-2 rounded">
                            <span className="text-muted-foreground">Error (With):</span><br/>
                            <span className="font-semibold">{(selectedAttribute.errorWithAttribute * 100).toFixed(1)}%</span>
                          </div>
                          <div className="bg-muted p-2 rounded">
                            <span className="text-muted-foreground">Error (W/O):</span><br/>
                            <span className="font-semibold">{(selectedAttribute.errorWithoutAttribute * 100).toFixed(1)}%</span>
                          </div>
                        </div>
                      </div>
                      
                      <div className="flex-1 overflow-hidden flex flex-col">
                        <h4 className="font-medium mb-3">Example Misclassifications</h4>
                        <ScrollArea className="flex-1">
                          {selectedAttribute.exampleMisclassifiedIds.length > 0 ? (
                            <div className="grid grid-cols-2 gap-3 pb-4">
                              {selectedAttribute.exampleMisclassifiedIds.map((id, i) => (
                                <div key={i} className="group relative rounded-md overflow-hidden border aspect-square bg-muted flex flex-col items-center justify-center cursor-pointer" onClick={() => router.push(`/analyze/${id}`)}>
                                  <div className="absolute inset-0 bg-background/10 group-hover:bg-background/0 transition-colors z-10" />
                                  <span className="text-xs text-muted-foreground">Mock Face</span>
                                  <div className="absolute bottom-0 left-0 right-0 p-2 bg-gradient-to-t from-black/80 to-transparent z-20">
                                    <div className="flex items-center justify-between text-[10px] text-white">
                                      <span>ID: {id.split('-').pop()}</span>
                                      <ExternalLink className="w-3 h-3" />
                                    </div>
                                  </div>
                                </div>
                              ))}
                            </div>
                          ) : (
                            <div className="flex items-center justify-center h-32 text-muted-foreground text-sm italic">
                              No examples available
                            </div>
                          )}
                        </ScrollArea>
                      </div>
                    </div>
                  ) : (
                    <div className="flex-1 flex items-center justify-center text-muted-foreground text-sm italic h-full">
                      Click a point on the scatter plot to drill down
                    </div>
                  )}
                </CardContent>
              </Card>
            </div>

            {/* Table */}
            <Card>
              <CardHeader>
                <CardTitle>Detailed Audit Results</CardTitle>
                <CardDescription>Comprehensive breakdown of relative performance metrics per demographic.</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="overflow-x-auto">
                  <table className="w-full text-sm text-left border-collapse">
                    <thead className="text-xs text-muted-foreground uppercase bg-muted/50 border-b">
                      <tr>
                        <th className="px-4 py-3">Attribute</th>
                        <th className="px-4 py-3 text-right">Error (With)</th>
                        <th className="px-4 py-3 text-right">Error (W/O)</th>
                        <th className="px-4 py-3 text-right">RP (%)</th>
                        <th className="px-4 py-3 text-right font-bold">CRP (%)</th>
                        <th className="px-4 py-3">Severity</th>
                      </tr>
                    </thead>
                    <tbody>
                      {results.map((r, i) => (
                        <tr 
                          key={i} 
                          className="border-b hover:bg-muted/50 cursor-pointer transition-colors"
                          onClick={() => setSelectedAttribute(r)}
                        >
                          <td className="px-4 py-3 font-medium">{r.attribute}</td>
                          <td className="px-4 py-3 text-right">{(r.errorWithAttribute * 100).toFixed(1)}%</td>
                          <td className="px-4 py-3 text-right">{(r.errorWithoutAttribute * 100).toFixed(1)}%</td>
                          <td className="px-4 py-3 text-right">{(r.rp * 100).toFixed(1)}%</td>
                          <td className="px-4 py-3 text-right font-bold">{(r.crp * 100).toFixed(1)}%</td>
                          <td className="px-4 py-3">
                            <Badge 
                              variant="outline"
                              style={{ 
                                color: getSeverityColor(r.severity),
                                borderColor: getSeverityColor(r.severity) 
                              }}
                            >
                              {r.severity}
                            </Badge>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </CardContent>
            </Card>
          </>
        )}
      </div>
    </div>
  );
}
