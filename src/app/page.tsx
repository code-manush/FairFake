"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { motion } from "framer-motion";
import { UploadCloud, Image as ImageIcon, Film, PlayCircle, Loader2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";

export default function LandingPage() {
  const router = useRouter();
  const [isUploading, setIsUploading] = useState(false);
  const [activeDemo, setActiveDemo] = useState<string | null>(null);

  const handleUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    e.preventDefault();
    if (!e.target.files || e.target.files.length === 0) return;
    
    setIsUploading(true);
    const file = e.target.files[0];
    const isVideo = file.type.startsWith("video/");
    
    try {
      const formData = new FormData();
      formData.append(isVideo ? "video" : "image", file);
      
      const res = await fetch(`http://localhost:8000/api/analyze/${isVideo ? 'video' : 'image'}`, {
        method: "POST",
        body: formData,
      });
      
      if (!res.ok) throw new Error("Analysis failed");
      const data = await res.json();
      
      const id = crypto.randomUUID();
      data.id = id;
      sessionStorage.setItem(`detection_${id}`, JSON.stringify(data));
      router.push(`/analyze/${id}`);
    } catch (err) {
      console.error(err);
      alert("Failed to analyze media. Is the FastAPI backend running?");
      setIsUploading(false);
    }
  };

  const handleDemoClick = (id: string) => {
    setActiveDemo(id);
    setTimeout(() => {
      router.push(`/analyze/${id}`);
    }, 500);
  };

  return (
    <div className="min-h-screen bg-background flex flex-col items-center justify-center p-6 sm:p-12">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6 }}
        className="w-full max-w-4xl space-y-8"
      >
        <div className="text-center space-y-4">
          <h1 className="text-4xl sm:text-6xl font-extrabold tracking-tight text-foreground">
            FairFake
          </h1>
          <p className="text-lg sm:text-xl text-muted-foreground max-w-2xl mx-auto">
            Detect deepfakes with high accuracy, and check if the detector is fair across demographic groups.
          </p>
          <div className="pt-4">
             <Button variant="outline" onClick={() => router.push('/dashboard')}>
               View Fairness Dashboard
             </Button>
          </div>
        </div>

        <Tabs defaultValue="upload" className="w-full max-w-2xl mx-auto">
          <TabsList className="grid w-full grid-cols-2">
            <TabsTrigger value="upload">Upload Media</TabsTrigger>
            <TabsTrigger value="demo">Try a Demo</TabsTrigger>
          </TabsList>
          
          <TabsContent value="upload" className="mt-6">
            <Card className="border-dashed border-2 border-border/60 bg-muted/20">
              <CardContent className="flex flex-col items-center justify-center py-20 text-center space-y-6">
                <div className="bg-primary/10 p-4 rounded-full">
                  <UploadCloud className="w-10 h-10 text-primary" />
                </div>
                <div className="space-y-1">
                  <h3 className="font-semibold text-lg">Drag & drop your file here</h3>
                  <p className="text-sm text-muted-foreground">Supports JPG, PNG, MP4, and MOV up to 50MB</p>
                </div>
                
                <div className="relative">
                  <input 
                    type="file" 
                    accept="image/*,video/*"
                    onChange={handleUpload}
                    className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
                    disabled={isUploading}
                  />
                  <Button disabled={isUploading}>
                    {isUploading ? (
                      <>
                        <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                        Analyzing...
                      </>
                    ) : (
                      "Browse Files"
                    )}
                  </Button>
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="demo" className="mt-6">
            <Card>
              <CardHeader>
                <CardTitle>Pre-baked Examples</CardTitle>
                <CardDescription>
                  Try our deepfake detection on these sample files to see the analysis report immediately.
                </CardDescription>
              </CardHeader>
              <CardContent className="grid sm:grid-cols-2 gap-4">
                {[
                  { id: "demo-img-real-1", label: "Real Image", icon: ImageIcon, type: "image" },
                  { id: "demo-img-fake-1", label: "Fake Image", icon: ImageIcon, type: "image" },
                  { id: "demo-vid-real-1", label: "Real Video", icon: Film, type: "video" },
                  { id: "demo-vid-fake-1", label: "Fake Video", icon: Film, type: "video" },
                ].map((demo) => (
                  <Button
                    key={demo.id}
                    variant="outline"
                    className="h-auto py-6 flex flex-col items-center justify-center space-y-2 hover:bg-primary/5 hover:border-primary/50 transition-colors"
                    onClick={() => handleDemoClick(demo.id)}
                    disabled={activeDemo !== null}
                  >
                    {activeDemo === demo.id ? (
                      <Loader2 className="w-6 h-6 animate-spin text-primary" />
                    ) : (
                      <demo.icon className="w-6 h-6 text-muted-foreground" />
                    )}
                    <span className="font-medium">{demo.label}</span>
                  </Button>
                ))}
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      </motion.div>
    </div>
  );
}
