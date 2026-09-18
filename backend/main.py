import os
import shutil
import tempfile
import cv2
import numpy as np
import torch
import math
import base64
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional

from deepfake_detector.src.model import XceptionDetector
from deepfake_detector.src.dataset import eval_transform

app = FastAPI(title="FairFake Deepfake API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = None

@app.on_event("startup")
def load_model():
    global model
    model_path = "best_model.pth"
    if os.path.exists(model_path):
        model = XceptionDetector(pretrained=False)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        print(f"Loaded deepfake detector from {model_path} on {device}")
    else:
        print(f"Warning: {model_path} not found.")

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)
        
    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate(self, x):
        self.model.eval()
        x.requires_grad_(True)
        self.model.zero_grad()
        out = self.model(x)
        out.backward()
        
        gradients = self.gradients.cpu().data.numpy()[0]
        activations = self.activations.cpu().data.numpy()[0]
        weights = np.mean(gradients, axis=(1, 2))
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        
        for i, w in enumerate(weights):
            cam += w * activations[i, :, :]
            
        cam = np.maximum(cam, 0) 
        if np.max(cam) != 0:
            cam = cam / np.max(cam) 
        return cam, torch.sigmoid(out).item()

def analyze_image_array(img: np.ndarray):
    cam_extractor = GradCAM(model, model.backbone.conv4)
    augmented = eval_transform(image=img)
    img_tensor = augmented['image'].unsqueeze(0).to(device)
    cam, prob = cam_extractor.generate(img_tensor)
    
    cam_resized = cv2.resize(cam, (img.shape[1], img.shape[0]))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img, 0.5, heatmap, 0.5, 0)
    _, buffer = cv2.imencode('.jpg', cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    b64 = base64.b64encode(buffer).decode('utf-8')
    gradcam_b64 = f"data:image/jpeg;base64,{b64}"
    
    return prob, gradcam_b64

def get_mock_attributes():
    import random
    return {
        "gender": random.choice(["Male", "Female"]),
        "ageBracket": random.choice(["Young", "Middle-aged", "Senior"]),
        "skinTone": random.choice(["Light_Skin", "Dark_Skin", "Medium_Skin"]),
        "glasses": random.choice([True, False]),
        "heavyMakeup": random.choice([True, False]),
    }

@app.post("/api/analyze/image")
async def analyze_image(image: UploadFile = File(...)):
    contents = await image.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    prob, heatmap = analyze_image_array(img)
    verdict = "FAKE" if prob > 0.5 else "REAL"
    score = round((1.0 - prob) * 100, 1)

    return {
        "mediaType": "image",
        "total_images_analyzed": 1,
        "image_authenticity_score": score,
        "results": [{
            "fake_probability": prob,
            "verdict": verdict,
            "gradcam_base64": heatmap,
        }],
        "faces": [{
            "boundingBox": [10, 10, 100, 100], 
            "attributes": get_mock_attributes()
        }]
    }

@app.post("/api/analyze/video")
async def analyze_video(video: UploadFile = File(...)):
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    contents = await video.read()
    tmp.write(contents)
    tmp.close()
    
    cap = cv2.VideoCapture(tmp.name)
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_s = total_frames / video_fps
    
    total_secs = min(int(math.ceil(duration_s)), 600)
    NORMAL_FPS = 3
    
    frame_results = []
    all_probs = []
    anomaly_seconds = set()
    
    for s in range(total_secs):
        start = int(s * video_fps)
        end = min(int((s + 1) * video_fps), total_frames - 1)
        if start >= end:
            continue
        step = max(1, (end - start) // NORMAL_FPS)
        indices = list(range(start, end, step))[:NORMAL_FPS]
        
        for fi in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, float(fi))
            ret, frame = cap.read()
            if not ret or frame is None:
                continue
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if frame.shape[0] < 50 or frame.shape[1] < 50:
                continue
            
            prob, heatmap = analyze_image_array(frame)
            all_probs.append(prob)
            if prob > 0.7:
                anomaly_seconds.add(s)
                
            frame_results.append({
                "second": s,
                "frame_index": fi,
                "fake_probability": prob,
                "gradcam_base64": heatmap
            })
            
    cap.release()
    os.unlink(tmp.name)
    
    if not all_probs:
        return {"error": "No frames analyzed"}
        
    max_prob = max(all_probs)
    fake_count = sum(1 for p in all_probs if p > 0.7)
    fake_ratio = fake_count / len(all_probs)
    raw_score = (1.0 - max_prob) * 70 + (1.0 - fake_ratio) * 30
    auth_score = round(max(0.0, min(100.0, raw_score)), 1)
    
    if max_prob >= 0.85:
        verdict = "FAKE"
    elif max_prob >= 0.70 or fake_ratio >= 0.30:
        verdict = "LIKELY FAKE"
    else:
        verdict = "REAL"
        
    frame_results.sort(key=lambda x: x["fake_probability"], reverse=True)
    for i, fr in enumerate(frame_results):
        if i >= 10:
            fr["gradcam_base64"] = "" 
            
    frame_results.sort(key=lambda x: (x["second"], x["frame_index"]))
    
    return {
        "mediaType": "video",
        "duration_seconds": round(duration_s, 2),
        "total_frames_analyzed": len(all_probs),
        "anomaly_seconds": sorted(list(anomaly_seconds)),
        "max_fake_probability": round(max_prob, 4),
        "video_authenticity_score": auth_score,
        "verdict": verdict,
        "frame_results": frame_results,
        "faces": [{
            "boundingBox": [10, 10, 100, 100],
            "attributes": get_mock_attributes()
        }]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
