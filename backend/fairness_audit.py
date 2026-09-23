import pandas as pd
import numpy as np
import os
import random

ATTRIBUTES = [
    "Male", "Female", "Young", "Senior", "Asian", "White", "Black",
    "Shiny_Skin", "Bald", "Wavy_Hair", "Receding_Hairline", "Bangs", 
    "Black_Hair", "Blond_Hair", "Brown_Hair", "No_Beard", "Mustache", 
    "Goatee", "Oval_Face", "Square_Face", "Double_Chin", "Chubby", 
    "Obstructed_Forehead", "Fully_Visible_Forehead", "Mouth_Closed", 
    "Smiling", "Big_Lips", "Big_Nose", "Pointy_Nose", "Heavy_Makeup", 
    "Wearing_Hat", "Wearing_Lipstick", "Eyeglasses", "Attractive"
]

def calculate_rp_and_crp(df_subset, attr):
    """Calculates RP and CRP using control groups as per Xu et al."""
    with_attr = df_subset[df_subset[attr] == 1]
    without_attr = df_subset[df_subset[attr] == 0]
    
    count_with = len(with_attr)
    count_without = len(without_attr)
    
    if count_with == 0 or count_without == 0:
        return 0.0, 0.0, count_with, count_without, 0.0, 0.0
        
    err_with = with_attr['is_error'].mean()
    err_without = without_attr['is_error'].mean()
    
    def safe_rp(e_w, e_wo):
        if e_wo == 0:
            return 0.0 if e_w == 0 else 5.0
        return max(-5.0, min(5.0, (e_w / e_wo) - 1.0))
    
    # RP(a) modified to show positive bias for higher errors, safely clamped:
    rp_data = safe_rp(err_with, err_without)
    
    # Control Group Sampling
    N = min(count_with, count_without)
    if N == 0:
        return rp_data, rp_data, count_with, count_without, float(err_with), float(err_without)
        
    control_with = with_attr.sample(n=N, random_state=42)
    control_without = without_attr.sample(n=N, random_state=42)
    
    err_control_with = control_with['is_error'].mean()
    err_control_without = control_without['is_error'].mean()
    rp_control = safe_rp(err_control_with, err_control_without)
    
    # Paper definition: CRP(a) = RP_data(a) - RP_control(a)
    crp = rp_data - rp_control
    
    return float(rp_data), float(crp), count_with, count_without, float(err_with), float(err_without)


def run_audit(csv_path="data/audit_dataset.csv"):
    if not os.path.exists(csv_path):
        return {"summaries": [], "auditResults": []}
        
    df = pd.read_csv(csv_path)
    
    # Calculate predictions (thresh 0.5)
    df['prediction'] = df['predicted_prob'].apply(lambda x: 'FAKE' if x >= 0.5 else 'REAL')
    df['is_correct'] = df['prediction'] == df['true_label']
    df['is_error'] = ~df['is_correct']
    
    models = df['model'].unique()
    
    # We will compute overall CRP, PDRP (pristine only), DDRP (fake only)
    summaries = []
    audit_results = []
    
    for model in models:
        df_model = df[df['model'] == model]
        
        total_images = len(df_model)
        accuracy = df_model['is_correct'].mean()
        
        real_mask = df_model['true_label'] == 'REAL'
        fake_mask = df_model['true_label'] == 'FAKE'
        
        df_pristine = df_model[real_mask]
        df_fake = df_model[fake_mask]
        
        fpr = (df_pristine['prediction'] == 'FAKE').mean() if real_mask.any() else 0.0
        fnr = (df_fake['prediction'] == 'REAL').mean() if fake_mask.any() else 0.0
        
        high_bias_count = 0
        
        # We need to make sure ATTRIBUTES only checks columns that exist
        available_attrs = [c for c in ATTRIBUTES if c in df_model.columns]
        
        for attr in available_attrs:
            # Overall CRP
            rp_overall, crp_overall, c_with, c_without, err_with, err_without = calculate_rp_and_crp(df_model, attr)
            
            # PDRP (Pristine Data Relative Performance)
            rp_pristine, pdrp, _, _, _, _ = calculate_rp_and_crp(df_pristine, attr)
            
            # DDRP (Deepfake Data Relative Performance)
            rp_fake, ddrp, _, _, _, _ = calculate_rp_and_crp(df_fake, attr)
            
            # Use abs(crp_overall) to determine severity for simplicity
            if abs(crp_overall) > 0.4:
                severity = "high"
                high_bias_count += 1
            elif abs(crp_overall) > 0.2:
                severity = "moderate"
            else:
                severity = "low"
                
            # Grab some example misclassified IDs for the dashboard
            misclassified = df_model[(df_model[attr] == 1) & df_model['is_error']]
            example_ids = misclassified['id'].head(2).tolist()
            
            audit_results.append({
                "attribute": attr,
                "model": model,
                "errorWithAttribute": err_with,
                "errorWithoutAttribute": err_without,
                "rp": rp_overall,
                "crp": crp_overall,
                "pdrp": pdrp,
                "ddrp": ddrp,
                "severity": severity,
                "sampleCount": {
                    "withAttribute": c_with,
                    "withoutAttribute": c_without
                },
                "exampleMisclassifiedIds": example_ids
            })
            
        summaries.append({
            "model": model,
            "totalImages": total_images,
            "overallAccuracy": float(accuracy),
            "overallFPR": float(fpr),
            "overallFNR": float(fnr),
            "highBiasAttributeCount": high_bias_count
        })
        
    return {
        "summaries": summaries,
        "auditResults": audit_results
    }

if __name__ == "__main__":
    res = run_audit()
    print(res["summaries"])
