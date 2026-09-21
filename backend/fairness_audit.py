import pandas as pd
import numpy as np
import os

def run_audit(csv_path="data/audit_dataset.csv"):
    if not os.path.exists(csv_path):
        return {"summaries": [], "auditResults": []}
        
    df = pd.read_csv(csv_path)
    
    # Calculate predictions (thresh 0.5)
    df['prediction'] = df['predicted_prob'].apply(lambda x: 'FAKE' if x >= 0.5 else 'REAL')
    df['is_correct'] = df['prediction'] == df['true_label']
    df['is_error'] = ~df['is_correct']
    
    models = df['model'].unique()
    
    attributes = [
        "Wearing_Glasses", "Male", "Female", "Dark_Skin", 
        "Young", "Senior", "Heavy_Makeup", "Facial_Hair", "Blond_Hair", "Bald"
    ]
    
    summaries = []
    audit_results = []
    
    for model in models:
        df_model = df[df['model'] == model]
        
        total_images = len(df_model)
        accuracy = df_model['is_correct'].mean()
        
        # FPR: predicted FAKE but true is REAL
        # FNR: predicted REAL but true is FAKE
        real_mask = df_model['true_label'] == 'REAL'
        fake_mask = df_model['true_label'] == 'FAKE'
        
        fpr = (df_model[real_mask]['prediction'] == 'FAKE').mean() if real_mask.any() else 0.0
        fnr = (df_model[fake_mask]['prediction'] == 'REAL').mean() if fake_mask.any() else 0.0
        
        high_bias_count = 0
        
        for attr in attributes:
            with_attr = df_model[df_model[attr] == 1]
            without_attr = df_model[df_model[attr] == 0]
            
            count_with = len(with_attr)
            count_without = len(without_attr)
            
            if count_with == 0 or count_without == 0:
                continue
                
            err_with = with_attr['is_error'].mean()
            err_without = without_attr['is_error'].mean()
            
            rp = (err_with - err_without) / (err_without + 1e-9)
            crp = abs(rp)
            
            if crp > 0.4:
                severity = "high"
                high_bias_count += 1
            elif crp > 0.2:
                severity = "moderate"
            else:
                severity = "low"
                
            # Grab some example misclassified IDs for the dashboard
            misclassified = with_attr[with_attr['is_error']]
            example_ids = misclassified['id'].head(2).tolist()
            
            audit_results.append({
                "attribute": attr,
                "model": model,
                "errorWithAttribute": float(err_with),
                "errorWithoutAttribute": float(err_without),
                "rp": float(rp),
                "crp": float(crp),
                "severity": severity,
                "sampleCount": {
                    "withAttribute": count_with,
                    "withoutAttribute": count_without
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
