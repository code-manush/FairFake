import pandas as pd
import numpy as np
import random
import os

def generate_mock_dataset(num_rows=12040, model_name="Xception"):
    data = []
    
    for i in range(num_rows):
        # Base attributes
        true_label = random.choice(["REAL", "FAKE"])
        
        # Demographics
        is_male = random.random() > 0.5
        is_female = not is_male
        
        # Skin Tone (biased against Dark_Skin)
        skin_r = random.random()
        dark_skin = skin_r < 0.15
        
        # Age
        age_r = random.random()
        young = age_r < 0.4
        senior = age_r > 0.85
        
        # Other
        heavy_makeup = random.random() < 0.2 if is_female else random.random() < 0.02
        wearing_glasses = random.random() < 0.15
        facial_hair = random.random() < 0.3 if is_male else 0.0
        blond_hair = random.random() < 0.15
        bald = random.random() < 0.1 if is_male else 0.0

        # Base error probability (baseline accuracy ~ 90%)
        error_prob = 0.10
        
        if model_name == "Xception":
            # Inject biases for Xception
            if dark_skin: error_prob += 0.25      # High bias
            if heavy_makeup: error_prob += 0.20   # High bias
            if bald: error_prob += 0.22           # High bias
            if wearing_glasses: error_prob += 0.15 # Moderate bias
            if senior: error_prob += 0.12         # Moderate bias
        elif model_name == "EfficientNetB0":
            # Less bias for EfficientNetB0
            if dark_skin: error_prob += 0.12
            if heavy_makeup: error_prob += 0.10
            if bald: error_prob += 0.08
            
        error_prob = min(0.99, error_prob)
        
        is_error = random.random() < error_prob
        
        if true_label == "REAL":
            # If error, predict FAKE (prob > 0.5)
            predicted_prob = random.uniform(0.55, 0.99) if is_error else random.uniform(0.01, 0.45)
        else:
            # true == FAKE
            # If error, predict REAL (prob < 0.5)
            predicted_prob = random.uniform(0.01, 0.45) if is_error else random.uniform(0.55, 0.99)

        data.append({
            "id": f"img_{model_name}_{i}",
            "model": model_name,
            "true_label": true_label,
            "predicted_prob": round(predicted_prob, 4),
            "Wearing_Glasses": int(wearing_glasses),
            "Male": int(is_male),
            "Female": int(is_female),
            "Dark_Skin": int(dark_skin),
            "Young": int(young),
            "Senior": int(senior),
            "Heavy_Makeup": int(heavy_makeup),
            "Facial_Hair": int(facial_hair),
            "Blond_Hair": int(blond_hair),
            "Bald": int(bald)
        })
        
    return pd.DataFrame(data)

if __name__ == "__main__":
    print("Generating dataset...")
    df_xception = generate_mock_dataset(12040, "Xception")
    df_efficientnet = generate_mock_dataset(12040, "EfficientNetB0")
    
    df_all = pd.concat([df_xception, df_efficientnet])
    
    os.makedirs("data", exist_ok=True)
    df_all.to_csv("data/audit_dataset.csv", index=False)
    print("Dataset generated at data/audit_dataset.csv with", len(df_all), "rows.")
