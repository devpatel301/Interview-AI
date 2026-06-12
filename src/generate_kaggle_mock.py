import pandas as pd
import numpy as np
import os

def create_synthetic_kaggle_dataset():
    """
    Generates a synthetic dataset mimicking a Kaggle Interview Performance dataset.
    This replaces the need for an API key while allowing us to train a real model.
    """
    np.random.seed(42)
    n_samples = 5000
    
    # Generate realistic features
    eye_contact = np.random.normal(loc=0.65, scale=0.15, size=n_samples)
    posture = np.random.normal(loc=0.70, scale=0.15, size=n_samples)
    speaking_pace = np.random.normal(loc=130, scale=30, size=n_samples)
    pause_ratio = np.random.normal(loc=0.25, scale=0.10, size=n_samples)
    
    # Clip to valid ranges
    eye_contact = np.clip(eye_contact, 0.0, 1.0)
    posture = np.clip(posture, 0.0, 1.0)
    speaking_pace = np.clip(speaking_pace, 80, 200)
    pause_ratio = np.clip(pause_ratio, 0.0, 0.6)
    
    # True confidence function (what the model needs to learn)
    # A mix of high eye contact, good posture, pace around 140-160, and low pauses.
    pace_score = 1.0 - (np.abs(speaking_pace - 150) / 70)
    pace_score = np.clip(pace_score, 0, 1)
    
    pause_score = 1.0 - (pause_ratio / 0.6)
    pause_score = np.clip(pause_score, 0, 1)
    
    confidence = (
        0.35 * eye_contact + 
        0.35 * posture + 
        0.15 * pace_score + 
        0.15 * pause_score + 
        np.random.normal(0, 0.05, n_samples) # Add some noise
    )
    confidence = np.clip(confidence, 0.0, 1.0)
    
    df = pd.DataFrame({
        "eye_contact_avg": eye_contact,
        "posture_avg": posture,
        "speaking_pace": speaking_pace,
        "pause_ratio": pause_ratio,
        "confidence": confidence
    })
    
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/kaggle_interview_dataset.csv", index=False)
    print("Created synthetic Kaggle dataset at data/kaggle_interview_dataset.csv")

if __name__ == "__main__":
    create_synthetic_kaggle_dataset()
