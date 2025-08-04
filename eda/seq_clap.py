import os
import torch
import pandas as pd
from msclap import CLAP

# Load pretrained CLAP model with 2023 weights
model = CLAP(version='2023', use_cuda=False)

# Set directory where the .wav files are
audio_dir = "../data/Data 1 August 2025/"

# Define your caption prompts
prompt_csv = pd.read_csv("prompts.csv")
captions = prompt_csv.columns.tolist()

"""
captions = [
    "a dog barking",
    "a plane cruising",
    "a train passing",
    "a quiet forest",
    "a car engine revving",
    "birds chirping"
]
"""


# Prepare to store results
results = []

# Loop over all .wav files in the directory
for filename in os.listdir(audio_dir):
    if filename.lower().endswith(".wav"):
        file_path = os.path.join(audio_dir, filename)
        print(f"Processing: {filename}")

        try:
            with torch.no_grad():
                audio_embed = model.get_audio_embeddings([file_path])
                text_embed = model.get_text_embeddings(captions)

                # Normalize
                audio_embed = torch.nn.functional.normalize(audio_embed, dim=-1)
                text_embed = torch.nn.functional.normalize(text_embed, dim=-1)

                similarities = (text_embed @ audio_embed.T).squeeze(1)  # shape: (N,)

            # Save scores for this file
            scores = {
                "file": filename,
                **{caption: score for caption, score in zip(captions, similarities.tolist())}
            }
            results.append(scores)

        except Exception as e:
            print(f"Failed to process {filename}: {e}")

# Save all results to CSV
df = pd.DataFrame(results)
df.to_csv("clap_similarity_scores_full_prompts_2.csv", index=False)
print("Saved results to clap_similarity_scores_full_prompts_2.csv")
