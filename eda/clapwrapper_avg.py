import torch
import pandas as pd
from msclap import CLAP

# Load pretrained CLAP model
model = CLAP(version='2023', use_cuda=False)

# Input audio + captions
audio_path = "../data/505321__marcelweiss__s63-amg-v8-engine-revs.wav"
captions = [
    "a dog barking",
    "a jet taking off",
    "a train passing",
    "a quiet forest",
    "a car engine revving"
]

# Dict to collect all similarity scores
scores_dict = {caption: [] for caption in captions}

# Number of times to run the classification task
iterations = 100

# Main loop
for i in range(iterations):
    with torch.no_grad():
        audio_embed = model.get_audio_embeddings([audio_path])
        text_embed = model.get_text_embeddings(captions)

        # Normalize and compute similarities
        audio_embed = torch.nn.functional.normalize(audio_embed, dim=-1)
        text_embed = torch.nn.functional.normalize(text_embed, dim=-1)
        similarities = (text_embed @ audio_embed.T).squeeze(1)

        for caption, score in zip(captions, similarities):
            scores_dict[caption].append(round(score.item(), 6))

    print("Finished", i+1, "iteration(s)")

# Save
df = pd.DataFrame(scores_dict)
df.index.name = "iteration"
df.to_csv("clapwrapper_similarity_scores.csv")

print("Saved similarity variations to clapwrapper_similarity_scores.csv")
