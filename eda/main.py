from msclap import CLAP
import torch

# Load pretrained CLAP model with 2023 weights
model = CLAP(version='2023', use_cuda=False)

# Define your audio and text inputs
audio_path = "../data/Data 18 June 2025/Data 18 June 2025/12.07 ezy68yk.m4a"

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

# Get embeddings
with torch.no_grad():
    audio_embed = model.get_audio_embeddings([audio_path])  # returns shape (1, d)
    text_embed = model.get_text_embeddings(captions)        # returns shape (N, d)

    # Normalize embeddings
    audio_embed = torch.nn.functional.normalize(audio_embed, dim=-1)
    text_embed = torch.nn.functional.normalize(text_embed, dim=-1)

    # Compute cosine similarities
    similarities = (text_embed @ audio_embed.T).squeeze(1)  # shape: (N,)

# Print results
for score, caption in sorted(zip(similarities.tolist(), captions), key=lambda x: -x[0]):
    print(f"{caption}: {score:.4f}")
