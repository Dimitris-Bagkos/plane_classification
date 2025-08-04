from msclap import CLAP
import torch
import torchaudio

# Load pretrained CLAP model with 2023 weights
model = CLAP(version='2023', use_cuda=False)

# Load the m4a file using torchaudio
audio_path = "../data/Data 18 June 2025/Data 18 June 2025/12.07 ezy68yk.m4a"
waveform, sr = torchaudio.load(audio_path)
"""
# Resample to 48kHz if needed
target_sr = 48000
if sr != target_sr:
    resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
    waveform = resampler(waveform)
"""
# Convert to mono if stereo
waveform = waveform.mean(dim=0, keepdim=True)

# Text captions
captions = [
    "a dog barking",
    "a plane cruising",
    "a train passing",
    "a quiet forest",
    "a car engine revving",
    "birds chirping"
]

# Get embeddings
with torch.no_grad():
    audio_embed = model.get_audio_embeddings(waveform)  # direct waveform instead of file path
    text_embed = model.get_text_embeddings(captions)

    # Normalize
    audio_embed = torch.nn.functional.normalize(audio_embed, dim=-1)
    text_embed = torch.nn.functional.normalize(text_embed, dim=-1)

    # Cosine similarity
    similarities = (text_embed @ audio_embed.T).squeeze(1)

# Print results
for score, caption in sorted(zip(similarities.tolist(), captions), key=lambda x: -x[0]):
    print(f"{caption}: {score:.4f}")
