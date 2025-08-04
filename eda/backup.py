import numpy as np
from msclap import CLAP

model = CLAP(model_fp=None, use_cuda=False)

audio = "505321__marcelweiss__s63-amg-v8-engine-revs.wav"

text_prompts = [
    "a dog barking",
    "a car revving",
    "a plane taking off",
    "a thunderstorm",
    "a baby crying"
]

# Load audio and get its embedding
from msclap.utils import load_and_process_audio

# Load and preprocess the audio
waveform, sr = load_and_process_audio(audio)

# Get embedding (must be in a list even if just one)
audio_embed = model.get_audio_embeddings_per_batch([waveform])

# Get text embeddings
text_embeds = model.get_text_embedding(text_prompts)

# Compute cosine similarity
audio_embed = audio_embed / np.linalg.norm(audio_embed, axis=1, keepdims=True)
text_embeds = text_embeds / np.linalg.norm(text_embeds, axis=1, keepdims=True)

# Dot product gives cosine similarity
similarities = audio_embed @ text_embeds.T

for prompt, score in zip(text_prompts, similarities[0]):
    print(f"{prompt:25s} -> Score: {score:.3f}")


# ------------------------------------------------------------------------------


import numpy as np
import torch
import torchaudio
from msclap import CLAP

# Load model
model = CLAP(model_fp=None, use_cuda=torch.cuda.is_available())

# Load audio using torchaudio
audio_path = "../data/505321__marcelweiss__s63-amg-v8-engine-revs.wav"
waveform, sample_rate = torchaudio.load(audio_path)

# Resample if necessary
target_sr = 48000
if sample_rate != target_sr:
    resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sr)
    waveform = resampler(waveform)

# Convert stereo to mono if needed
if waveform.shape[0] > 1:
    waveform = waveform.mean(dim=0, keepdim=True)

# Ensure it's in the right shape: (1, n_samples)
waveform = waveform.squeeze().unsqueeze(0)

# Get audio embedding
audio_embed = model.get_audio_embeddings_per_batch([waveform])

# Text prompts
text_prompts = [
    "a dog barking",
    "a car revving",
    "a plane taking off",
    "a thunderstorm",
    "a baby crying"
]

# Get text embeddings
text_embeds = model.get_text_embedding(text_prompts)

# Normalize
audio_embed = audio_embed / np.linalg.norm(audio_embed, axis=1, keepdims=True)
text_embeds = text_embeds / np.linalg.norm(text_embeds, axis=1, keepdims=True)

# Cosine similarity
similarities = audio_embed @ text_embeds.T

# Show results
for prompt, score in zip(text_prompts, similarities[0]):
    print(f"{prompt:25s} -> Score: {score:.3f}")

# Top match
best_idx = np.argmax(similarities[0])
print(f"\n🎯 Best match: '{text_prompts[best_idx]}' with score {similarities[0][best_idx]:.3f}")



