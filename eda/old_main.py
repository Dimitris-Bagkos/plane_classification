# In this version, the wrong clap model was used (instead of using the intended CLAPWrapper, the barebones CLAP model
# was used). That meant randomised weights were used and the model was no good in classifying the sounds

from msclap.models.clap import CLAP  # your clap.py file
import torch

# Define parameters
params = {
    'audioenc_name': "Cnn14",
    'sample_rate': 48000,
    'window_size': 1024,
    'hop_size': 320,
    'mel_bins': 64,
    'fmin': 50,
    'fmax': 14000,
    'classes_num': 527,            # same as AudioSet classes, unused in inference
    'out_emb': 2048,               # Cnn14 outputs 2048
    'text_model': "bert-base-uncased",
    'transformer_embed_dim': 768,
    'd_proj': 512,                 # projection dim for both audio and text
}

model = CLAP(**params)
model.eval()

import torchaudio

waveform, sr = torchaudio.load("./data/505321__marcelweiss__s63-amg-v8-engine-revs.wav")

# Resample if needed
if sr != params['sample_rate']:
    waveform = torchaudio.transforms.Resample(sr, params['sample_rate'])(waveform)

# CLAP expects shape: (batch_size, data_len)
waveform = waveform.mean(dim=0, keepdim=True)  # mono

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(params["text_model"])
captions = [
    "a dog barking",
    "a jet taking off",
    "a train passing",
    "a quiet forest",
    "a car engine revving"
]

text_inputs = tokenizer(
    captions,
    padding=True,
    truncation=True,
    return_tensors="pt"
)

with torch.no_grad():
    text_embed, audio_embed, scale = model(waveform, text_inputs)

    # Normalize
    audio_embed = torch.nn.functional.normalize(audio_embed, dim=-1)
    text_embed = torch.nn.functional.normalize(text_embed, dim=-1)

    similarities = scale * text_embed @ audio_embed.T
    similarities = similarities.squeeze(1)

for score, caption in sorted(zip(similarities.tolist(), captions), key=lambda x: -x[0]):
    print(f"{caption}: {score:.4f}")

