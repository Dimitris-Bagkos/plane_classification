import time
import torch
import torchaudio
from transformers import AutoTokenizer
from msclap.models.clap import CLAP
import pandas as pd

# Static parameters
params = {
    'audioenc_name': "Cnn14",
    'sample_rate': 48000,
    'window_size': 1024,
    'hop_size': 320,
    'mel_bins': 64,
    'fmin': 50,
    'fmax': 14000,
    'classes_num': 527,
    'out_emb': 2048,
    'text_model': "bert-base-uncased",
    'transformer_embed_dim': 768,
    'd_proj': 512,
}

captions = [
    "a dog barking",
    "a jet taking off",
    "a train passing",
    "a quiet forest",
    "a car engine revving"
]

# Load static assets outside loop
waveform, sr = torchaudio.load("./data/505321__marcelweiss__s63-amg-v8-engine-revs.wav")
if sr != params['sample_rate']:
    waveform = torchaudio.transforms.Resample(sr, params['sample_rate'])(waveform)
waveform = waveform.mean(dim=0, keepdim=True)

tokenizer = AutoTokenizer.from_pretrained(params["text_model"])
text_inputs = tokenizer(
    captions,
    padding=True,
    truncation=True,
    return_tensors="pt"
)

# Store results
scores_dict = {caption: [] for caption in captions}
num_iterations = 1000

start_time = time.time()

for i in range(num_iterations):
    model = CLAP(**params)
    model.eval()

    with torch.no_grad():
        text_embed, audio_embed, scale = model(waveform, text_inputs)

        audio_embed = torch.nn.functional.normalize(audio_embed, dim=-1)
        text_embed = torch.nn.functional.normalize(text_embed, dim=-1)

        similarities = (scale * text_embed @ audio_embed.T).squeeze(1)

        for caption, score in zip(captions, similarities.tolist()):
            scores_dict[caption].append(score)

    print("Finished", i, "iteration(s).")

end_time = time.time()
elapsed_time = end_time - start_time
print("Elapsed time:", elapsed_time)

# Save
df = pd.DataFrame(scores_dict)
df.index.name = "iteration"
df.to_csv("clap_similarity_scores.csv")

print("Saved similarity variations to clap_similarity_variability.csv")


