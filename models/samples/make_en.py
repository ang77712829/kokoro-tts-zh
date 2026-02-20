# This file is hardcoded to transparently reproduce HEARME_en.wav
# Therefore it may NOT generalize gracefully to other texts
# Refer to Usage in README.md for more general usage patterns

# pip install kokoro>=0.8.1
from kokoro import KModel, KPipeline
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
import tqdm

REPO_ID = 'hexgrad/Kokoro-82M-v1.1-zh'
SAMPLE_RATE = 24000

# How much silence to insert between paragraphs: 5000 is about 0.2 seconds
N_ZEROS = 5000

# Whether to join sentences in paragraphs 1 and 3
JOIN_SENTENCES = True

device = 'cuda' if torch.cuda.is_available() else 'cpu'

texts = [(
"[Kokoro](/kˈQkəɹQ/) is an open-weight series of small but powerful TTS models.",
), (
"This model is the result of a short training run that added 100 Chinese speakers from a professional dataset.",
"The Chinese data was freely and permissively granted to us by LongMaoData, a professional dataset company. Thank you for making this model possible.",
), (
"Separately, some crowdsourced synthetic English data also entered the training mix:",
"1 hour of Maple, an American female.",
"1 hour of [Sol](/sˈOl/), another American female.",
"And 1 hour of Vale, an older British female.",
), (
"This model is not a strict upgrade over its predecessor since it drops many voices, but it is released early to gather feedback on new voices and tokenization.",
"Aside from the Chinese dataset and the 3 hours of English, the rest of the data was left behind for this training run.",
"The goal is to push the model series forward and ultimately restore some of the voices that were left behind.",
), (
"Current guidance from the U.S. Copyright Office indicates that synthetic data generally does not qualify for copyright protection.",
"Since this synthetic data is crowdsourced, the model trainer is not bound by any Terms of Service.",
"This Apache licensed model also aligns with OpenAI's stated mission of broadly distributing the benefits of AI.",
"If you would like to help further that mission, consider contributing permissive audio data to the cause.",
)]

if JOIN_SENTENCES:
    for i in (1, 3):
        texts[i] = [' '.join(texts[i])]

model = KModel(repo_id=REPO_ID).to(device).eval()
en_pipelines = [KPipeline(lang_code='b' if british else 'a', repo_id=REPO_ID, model=model) for british in (False, True)]

path = Path(__file__).parent

wavs = []
for paragraph in tqdm.tqdm(texts):
    for i, sentence in enumerate(paragraph):
        voice, british = 'bf_vale', True
        if 'Maple' in sentence:
            voice, british = 'af_maple', False
        elif 'Sol' in sentence:
            voice, british = 'af_sol', False
        generator = en_pipelines[british](sentence, voice=voice)
        f = path / f'en{len(wavs):02}.wav'
        result = next(generator)
        wav = result.audio
        sf.write(f, wav, SAMPLE_RATE)
        if i == 0 and wavs and N_ZEROS > 0:
            wav = np.concatenate([np.zeros(N_ZEROS), wav])
        wavs.append(wav)

sf.write(path / 'HEARME_en.wav', np.concatenate(wavs), SAMPLE_RATE)
