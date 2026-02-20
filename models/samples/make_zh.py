# This file is hardcoded to transparently reproduce HEARME_zh.wav
# Therefore it may NOT generalize gracefully to other texts
# Refer to Usage in README.md for more general usage patterns

# pip install kokoro>=0.8.1 "misaki[zh]>=0.8.1"
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

VOICE = 'zf_001' if True else 'zm_010'

device = 'cuda' if torch.cuda.is_available() else 'cpu'

texts = [(
"Kokoro 是一系列体积虽小但功能强大的 TTS 模型。",
), (
"该模型是经过短期训练的结果，从专业数据集中添加了100名中文使用者。",
"中文数据由专业数据集公司「龙猫数据」免费且无偿地提供给我们。感谢你们让这个模型成为可能。",
), (
"另外，一些众包合成英语数据也进入了训练组合：",
"1小时的 Maple，美国女性。",
"1小时的 Sol，另一位美国女性。",
"和1小时的 Vale，一位年长的英国女性。",
), (
"由于该模型删除了许多声音，因此它并不是对其前身的严格升级，但它提前发布以收集有关新声音和标记化的反馈。",
"除了中文数据集和3小时的英语之外，其余数据都留在本次训练中。",
"目标是推动模型系列的发展，并最终恢复一些被遗留的声音。",
), (
"美国版权局目前的指导表明，合成数据通常不符合版权保护的资格。",
"由于这些合成数据是众包的，因此模型训练师不受任何服务条款的约束。",
"该 Apache 许可模式也符合 OpenAI 所宣称的广泛传播 AI 优势的使命。",
"如果您愿意帮助进一步完成这一使命，请考虑为此贡献许可的音频数据。",
)]

if JOIN_SENTENCES:
    for i in (1, 3):
        texts[i] = [''.join(texts[i])]

en_pipeline = KPipeline(lang_code='a', repo_id=REPO_ID, model=False)
def en_callable(text):
    if text == 'Kokoro':
        return 'kˈOkəɹO'
    elif text == 'Sol':
        return 'sˈOl'
    return next(en_pipeline(text)).phonemes

# HACK: Mitigate rushing caused by lack of training data beyond ~100 tokens
# Simple piecewise linear fn that decreases speed as len_ps increases
def speed_callable(len_ps):
    speed = 0.8
    if len_ps <= 83:
        speed = 1
    elif len_ps < 183:
        speed = 1 - (len_ps - 83) / 500
    return speed * 1.1

model = KModel(repo_id=REPO_ID).to(device).eval()
zh_pipeline = KPipeline(lang_code='z', repo_id=REPO_ID, model=model, en_callable=en_callable)

path = Path(__file__).parent

wavs = []
for paragraph in tqdm.tqdm(texts):
    for i, sentence in enumerate(paragraph):
        generator = zh_pipeline(sentence, voice=VOICE, speed=speed_callable)
        f = path / f'zh{len(wavs):02}.wav'
        result = next(generator)
        wav = result.audio
        sf.write(f, wav, SAMPLE_RATE)
        if i == 0 and wavs and N_ZEROS > 0:
            wav = np.concatenate([np.zeros(N_ZEROS), wav])
        wavs.append(wav)

sf.write(path / f'HEARME_{VOICE}.wav', np.concatenate(wavs), SAMPLE_RATE)
