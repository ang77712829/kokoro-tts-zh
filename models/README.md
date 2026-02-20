---
license: apache-2.0
base_model:
- hexgrad/Kokoro-82M
pipeline_tag: text-to-speech
---
🐈 GitHub: https://github.com/hexgrad/kokoro

<audio controls><source src="https://huggingface.co/hexgrad/Kokoro-82M-v1.1-zh/resolve/main/samples/HEARME_en.wav" type="audio/wav"></audio>

**Kokoro** is an open-weight series of small but powerful TTS models.

This model is the result of a short training run that added 100 Chinese speakers from a professional dataset. The Chinese data was freely and permissively granted to us by [LongMaoData](https://www.longmaosoft.com/), a professional dataset company. Thank you for making this model possible.

Separately, some crowdsourced synthetic English data also entered the training mix:<sup>[1]</sup>
- 1 hour of Maple, an American female.
- 1 hour of Sol, another American female.
- And 1 hour of Vale, an older British female.

This model is not a strict upgrade over its predecessor since it drops many voices, but it is released early to gather feedback on new voices and tokenization. Aside from the Chinese dataset and the 3 hours of English, the rest of the data was left behind for this training run. The goal is to push the model series forward and ultimately restore some of the voices that were left behind.

Current guidance from the U.S. Copyright Office indicates that synthetic data generally does not qualify for copyright protection. Since this synthetic data is crowdsourced, the model trainer is not bound by any Terms of Service. This Apache licensed model also aligns with OpenAI's stated mission of broadly distributing the benefits of AI. If you would like to help further that mission, consider contributing permissive audio data to the cause.

<sup>[1] LongMaoData had no involvement in the crowdsourced synthetic English data.</sup><br/>
<sup>[2] The following Chinese text is machine-translated.</sup>

> Kokoro 是一系列体积虽小但功能强大的 TTS 模型。
>
> 该模型是经过短期训练的结果，从专业数据集中添加了100名中文使用者。中文数据由专业数据集公司「[龙猫数据](https://www.longmaosoft.com/)」免费且无偿地提供给我们。感谢你们让这个模型成为可能。
>
> 另外，一些众包合成英语数据也进入了训练组合：
> - 1小时的 Maple，美国女性。
> - 1小时的 Sol，另一位美国女性。
> - 和1小时的 Vale，一位年长的英国女性。
>
> 由于该模型删除了许多声音，因此它并不是对其前身的严格升级，但它提前发布以收集有关新声音和标记化的反馈。除了中文数据集和3小时的英语之外，其余数据都留在本次训练中。目标是推动模型系列的发展，并最终恢复一些被遗留的声音。
>
> 美国版权局目前的指导表明，合成数据通常不符合版权保护的资格。由于这些合成数据是众包的，因此模型训练师不受任何服务条款的约束。该 Apache 许可模式也符合 OpenAI 所宣称的广泛传播 AI 优势的使命。如果您愿意帮助进一步完成这一使命，请考虑为此贡献许可的音频数据。

<audio controls><source src="https://huggingface.co/hexgrad/Kokoro-82M-v1.1-zh/resolve/main/samples/HEARME_zf_001.wav" type="audio/wav"></audio>

<audio controls><source src="https://huggingface.co/hexgrad/Kokoro-82M-v1.1-zh/resolve/main/samples/HEARME_zm_010.wav" type="audio/wav"></audio>

- [Releases](#releases)
- [Usage](#usage)
- [Samples](https://huggingface.co/hexgrad/Kokoro-82M-v1.1-zh/blob/main/samples) ↗️
- [Model Facts](#model-facts)
- [Acknowledgements](#acknowledgements)

### Releases

| Model | Published | Training Data | Langs & Voices | SHA256 |
| ----- | --------- | ------------- | -------------- | ------ |
| **v1.1-zh** | **2025 Feb 26** | **>100 hours** | **2 & 103** | `b1d8410f` |
| [v1.0](https://huggingface.co/hexgrad/Kokoro-82M) | 2025 Jan 27 | Few hundred hrs | 8 & 54 | `496dba11` |
| [v0.19](https://huggingface.co/hexgrad/kLegacy/tree/main/v0.19) | 2024 Dec 25 | <100 hrs | 1 & 10 | `3b0c392f` |

| Training Costs | v0.19 | v1.0 | v1.1-zh | **Total** |
| -------------- | ----- | ---- | ------- | --------- |
| in A100 80GB GPU hours | 500 | 500 | 120 | **1120** |
| average hourly rate | $0.80/h | $1.20/h | $0.90/h | |
| in USD | $400 | $600 | $110 | **$1110** |

### Usage
You can run this cell on [Google Colab](https://colab.research.google.com/).
```py
!pip install -q kokoro>=0.8.2 "misaki[zh]>=0.8.2" soundfile
!apt-get -qq -y install espeak-ng > /dev/null 2>&1
from IPython.display import display, Audio

!wget https://huggingface.co/hexgrad/Kokoro-82M-v1.1-zh/resolve/main/samples/make_en.py
!python make_en.py
display(Audio('HEARME_en.wav', rate=24000, autoplay=True))

!wget https://huggingface.co/hexgrad/Kokoro-82M-v1.1-zh/resolve/main/samples/make_zh.py
!python make_zh.py
display(Audio('HEARME_zf_001.wav', rate=24000, autoplay=False))
```
TODO: Improve usage. Similar to https://hf.co/hexgrad/Kokoro-82M#usage but you should pass `repo_id='hexgrad/Kokoro-82M-v1.1-zh'` when constructing a `KModel` or `KPipeline`. See [`make_en.py`](https://huggingface.co/hexgrad/Kokoro-82M-v1.1-zh/blob/main/samples/make_en.py) and [`make_zh.py`](https://huggingface.co/hexgrad/Kokoro-82M-v1.1-zh/blob/main/samples/make_zh.py).

### Model Facts

**Architecture:**
- StyleTTS 2: https://arxiv.org/abs/2306.07691
- ISTFTNet: https://arxiv.org/abs/2203.02395
- Decoder only: no diffusion, no encoder release
- 82 million parameters, same as https://hf.co/hexgrad/Kokoro-82M

**Architected by:** Li et al @ https://github.com/yl4579/StyleTTS2

**Trained by**: `@rzvzn` on Discord

**Languages:** English, Chinese

**Model SHA256 Hash:** `b1d8410fa44dfb5c15471fd6c4225ea6b4e9ac7fa03c98e8bea47a9928476e2b`

### Acknowledgements
TODO: Write acknowledgements. Similar to https://hf.co/hexgrad/Kokoro-82M#acknowledgements

<img src="https://static0.gamerantimages.com/wordpress/wp-content/uploads/2024/08/terminator-zero-41-1.jpg" width="400" alt="kokoro" />
