from fastapi import FastAPI, Request, Form, Body
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import os
import numpy as np
import soundfile as sf
from io import BytesIO
from pathlib import Path
import torch
from typing import Optional, Dict, Any

# 猴子补丁：修改hf_hub_download函数，使其返回本地文件路径
from huggingface_hub import file_download
original_hf_hub_download = file_download.hf_hub_download

def local_hf_hub_download(repo_id, filename, **kwargs):
    # 直接返回本地文件路径
    file_path = Path(repo_id) / filename
    if file_path.exists():
        return str(file_path)
    else:
        raise FileNotFoundError(f"File not found: {file_path}")

file_download.hf_hub_download = local_hf_hub_download

# 现在导入kokoro
from kokoro import KModel, KPipeline

# 获取项目根目录
import os
from pathlib import Path
# 获取项目根目录
# 尝试多个可能的路径来找到models目录
potential_paths = [
    # 当前工作目录
    Path.cwd() / "models",
    # 当前文件的父目录的父目录
    Path(__file__).parent.parent / "models",
    # 当前文件的父目录的父目录的父目录
    Path(__file__).parent.parent.parent / "models",
    # 硬编码的项目根目录路径（相对于脚本位置）
    Path(__file__).resolve().parent.parent.parent / "models",
    # Docker容器中的路径
    Path("/app/models")
]

# 找到第一个存在的models目录
SHARED_MODELS_DIR = None
for path in potential_paths:
    if path.exists():
        SHARED_MODELS_DIR = path
        break

# 如果还是找不到，使用当前工作目录下的models
if SHARED_MODELS_DIR is None:
    SHARED_MODELS_DIR = Path.cwd() / "models"

# 打印找到的模型路径
print(f"模型路径搜索结果:", flush=True)
for path in potential_paths:
    print(f"{path}: {'存在' if path.exists() else '不存在'}", flush=True)
print(f"最终使用的模型路径: {SHARED_MODELS_DIR}", flush=True)

# 猴子补丁：修改KModel.MODEL_NAMES字典，添加本地模型路径
KModel.MODEL_NAMES[str(SHARED_MODELS_DIR)] = 'kokoro-v1_1-zh.pth'

app = FastAPI()

# 设置静态文件目录和模板目录
app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")

# 模型路径
MODEL_PATH = str(SHARED_MODELS_DIR)
SAMPLE_RATE = 24000

# 加载模型
print("=======================================", flush=True)
print("GPU 检测信息:", flush=True)
print(f"PyTorch 版本: {torch.__version__}", flush=True)
print(f"CUDA 可用: {torch.cuda.is_available()}", flush=True)
print(f"CUDA 版本: {torch.version.cuda}", flush=True)
print(f"GPU 设备数: {torch.cuda.device_count()}", flush=True)
if torch.cuda.is_available():
    print(f"GPU 名称: {torch.cuda.get_device_name(0)}", flush=True)
    print(f"GPU 内存总量: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB", flush=True)
    print(f"当前 GPU 内存使用: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB", flush=True)
else:
    print("没有检测到可用的 GPU，将使用 CPU 进行推理", flush=True)
print("=======================================", flush=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'  # 优先使用GPU推理
print(f"正在加载模型到 {device}...", flush=True)
model = KModel(repo_id=MODEL_PATH)
model = model.to(device).eval()
print(f"模型加载成功，使用{device}进行推理", flush=True)
print(f"模型路径: {MODEL_PATH}", flush=True)
if device == 'cuda':
    print(f"模型加载后 GPU 内存使用: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB", flush=True)

# 创建英文pipeline
en_pipeline = KPipeline(lang_code='a', repo_id=MODEL_PATH, model=False)

# 定义英文发音处理函数
def en_callable(text):
    if text == 'Kokoro':
        return 'kˈOkəɹO'
    elif text == 'Sol':
        return 'sˈOl'
    try:
        return next(en_pipeline(text)).phonemes
    except Exception as e:
        print(f"en_callable处理失败: {e}")
        return text

# 创建中文pipeline，添加en_callable参数
zh_pipeline = KPipeline(lang_code='z', repo_id=MODEL_PATH, model=model, en_callable=en_callable)

# 获取可用的声音列表
def get_available_voices():
    voices_dir = Path(MODEL_PATH) / "voices"
    if voices_dir.exists():
        return [f.stem for f in voices_dir.glob("*.pt")]
    return []

# 速度调整函数
def speed_callable(len_ps):
    speed = 0.8
    if len_ps <= 83:
        speed = 1
    elif len_ps < 183:
        speed = 1 - (len_ps - 83) / 500
    return speed * 1.1

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    voices = get_available_voices()
    return templates.TemplateResponse("index.html", {"request": request, "voices": voices})

@app.post("/api/tts")
async def tts_post(
    request: Request,
    text: Optional[str] = Form(None),
    voice: Optional[str] = Form(None),
    speed: float = Form(1.0)
):
    # 检查是否是JSON请求
    content_type = request.headers.get("Content-Type", "")
    print(f"Content-Type: {content_type}")
    
    if "application/json" in content_type:
        # 解析JSON请求体
        try:
            json_data = await request.json()
            # 尝试打印JSON请求体，处理编码错误
            try:
                print(f"JSON请求体: {json_data}")
            except UnicodeEncodeError as e:
                print(f"JSON请求体包含无法编码的字符，错误: {e}")
                # 尝试使用其他编码方式打印
                try:
                    import json
                    print(f"JSON请求体: {json.dumps(json_data, ensure_ascii=False)}")
                except Exception as e2:
                    print(f"无法打印JSON请求体: {e2}")
            
            # 尝试从不同的参数名称获取值
            text = json_data.get("text") or json_data.get("input") or json_data.get("prompt")
            voice = json_data.get("voice") or json_data.get("speaker") or json_data.get("character")
            speed = json_data.get("speed", json_data.get("rate", 1.0))
            
            # 尝试打印解析后的参数，处理编码错误
            try:
                print(f"解析后的参数 - text: {text}, voice: {voice}, speed: {speed}")
            except UnicodeEncodeError as e:
                print(f"解析后的参数包含无法编码的字符，错误: {e}")
                # 只打印参数类型和长度
                print(f"解析后的参数 - text类型: {type(text)}, text长度: {len(text) if text else 0}, voice: {voice}, speed: {speed}")
        except Exception as e:
            print(f"解析JSON请求体失败: {e}")
            from fastapi.responses import JSONResponse
            return JSONResponse(status_code=400, content={"error": f"Failed to parse JSON request body: {str(e)}"})
    else:
        try:
            print(f"Form请求参数 - text: {text}, voice: {voice}, speed: {speed}")
        except UnicodeEncodeError as e:
            print(f"Form请求参数包含无法编码的字符，错误: {e}")
            print(f"Form请求参数 - text类型: {type(text)}, text长度: {len(text) if text else 0}, voice: {voice}, speed: {speed}")
    
    # 验证必要参数
    if not text or not voice:
        from fastapi.responses import JSONResponse
        try:
            print(f"缺少必要参数 - text: {text}, voice: {voice}")
        except UnicodeEncodeError as e:
            print(f"缺少必要参数，错误: {e}")
        return JSONResponse(status_code=400, content={"error": "Missing required parameters: text and voice"})
    
    return await process_tts(text, voice, speed)

@app.get("/api/tts")
async def tts_get(text: str, voice: str = "zf_001", speed: float = 1.0):
    return await process_tts(text, voice, speed)

@app.get("/api/tts/tts")
async def tts_get_tts(text: str, character: str = None, voice: str = "zf_001", speed: float = 1.0, emotion: str = "default"):
    # 解析character参数
    if character:
        # 解码URL编码的字符串
        import urllib.parse
        character_decoded = urllib.parse.unquote(character)
        # 解析键值对
        for param in character_decoded.split('&'):
            if '=' in param:
                key, value = param.split('=', 1)
                if key == 'voice':
                    voice = value
                elif key == 'speed':
                    try:
                        speed = float(value)
                    except ValueError:
                        pass
    return await process_tts(text, voice, speed)

async def process_tts(text: str, voice: str, speed: float):
    # 清理文本，移除多余的换行符和空格，并处理特殊字符
    try:
        # 先清理文本中的特殊字符，避免后续处理出错
        text = ''.join([c if c.isprintable() or c.isspace() else ' ' for c in text])
        text = text.strip().replace('\n', ' ').replace('\r', ' ').replace('  ', ' ')
        # 尝试打印清理后的文本信息，处理编码错误
        try:
            print(f"清理后的文本长度: {len(text)}, 语速: {speed}, 声音: {voice}", flush=True)
        except UnicodeEncodeError as e:
            print(f"打印清理后的文本信息时遇到编码错误: {e}", flush=True)
            # 只打印基本信息
            print(f"清理后的文本长度: {len(text)}, 语速: {speed}, 声音: {voice}", flush=True)
    except Exception as e:
        print(f"清理文本时遇到错误: {e}", flush=True)
        # 使用更简单的文本处理方式
        text = text.strip()
    
    # 生成语音
    try:
        # 检测文本是否主要为英文
        import re
        english_chars = len(re.findall(r'[a-zA-Z]', text))
        total_chars = len(text)
        is_english = english_chars / total_chars > 0.6 if total_chars > 0 else False
        print(f"英文字符数: {english_chars}, 总字符数: {total_chars}, 是否为英文: {is_english}", flush=True)
        
        # 分割长文本，每段不超过100个字符
        max_segment_length = 100
        segments = []
        current_segment = ""
        
        # 按标点符号分割文本
        punctuation = ["。", "，", "！", "？", ".", ",", "!", "?"]
        
        for char in text:
            current_segment += char
            if len(current_segment) >= max_segment_length and char in punctuation:
                segments.append(current_segment)
                current_segment = ""
        
        if current_segment:
            segments.append(current_segment)
        
        print(f"文本分割为 {len(segments)} 段", flush=True)
        
        # 处理每段文本并拼接音频
        all_wavs = []
        
        for i, segment in enumerate(segments):
            # 先清理段落中的特殊字符，避免打印和处理时出错
            segment = ''.join([c if c.isprintable() or c.isspace() else ' ' for c in segment])
            # 尝试打印处理的段落，处理编码错误
            try:
                print(f"处理第 {i+1} 段: {segment}", flush=True)
            except UnicodeEncodeError as e:
                print(f"处理第 {i+1} 段时遇到编码错误: {e}", flush=True)
                # 只打印段落长度
                print(f"处理第 {i+1} 段，长度: {len(segment)}", flush=True)
            
            try:
                if is_english:
                    # 如果主要为英文，使用英文pipeline
                    print(f"使用英文pipeline处理", flush=True)
                    # 定义英文pipeline的速度函数
                    def en_speed_callable(len_ps):
                        return speed
                    
                    try:
                        generator = en_pipeline(segment, voice=voice, speed=en_speed_callable)
                        result = next(generator)
                        wav_segment = result.audio
                        print(f"英文pipeline处理成功，音频数据类型: {type(wav_segment)}", flush=True)
                        
                        # 如果英文pipeline返回None，尝试使用中文pipeline
                        if wav_segment is None:
                            print(f"英文pipeline返回None，尝试使用中文pipeline", flush=True)
                            def zh_speed_callable(len_ps):
                                return speed
                            
                            generator = zh_pipeline(segment, voice=voice, speed=zh_speed_callable)
                            result = next(generator)
                            wav_segment = result.audio
                            print(f"中文pipeline处理成功，音频数据类型: {type(wav_segment)}", flush=True)
                    except Exception as e:
                        print(f"英文pipeline处理失败: {e}", flush=True)
                        # 尝试使用中文pipeline
                        print(f"尝试使用中文pipeline处理", flush=True)
                        def zh_speed_callable(len_ps):
                            return speed
                        
                        generator = zh_pipeline(segment, voice=voice, speed=zh_speed_callable)
                        result = next(generator)
                        wav_segment = result.audio
                        print(f"中文pipeline处理成功，音频数据类型: {type(wav_segment)}", flush=True)
                else:
                    # 否则使用中文pipeline
                    print(f"使用中文pipeline处理", flush=True)
                    # 定义中文pipeline的速度函数
                    def zh_speed_callable(len_ps):
                        return speed
                    
                    generator = zh_pipeline(segment, voice=voice, speed=zh_speed_callable)
                    result = next(generator)
                    wav_segment = result.audio
                    print(f"中文pipeline处理成功，音频数据类型: {type(wav_segment)}", flush=True)
                
                # 确保wav_segment是有效的
                if wav_segment is not None:
                    # 处理不同类型的音频数据
                    if isinstance(wav_segment, torch.Tensor):
                        # 将PyTorch张量转换为NumPy数组
                        wav_segment = wav_segment.cpu().numpy()
                        print(f"转换PyTorch张量为NumPy数组，形状: {wav_segment.shape}", flush=True)
                    elif isinstance(wav_segment, np.ndarray):
                        print(f"音频数据形状: {wav_segment.shape}", flush=True)
                    else:
                        print(f"未知音频数据类型: {type(wav_segment)}", flush=True)
                        continue
                    
                    # 确保音频数据是二维的
                    if len(wav_segment.shape) == 1:
                        wav_segment = np.expand_dims(wav_segment, axis=1)
                        print(f"调整音频数据形状为: {wav_segment.shape}", flush=True)
                    
                    all_wavs.append(wav_segment)
            except Exception as e:
                print(f"处理第 {i+1} 段失败: {e}", flush=True)
                # 尝试使用中文pipeline处理
                try:
                    print(f"尝试使用中文pipeline处理英文文本", flush=True)
                    def zh_speed_callable(len_ps):
                        return speed
                    
                    generator = zh_pipeline(segment, voice=voice, speed=zh_speed_callable)
                    result = next(generator)
                    wav_segment = result.audio
                    
                    if wav_segment is not None:
                        if isinstance(wav_segment, torch.Tensor):
                            wav_segment = wav_segment.cpu().numpy()
                        if len(wav_segment.shape) == 1:
                            wav_segment = np.expand_dims(wav_segment, axis=1)
                        all_wavs.append(wav_segment)
                    print(f"中文pipeline处理英文文本成功", flush=True)
                except Exception as e2:
                    print(f"中文pipeline处理英文文本也失败: {e2}", flush=True)
        
        # 拼接所有音频段
        if not all_wavs:
            print("没有生成有效的音频数据", flush=True)
            from fastapi.responses import JSONResponse
            return JSONResponse(status_code=400, content={"error": "生成的音频数据无效"})
        
        # 拼接音频段
        wav = np.concatenate(all_wavs)
        print(f"拼接后的音频数据形状: {wav.shape}", flush=True)
        
    except Exception as e:
        print(f"处理文本失败: {e}", flush=True)
        import traceback
        traceback.print_exc()
        # 如果处理失败，返回错误
        from fastapi.responses import JSONResponse
        return JSONResponse(status_code=400, content={"error": f"处理文本失败: {str(e)}"})
    
    # 确保wav是有效的
    if wav is None:
        from fastapi.responses import JSONResponse
        return JSONResponse(status_code=400, content={"error": "生成的音频数据无效"})
    
    # 处理音频数据形状
    if isinstance(wav, torch.Tensor):
        # 将PyTorch张量转换为NumPy数组
        wav = wav.cpu().numpy()
        print(f"将PyTorch张量转换为NumPy数组，形状: {wav.shape}", flush=True)
    
    if isinstance(wav, np.ndarray):
        # 如果是一维数组，添加一个维度
        if len(wav.shape) == 1:
            wav = np.expand_dims(wav, axis=1)
            print(f"音频数据形状调整为: {wav.shape}", flush=True)
        elif len(wav.shape) > 2:
            from fastapi.responses import JSONResponse
            return JSONResponse(status_code=400, content={"error": "音频数据形状不正确"})
    else:
        from fastapi.responses import JSONResponse
        return JSONResponse(status_code=400, content={"error": "音频数据类型不正确"})
    
    try:
        # 将音频数据转换为字节流
        buffer = BytesIO()
        sf.write(buffer, wav, SAMPLE_RATE, format='WAV')
        buffer.seek(0)
        print(f"音频数据写入成功，大小: {buffer.getbuffer().nbytes} 字节", flush=True)
        
        return StreamingResponse(buffer, media_type="audio/wav")
    except Exception as e:
        print(f"写入音频数据失败: {e}", flush=True)
        from fastapi.responses import JSONResponse
        return JSONResponse(status_code=500, content={"error": f"写入音频数据失败: {str(e)}"})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)