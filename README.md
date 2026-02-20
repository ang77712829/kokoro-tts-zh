# Kokoro TTS 中文语音合成系统

## 项目简介

Kokoro TTS 是一个轻量级但功能强大的中文语音合成系统，基于先进的深度学习模型，能够生成自然流畅的中文语音。

## 功能特性

- ✅ 支持中文语音合成
- ✅ 支持英文语音合成
- ✅ 多种声音模型可选
- ✅ 支持语速调节
- ✅ GPU 加速推理（可选）
- ✅ CPU 运行模式（兼容所有设备）
- ✅ Docker 容器化部署
- ✅ RESTful API 接口
- ✅ 支持 OpenAI 风格的 API 调用
- ✅ 支持 Tavern AI 等第三方应用集成
- ✅ 自动依赖管理

## 系统要求

### CPU 版本
- Python 3.8+
- 2GB 以上内存

### GPU 版本
- Python 3.8+
- CUDA 11.7+ 或 12.1+
- NVIDIA GPU（至少 4GB 显存）

## 快速开始

### 前置要求
- **Git LFS**：由于模型文件较大，需要安装 Git LFS 来克隆项目
  - 安装方法：https://git-lfs.com/
  - 安装后初始化：`git lfs install`

### 方法一：一键启动脚本

1. 克隆项目到本地（使用 Git LFS）
   ```bash
   git clone https://github.com/ang77712829/kokoro-tts-zh.git
   cd kokoro-tts-zh
   ```

2. 运行启动脚本
   ```bash
   python run-tts.py
   ```

3. 根据提示选择启动 CPU 版本或 GPU 版本

4. 服务启动后，访问以下地址：
   - 本地访问：http://localhost:8000（CPU）或 http://localhost:8001（GPU）
   - 局域网访问：http://<你的IP地址>:8000 或 http://<你的IP地址>:8001

### 方法二：手动安装

#### CPU 版本

1. 进入 CPU 版本目录
   ```bash
   cd tts-project-cpu
   ```

2. 安装依赖
   ```bash
   pip install -r requirements.txt
   ```

3. 启动服务
   ```bash
   python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
   ```

#### GPU 版本

1. 进入 GPU 版本目录
   ```bash
   cd tts-project-gpu
   ```

2. 安装依赖
   ```bash
   pip install -r requirements.txt
   ```

3. 启动服务
   ```bash
   python -m uvicorn app.main:app --host 0.0.0.0 --port 8001 --reload
   ```

## API 文档

### 基本 API 接口

#### POST /api/tts

**请求参数**：
- `text`：要合成的文本
- `voice`：声音模型（默认为 zf_001）
- `speed`：语速（默认为 1.0）

**请求示例**：
```json
{
  "text": "你好，这是一个语音合成示例。",
  "voice": "zf_001",
  "speed": 1.0
}
```

**响应**：
- 返回音频文件（WAV 格式）

### OpenAI 风格 API

#### POST /api/tts

**请求参数**：
- `input`：要合成的文本（与 text 参数等价）
- `voice`：声音模型
- `speed`：语速

**请求示例**：
```json
{
  "input": "你好，这是一个语音合成示例。",
  "voice": "zf_001",
  "speed": 1.0
}
```

## Docker 部署

### CPU 版本

1. 进入 Docker CPU 目录
   ```bash
   cd docker/cpu
   ```

2. 构建并启动容器
   ```bash
   docker-compose up -d
   ```

3. 服务将在 http://localhost:8000 运行

### GPU 版本

1. 进入 Docker GPU 目录
   ```bash
   cd docker/gpu
   ```

2. 构建并启动容器
   ```bash
   docker-compose up -d
   ```

3. 服务将在 http://localhost:8001 运行

## 支持的声音模型

- `zf_001`：默认中文女声
- `af_maple`：英文女声
- 更多模型请参考 models 目录

## 常见问题

### Q: 服务启动失败，提示缺少依赖

A: 运行一键启动脚本 `run-tts.py`，它会自动安装所有必要的依赖。

### Q: GPU 版本无法使用 GPU 加速

A: 确保你的系统已安装正确版本的 CUDA，并且使用了支持 CUDA 的 PyTorch 版本。一键启动脚本会自动检测并安装适合的 PyTorch 版本。

### Q: 如何集成到其他应用？

A: 可以通过 RESTful API 接口集成到任何支持 HTTP 请求的应用中，例如 Tavern AI、Bot 框架等。

## 项目结构

```
tts-project/
├── tts-project-cpu/        # CPU 版本
│   ├── app/
│   │   ├── main.py         # 主应用
│   │   └── templates/
│   └── requirements.txt    # 依赖文件
├── tts-project-gpu/        # GPU 版本
│   ├── app/
│   │   ├── main.py         # 主应用
│   │   └── templates/
│   └── requirements.txt    # 依赖文件
├── docker/                 # Docker 配置
│   ├── cpu/                # CPU 版本 Docker 配置
│   └── gpu/                # GPU 版本 Docker 配置
├── models/                 # 模型文件
├── run-tts.py              # 一键启动脚本
├── README.md               # 中文文档
└── README_EN.md            # 英文文档
```

## 许可证

本项目采用 MIT 许可证。

## 更新日志

### v1.0.0 (2026-02-21)
- 初始版本发布
- 支持中英文语音合成
- 支持 CPU 和 GPU 运行模式
- 提供 Docker 部署方案
- 支持 RESTful API 接口
- 支持 OpenAI 风格的 API 调用
- 支持 Tavern AI 等第三方应用集成

## 致谢

本项目基于以下优秀的开源项目和模型：

### 核心模型
- **Kokoro-82M-v1.1-zh**：轻量级但功能强大的TTS模型
  - 模型地址：https://huggingface.co/hexgrad/Kokoro-82M-v1.1-zh
  - 感谢 LongMaoData 提供的中文数据集
  - 感谢 hexgrad 团队的模型训练和开源贡献

### 技术架构
- **StyleTTS 2**：先进的语音合成架构
  - 论文：https://arxiv.org/abs/2306.07691
  - 项目：https://github.com/yl4579/StyleTTS2

- **ISTFTNet**：高效的声码器
  - 论文：https://arxiv.org/abs/2203.02395

### 依赖库
- PyTorch：深度学习框架
- FastAPI：Web API框架
- Uvicorn：ASGI服务器
- NumPy：数值计算库
- jieba：中文分词库

## 联系方式

- GitHub: https://github.com/ang77712829/kokoro-tts-zh

---

**注意**：本项目仅供学习和研究使用，请勿用于商业用途。