# Kokoro TTS Chinese Speech Synthesis System

## Project Introduction

Kokoro TTS is a lightweight but powerful Chinese speech synthesis system based on advanced deep learning models, capable of generating natural and fluent Chinese speech.

## Features

- ✅ Support for Chinese speech synthesis
- ✅ Support for English speech synthesis
- ✅ Multiple voice models available
- ✅ Support for speech rate adjustment
- ✅ GPU accelerated inference (optional)
- ✅ CPU runtime mode (compatible with all devices)
- ✅ Docker containerized deployment
- ✅ RESTful API interface
- ✅ Support for OpenAI-style API calls
- ✅ Support for integration with third-party applications like Tavern AI
- ✅ Automatic dependency management

## System Requirements

### CPU Version
- Python 3.8+
- 2GB+ RAM

### GPU Version
- Python 3.8+
- CUDA 11.7+ or 12.1+
- NVIDIA GPU (at least 4GB VRAM)

## Quick Start

### Method 1: One-click Startup Script

1. Clone the project to your local machine

2. Run the startup script
   ```bash
   python run-tts.py
   ```

3. Follow the prompts to select CPU version or GPU version

4. After the service starts, access the following addresses:
   - Local access: http://localhost:8000 (CPU) or http://localhost:8001 (GPU)
   - LAN access: http://<your-IP-address>:8000 or http://<your-IP-address>:8001

### Method 2: Manual Installation

#### CPU Version

1. Enter the CPU version directory
   ```bash
   cd tts-project-cpu
   ```

2. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```

3. Start the service
   ```bash
   python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
   ```

#### GPU Version

1. Enter the GPU version directory
   ```bash
   cd tts-project-gpu
   ```

2. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```

3. Start the service
   ```bash
   python -m uvicorn app.main:app --host 0.0.0.0 --port 8001 --reload
   ```

## API Documentation

### Basic API Interface

#### POST /api/tts

**Request Parameters**:
- `text`: Text to synthesize
- `voice`: Voice model (default: zf_001)
- `speed`: Speech rate (default: 1.0)

**Request Example**:
```json
{
  "text": "Hello, this is a speech synthesis example.",
  "voice": "zf_001",
  "speed": 1.0
}
```

**Response**:
- Returns audio file (WAV format)

### OpenAI-style API

#### POST /api/tts

**Request Parameters**:
- `input`: Text to synthesize (equivalent to text parameter)
- `voice`: Voice model
- `speed`: Speech rate

**Request Example**:
```json
{
  "input": "Hello, this is a speech synthesis example.",
  "voice": "zf_001",
  "speed": 1.0
}
```

## Docker Deployment

### CPU Version

1. Enter the Docker CPU directory
   ```bash
   cd docker/cpu
   ```

2. Build and start the container
   ```bash
   docker-compose up -d
   ```

3. The service will run at http://localhost:8000

### GPU Version

1. Enter the Docker GPU directory
   ```bash
   cd docker/gpu
   ```

2. Build and start the container
   ```bash
   docker-compose up -d
   ```

3. The service will run at http://localhost:8001

## Supported Voice Models

- `zf_001`: Default Chinese female voice
- `af_maple`: English female voice
- More models please refer to the models directory

## Frequently Asked Questions

### Q: Service fails to start, missing dependencies

A: Run the one-click startup script `run-tts.py`, which will automatically install all necessary dependencies.

### Q: GPU version cannot use GPU acceleration

A: Ensure your system has the correct version of CUDA installed and is using a CUDA-enabled PyTorch version. The one-click startup script will automatically detect and install the appropriate PyTorch version.

### Q: How to integrate with other applications?

A: Can be integrated into any application that supports HTTP requests through the RESTful API interface, such as Tavern AI, Bot frameworks, etc.

## Project Structure

```
tts-project/
├── tts-project-cpu/        # CPU version
│   ├── app/
│   │   ├── main.py         # Main application
│   │   └── templates/
│   └── requirements.txt    # Dependency file
├── tts-project-gpu/        # GPU version
│   ├── app/
│   │   ├── main.py         # Main application
│   │   └── templates/
│   └── requirements.txt    # Dependency file
├── docker/                 # Docker configuration
│   ├── cpu/                # CPU version Docker configuration
│   └── gpu/                # GPU version Docker configuration
├── models/                 # Model files
├── run-tts.py              # One-click startup script
├── README.md               # Chinese documentation
└── README_EN.md            # English documentation
```

## License

This project is licensed under the MIT License.

## Changelog

### v1.0.0 (2026-02-21)
- Initial version release
- Support for Chinese and English speech synthesis
- Support for CPU and GPU runtime modes
- Provide Docker deployment方案
- Support for RESTful API interface
- Support for OpenAI-style API calls
- Support for integration with third-party applications like Tavern AI

## Contact

- GitHub: https://github.com/ang77712829/kokoro-tts-zh

---

**Note**: This project is for learning and research purposes only, not for commercial use.