# MLX-LLM.cpp

MLX-LLM.cpp is a high-performance C/C++ library for Large Language Models (LLM), Vision-Language Models (VLM), and Whisper ASR inference. It leverages [MLX](https://github.com/ml-explore/mlx) to run efficiently on Apple Silicon hardware.

## Features

- 🚀 High-performance inference on Apple Silicon
- 🧠 Support for multiple model architectures (LLM, VLM, ASR)
- 🔧 Quantization support for memory efficiency
- 🎯 Easy-to-use C++ API
- 🐍 Python preprocessing utilities

## Supported Models

| Family | Models | Description |
|--------|--------|-------------|
| LLaMA 2 | llama_2_7b_chat_hf | Chat-optimized LLaMA 2 models |
| LLaMA 3 | llama_3_8b | Latest LLaMA 3 models |
| LLaMA 3.2 | Llama-3.2-11B-Vision-Instruct | Vision-Language model for multimodal tasks |
| TinyLLaMA | tiny_llama_1.1B_chat_v1.0 | Compact model for edge deployment |
| Whisper | **All Whisper models** | All sizes (tiny, base, small, medium, large) and variants |
| Gemma 3 | **All Gemma 3 models** | All sizes (2B, 9B, 27B) with various quantization (bf16, fp16, 4bit, 8bit) |

## Installation

### Prerequisites

First, install [MLX](https://github.com/ml-explore/mlx) on your system:

```bash
git clone https://github.com/ml-explore/mlx.git mlx && cd mlx
mkdir -p build && cd build
cmake .. && make -j
make install
```

## Building MLX-LLM.cpp

Clone the repository and its submodules:

```bash
git clone https://github.com/grorge123/mlx-llm.cpp.git
cd mlx-llm.cpp
git submodule update --init --recursive
```

Build the example:

```bash
mkdir build && cd build
cmake ..
cmake --build .
```

## Usage

### 1. Language Models (LLM)

Refer to `example/llm.cpp` for a simple demonstration using TinyLLaMA 1.1B.

#### Downloading Model Weights and Tokenizer

```bash
mkdir tiny && cd tiny
wget https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0/resolve/main/model.safetensors
cd ..
wget https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0/resolve/main/tokenizer.json
```

#### Running the LLM Example

From the `build` directory:

```bash
./llm
```

This will generate text using the TinyLLaMA 1.1B model.

### 2. Vision-Language Models (VLM)

For Vision-Language Models, you need to preprocess inputs and postprocess outputs. The library supports Gemma 3 and LLaMA 3.2 Vision models.

#### Method 1: Using Python Preprocessing Scripts

1. **Encode inputs** (image + text prompt):
   ```bash
   python3 example/encode.py <model_path> <image_path> "<your_prompt>"
   ```

2. **Run VLM inference**:
   ```bash
   ./vlm
   ```

3. **Decode outputs**:
   ```bash
   python3 example/decode.py <model_path> output.npy
   ```

#### Method 2: Using WASI Processor

Alternatively, you can use the [WASI Processor](https://github.com/second-state/wasi_processor) for encoding and decoding:

```bash
# Install WASI processor
git clone https://github.com/second-state/wasi_processor.git

# Use WASI processor for encoding/decoding
# (Refer to WASI processor documentation for detailed usage)
```

#### Example VLM Workflow

```bash
# Download a vision model (supports any Gemma 3 variant)
# Examples: gemma-3-2b-it, gemma-3-9b-it, gemma-3-27b-it
# With any quantization: bf16, fp16, 4bit, 8bit
mkdir gemma-3-4b-it-bf16
# Download model files from Hugging Face

# Encode image and prompt
python3 example/encode.py ./gemma-3-4b-it-bf16 ./example/sample_image.jpg "What's in this image?"

# Run inference
./vlm

# Decode the output
python3 example/decode.py ./gemma-3-4b-it-bf16 output.npy
```

### 3. Speech Recognition (Whisper)

For speech-to-text transcription using Whisper models. **All Whisper model variants are supported**:

- whisper-tiny (39M parameters)
- whisper-base (74M parameters) 
- whisper-small (244M parameters)
- whisper-medium (769M parameters)
- whisper-large (1550M parameters)
- whisper-large-v2, whisper-large-v3

```bash
# Download any Whisper model (example with tiny)
mkdir whisper-tiny
# Download model files from Hugging Face or OpenAI

# Run transcription
./whisper
```

Refer to `example/whisper.cpp` for implementation details.

### 4. Model Quantization

The library supports 4-bit quantization for memory efficiency:

```cpp
// Quantize model to 4-bit with group size of 128
Model = std::dynamic_pointer_cast<llm::Transformer>(Model->toQuantized(128, 4));
```

## Available Examples

- `example/llm.cpp` - Text generation with language models
- `example/vlm.cpp` - Vision-language model inference  
- `example/whisper.cpp` - Speech recognition
- `example/encode.py` - Image and text preprocessing for VLM
- `example/decode.py` - Token decoding for readable output

## Dependencies

- **MLX**: Apple's machine learning framework for Apple Silicon
- **transformers**: For model preprocessing (Python scripts)
- **PIL (Pillow)**: For image processing
- **tokenizers**: For text tokenization
- **spdlog**: For logging (C++)
- **CMake**: Build system

## Performance

MLX-LLM.cpp leverages Apple Silicon's unified memory architecture and Metal Performance Shaders for optimal performance:

- **Memory Efficiency**: 4-bit quantization reduces memory usage by ~75%
- **Speed**: Native Metal compute provides significant acceleration
- **Power Efficiency**: Optimized for Apple Silicon's power characteristics

## Model Download

You can download models from Hugging Face:

```bash
# For LLaMA models
git lfs clone https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0

# For Gemma 3 models (all sizes and quantizations supported)
git lfs clone https://huggingface.co/mlx-community/gemma-3-4b-pt-8bit
git lfs clone https://huggingface.co/mlx-community/gemma-3-4b-pt-4bit  
git lfs clone https://huggingface.co/mlx-community/gemma-3-4b-pt-bf16
# Also supports 4bit, 8bit, fp16, bf16 quantized versions

# For Whisper models (all sizes supported)
git lfs clone https://huggingface.co/openai/whisper-tiny
git lfs clone https://huggingface.co/openai/whisper-base
git lfs clone https://huggingface.co/openai/whisper-small
git lfs clone https://huggingface.co/openai/whisper-medium
git lfs clone https://huggingface.co/openai/whisper-large-v3
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [MLX](https://github.com/ml-explore/mlx) - Apple's machine learning framework
- [Hugging Face](https://huggingface.co) - Model hub and transformers library
- [WASI Processor](https://github.com/second-state/wasi_processor) - Alternative processing pipeline