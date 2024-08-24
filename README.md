# MLX-LLM.cpp

MLX-LLM.cpp is a C/C++ library for LLM inference, based on [mlx-llm](https://github.com/riccardomusmeci/mlx-llm). It leverages [MLX](https://github.com/ml-explore/mlx) to run on Apple Silicon.

## Supported Models

| Family | Models |
|--------|--------|
| LLaMA 2 | llama_2_7b_chat_hf |
| LLaMA 3 | llama_3_8b |
| TinyLLaMA | tiny_llama_1.1B_chat_v1.0 |

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

Refer to `example/main.cpp` for a simple demonstration using TinyLLaMA 1.1B.

### Downloading Model Weights and Tokenizer

```bash
mkdir tiny && cd tiny
wget https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0/resolve/main/model.safetensors
cd ..
wget https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0/resolve/main/tokenizer.json
```

### Running the Example

From the `build` directory:

```bash
./main
```

This will generate results using the TinyLLaMA 1.1B model.