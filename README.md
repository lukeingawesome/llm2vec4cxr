# LLM2Vec4CXR

This repository contains code for using LLM2Vec models for chest X-ray report analysis.

## Installation

### Clone the Repository

```bash
git clone https://github.com/lukeingawesome/llm2vec4cxr.git
cd llm2vec4cxr
```

### Set Up Environment

Create and activate a conda environment with Python 3.8:

```bash
conda env create -f environment.yml
conda activate llm2vec4cxr
```

If you don't have the environment.yml file, you can create the environment manually:

```bash
conda create -n llm2vec4cxr python=3.8
conda activate llm2vec4cxr
pip install torch==2.4.1 torchvision==0.19.1 transformers==4.44.2 pandas llm2vec accelerate flash-attn==2.7.0.post2 #Recommended
```

### Install Package in Development Mode

Install the package in development mode to ensure any changes to the code are immediately reflected:

```bash
pip install -e .
```

## Usage

### Running the Tutorial

The repository includes a tutorial script that demonstrates how to use the LLM2Vec model for chest X-ray report analysis:

```bash
python tutorial.py
```

This script will:
1. Load the LLM2Vec model
2. Process a sample chest X-ray report
3. Compare the report with different pleural effusion status options
4. Display similarity scores for each option

## Model Details

The tutorial uses a pre-trained LLM2Vec model that has been fine-tuned for chest X-ray report analysis. The model is loaded from the specified path in the tutorial.py script.

## Special Thanks

This project is built upon the following repositories:

- [LLM2Vec](https://github.com/McGill-NLP/llm2vec) - A framework for converting decoder-only LLMs into text encoders
- [LLM2CLIP](https://github.com/microsoft/LLM2CLIP/tree/main) - Microsoft's implementation for connecting LLMs with CLIP models

We gratefully acknowledge the work of the researchers and developers behind these projects.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
