# LLM2Vec4CXR

This repository contains code for using LLM2Vec models for chest X-ray report analysis.

## Installation

### Clone the Repository

```bash
git clone https://github.com/lukeingawesome/llm2vec4cxr.git
cd llm2vec4cxr
```

### Download the Model

Download the pre-trained model from Google Drive:

Option 1 - Using command line:
```bash
# Create a models directory
mkdir -p models

# Download the model using wget or curl
wget --no-check-certificate 'https://drive.google.com/uc?export=download&id=1JB8hTgmVC1yJts8tmQamBij6LDcDfk0l' -O models/pytorch_model.bin
```

Option 2 - Direct download from Google Drive:
1. Visit [this Google Drive link](https://drive.google.com/file/d/1JB8hTgmVC1yJts8tmQamBij6LDcDfk0l/view?usp=sharing)
2. Click the "Download" button in the top right corner
3. Move the downloaded file to the `models` directory in your project folder
4. Rename the file to `pytorch_model.bin` if necessary

### Set Up Environment

Create and activate a conda environment with Python 3.8:

```bash
conda env create -f environment.yml
conda activate llm2vec4cxr
```

If you don't have the environment.yml file, you can create the environment manually: (Not yet tested)

```bash
conda create -n llm2vec4cxr python=3.8
conda activate llm2vec4cxr
pip install torch==2.4.1 torchvision==0.19.1 transformers==4.44.2 pandas accelerate flash-attn==2.7.0.post2 #Recommended
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
# Run with default model path
python tutorial.py

# Or specify a custom model path
python tutorial.py --model_path /path/to/your/model/pytorch_model.bin
```

This script will:
1. Load the LLM2Vec model from the specified path
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
