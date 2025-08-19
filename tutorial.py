"""
LLM2Vec4CXR Tutorial - Model Comparison Demo

This script demonstrates the difference between the fine-tuned LLM2Vec4CXR model
and the baseline model on chest X-ray report analysis tasks.

The script:
1. Loads the fine-tuned model from HuggingFace (lukeingawesome/llm2vec4cxr)
2. Loads the baseline model for comparison
3. Compares their performance on pleural effusion status determination
4. Shows similarity scores for various medical conditions

Usage:
    python tutorial.py
    python tutorial.py --device cuda
"""

import torch
import torch.nn.functional as F
from llm2vec_wrapper import LLM2VecWrapper as LLM2Vec
import argparse

def load_llm2vec4cxr_model():
    """
    Load the fine-tuned LLM2Vec4CXR model from HuggingFace.
    
    Returns:
        LLM2Vec: The fine-tuned model
    """
    print("Loading LLM2Vec4CXR model from HuggingFace...")
    
    # Load the model (this reads llm2vec_config.json automatically)
    # Latent attention weights are automatically loaded by the wrapper
    model = LLM2Vec.from_pretrained(
        base_model_name_or_path='lukeingawesome/llm2vec4cxr',
        pooling_mode="latent_attention",
        max_length=512,
        enable_bidirectional=True,
        torch_dtype=torch.bfloat16,
        use_safetensors=True,
    )
    print("✓ Model loaded successfully")
    
    return model

def load_baseline_model():
    """
    Load the baseline model (original base model with mean pooling).
    
    Returns:
        LLM2Vec: The baseline model for comparison
    """
    print("Loading baseline model...")
    model = LLM2Vec.from_pretrained(
        base_model_name_or_path='microsoft/LLM2CLIP-Llama-3.2-1B-Instruct-CC-Finetuned',
        enable_bidirectional=True,
        pooling_mode="mean",
        max_length=512,
        torch_dtype=torch.bfloat16,
    )
    print("✓ Baseline model loaded successfully")
    return model

def tokenize(texts, tokenizer, max_length):
    """
    Tokenize texts with special handling for separator-based splitting.
    
    Args:
        texts (list): List of texts to tokenize
        tokenizer: The tokenizer to use
        max_length (int): Maximum sequence length
    
    Returns:
        dict: Tokenized inputs with attention masks
    """
    texts_2 = []
    original_texts = []
    separator = '!@#$%^&*()'
    
    for text in texts:
        parts = text.split(separator)
        texts_2.append(parts[1] if len(parts) > 1 else "")
        original_texts.append("".join(parts))

    # Tokenize original texts
    tokenized = tokenizer(
        original_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    
    # Create embedding masks
    embed_mask = None
    for t_i, t in enumerate(texts_2):
        ids = tokenizer(
            [t],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
            add_special_tokens=False,
        )
        
        e_m = torch.zeros_like(tokenized["attention_mask"][t_i])
        if len(ids["input_ids"][0]) > 0:
            e_m[-len(ids["input_ids"][0]):] = torch.ones(len(ids["input_ids"][0]))
            
        if embed_mask is None:
            embed_mask = e_m.unsqueeze(0)
        else:
            embed_mask = torch.cat((embed_mask, e_m.unsqueeze(0)), dim=0)

    tokenized["embed_mask"] = embed_mask
    return tokenized

def compute_similarities(model, texts, device):
    """
    Compute similarity scores for a given model and texts using manual tokenization.
    
    Args:
        model: The LLM2Vec model
        texts (list): List of texts to compare
        device: The device to run computations on
    
    Returns:
        tuple: (embeddings, similarities)
    """
    tokenizer = model.tokenizer
    with torch.no_grad():
        tokenized = tokenize(texts, tokenizer, 512).to(device)
        tokenized = tokenized.to(torch.bfloat16)
        embeddings = model(tokenized)
        similarities = F.cosine_similarity(embeddings[0], embeddings[1:], dim=1)
    return embeddings, similarities

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load both models and configure tokenizers
    llm2vec4cxr_model = load_llm2vec4cxr_model()
    baseline_model = load_baseline_model()
    
    # Configure tokenizers (same as tutorial_past.py)
    llm2vec4cxr_model.tokenizer.padding_side = 'left'
    baseline_model.tokenizer.padding_side = 'left'
    
    # Move models to device and set precision (same as tutorial_past.py)
    llm2vec4cxr_model = llm2vec4cxr_model.to(device).to(torch.bfloat16)
    baseline_model = baseline_model.to(device).to(torch.bfloat16)
    llm2vec4cxr_model.eval()
    baseline_model.eval()
    
    # Define input text and comparison options
    separator = '!@#$%^&*()'
    instruction = 'Determine the change or the status of the pleural effusion.; '
    report = 'There is a small increase in the left-sided effusion. There continues to be volume loss at both bases. The right-sided PICC line tip is in the distal SVC.'
    text = instruction + separator + report
    
    comparison_options = [
        'No pleural effusion',
        'Pleural effusion',
        'Effusion is seen in the right',
        'Effusion is seen in the left',
        'Costophrenic angle is blunting', #Similar
        'PE is found in the left', #Abbreviation
        'Eff. is seen in the left', #Abbreviation
        'Pleural effusion is improving',
        'Pleural effusion is stable',
        'Pleural effusion is worsening',
        'There is a man behind the door',
        'There is no pneumothorax',
        'There is a pneumothorax',
    ]
    
    all_texts = [text] + comparison_options
    
    print("Computing similarities...")
    
    # Compute similarities for both models using manual tokenization (same as tutorial_past.py)
    _, similarities_llm2vec4cxr = compute_similarities(llm2vec4cxr_model, all_texts, device)
    _, similarities_baseline = compute_similarities(baseline_model, all_texts, device)
    
    # Print comparison results with pretty formatting
    print("\n" + "="*75)
    print("Similarity Score Comparison".center(75))
    print(f"Original text: {report}")
    print("="*75)
    print("│" + "Option".center(35) + "│" + "LLM2Vec4CXR".center(18) + "│" + "Baseline".center(18) + "│")
    print("├" + "─"*35 + "┼" + "─"*18 + "┼" + "─"*18 + "┤")
    
    for option, score_llm2vec4cxr, score_baseline in zip(comparison_options, similarities_llm2vec4cxr, similarities_baseline):
        print("│" + option.ljust(35) + "│" + 
              f"{score_llm2vec4cxr.item():.4f}".center(18) + "│" + 
              f"{score_baseline.item():.4f}".center(18) + "│")
    
    print("└" + "─"*35 + "┴" + "─"*18 + "┴" + "─"*18 + "┘")
    print("="*75)
    
    # Show which model performed better
    best_llm2vec4cxr = torch.argmax(similarities_llm2vec4cxr).item()
    best_baseline = torch.argmax(similarities_baseline).item()
    
    print(f"\nBest match for LLM2Vec4CXR: {comparison_options[best_llm2vec4cxr]} ({similarities_llm2vec4cxr[best_llm2vec4cxr].item():.4f})")
    print(f"Best match for Baseline: {comparison_options[best_baseline]} ({similarities_baseline[best_baseline].item():.4f})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run LLM2Vec4CXR vs Baseline model comparison')
    parser.add_argument('--device', type=str, choices=['cuda', 'cpu', 'auto'], default='auto',
                       help='Device to run the models on (default: auto)')
    args = parser.parse_args()
    main()
    