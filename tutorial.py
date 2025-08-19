"""
LLM2Vec4CXR Tutorial - Model Comparison Demo

This script demonstrates the difference between the fine-tuned LLM2Vec4CXR model
and the baseline model on chest X-ray report analysis tasks.

The script:
1. Loads both models from Hugging Face 
2. Compares their performance on pleural effusion status determination
3. Shows similarity scores for various medical conditions

Usage:
    python tutorial.py
    python tutorial.py --device cuda
"""

import torch
import torch.nn.functional as F
from llm2vec_wrapper import LLM2VecWrapper as LLM2Vec
import os
import argparse

def load_llm2vec4cxr_model():
    """
    Load the fine-tuned LLM2Vec4CXR model from Hugging Face.
    
    Returns:
        LLM2Vec: The fine-tuned model with convenient methods
    """
    print("Loading LLM2Vec4CXR model from Hugging Face...")
    model = LLM2Vec.from_pretrained(
        base_model_name_or_path='lukeingawesome/llm2vec4cxr',
        enable_bidirectional=True,
        pooling_mode="latent_attention",
        max_length=512,
        torch_dtype=torch.bfloat16,
    )
    print("✓ LLM2Vec4CXR model loaded successfully")
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

# Note: The tokenize function has been replaced by the convenient 
# encode_with_instruction() method in the LLM2VecWrapper class

def compute_similarities(model, texts):
    """
    Compute similarity scores for a given model and texts using the convenient method.
    
    Args:
        model: The LLM2Vec model with encode_with_instruction method
        texts (list): List of texts to compare (first one is reference)
    
    Returns:
        tuple: (embeddings, similarities)
    """
    # Use the convenient encode_with_instruction method
    embeddings = model.encode_with_instruction(texts)
    similarities = F.cosine_similarity(embeddings[0], embeddings[1:], dim=1)
    return embeddings, similarities

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load both models from Hugging Face
    llm2vec4cxr_model = load_llm2vec4cxr_model()
    llm2vec4cxr_model = llm2vec4cxr_model.to(device)
    llm2vec4cxr_model.eval()
    
    baseline_model = load_baseline_model()
    baseline_model = baseline_model.to(device)
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
        'Pleural effusion is worsening'
    ]
    
    all_texts = [text] + comparison_options
    
    print("Computing similarities...")
    
    # Compute similarities for both models using the convenient methods
    _, similarities_llm2vec4cxr = compute_similarities(llm2vec4cxr_model, all_texts)
    _, similarities_baseline = compute_similarities(baseline_model, all_texts)
    
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
    