"""
LLM2Vec4CXR Tutorial - Model Comparison Demo

This script demonstrates the difference between the fine-tuned LLM2Vec4CXR model
and the baseline model on chest X-ray report analysis tasks.

Usage:
    python tutorial.py
"""

import torch
from llm2vec_wrapper import LLM2VecWrapper as LLM2Vec

def main():
    print("=== LLM2Vec4CXR vs Baseline Model Comparison ===\n")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load models
    print("\nLoading models...")
    
    # Load fine-tuned LLM2Vec4CXR model
    llm2vec4cxr_model = LLM2Vec.from_pretrained(
        base_model_name_or_path='lukeingawesome/llm2vec4cxr',
        pooling_mode="latent_attention",
        max_length=512,
        enable_bidirectional=True,
        torch_dtype=torch.bfloat16,
        use_safetensors=True,
    ).to(device).eval()
    
    # Load baseline model
    baseline_model = LLM2Vec.from_pretrained(
        base_model_name_or_path='microsoft/LLM2CLIP-Llama-3.2-1B-Instruct-CC-Finetuned',
        enable_bidirectional=True,
        pooling_mode="mean",
        max_length=512,
        torch_dtype=torch.bfloat16,
    ).to(device).eval()
    
    # Configure tokenizers
    llm2vec4cxr_model.tokenizer.padding_side = 'left'
    baseline_model.tokenizer.padding_side = 'left'
    
    print("✓ Models loaded successfully\n")
    
    # Define the medical text analysis task
    instruction = 'Determine the change or the status of the pleural effusion.; '
    report = 'There is a small increase in the left-sided effusion. There continues to be volume loss at both bases. The right-sided PICC line tip is in the distal SVC.'
    query_text = instruction + '!@#$%^&*()' + report
    
    # Define comparison options
    comparison_options = [
        'No pleural effusion',
        'Pleural effusion',
        'Effusion is seen in the right',
        'Effusion is seen in the left',
        'Costophrenic angle is blunting',
        'PE is found in the left',
        'Eff. is seen in the left',
        'Pleural effusion is improving',
        'Pleural effusion is stable',
        'Pleural effusion is worsening',
        'There is no pneumothorax',
        'There is a pneumothorax',
    ]
    
    print("Computing similarities...")
    
    # Compute similarities using the wrapper methods
    similarities_llm2vec4cxr = llm2vec4cxr_model.compute_similarities(
        query_text, comparison_options, device=device
    )
    
    similarities_baseline = baseline_model.compute_similarities(
        query_text, comparison_options, device=device
    )
    
    # Display results
    print("\n" + "="*75)
    print("Similarity Score Comparison".center(75))
    print(f"Query: {report}")
    print("="*75)
    print("│" + "Option".center(35) + "│" + "LLM2Vec4CXR".center(18) + "│" + "Baseline".center(18) + "│")
    print("├" + "─"*35 + "┼" + "─"*18 + "┼" + "─"*18 + "┤")
    
    for option, score_llm2vec4cxr, score_baseline in zip(comparison_options, similarities_llm2vec4cxr, similarities_baseline):
        print("│" + option.ljust(35) + "│" + 
              f"{score_llm2vec4cxr.item():.4f}".center(18) + "│" + 
              f"{score_baseline.item():.4f}".center(18) + "│")
    
    print("└" + "─"*35 + "┴" + "─"*18 + "┴" + "─"*18 + "┘")
    print("="*75)
    
    # Show best matches
    best_llm2vec4cxr = torch.argmax(similarities_llm2vec4cxr).item()
    best_baseline = torch.argmax(similarities_baseline).item()
    
    print(f"\nBest match for LLM2Vec4CXR: {comparison_options[best_llm2vec4cxr]} ({similarities_llm2vec4cxr[best_llm2vec4cxr].item():.4f})")
    print(f"Best match for Baseline: {comparison_options[best_baseline]} ({similarities_baseline[best_baseline].item():.4f})")
    
    print("\n🎯 The fine-tuned LLM2Vec4CXR model shows more nuanced similarity scores")
    print("   and better understanding of medical terminology compared to the baseline!")

if __name__ == "__main__":
    main()
    