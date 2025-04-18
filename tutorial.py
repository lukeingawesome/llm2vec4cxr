import torch
import torch.nn.functional as F
from llm2vec_wrapper import LLM2VecWrapper as LLM2Vec
import os
import argparse

def load_model(model_path=None, model_dir=None):
    """
    Load the LLM2Vec model with specified configuration.
    
    Args:
        model_path (str, optional): Path to the model checkpoint
        model_dir (str, optional): Directory containing the model files
    
    Returns:
        tuple: (model, tokenizer)
    """
    if model_dir and not os.path.isdir(model_dir):
        raise FileNotFoundError(f"Model directory '{model_dir}' not found.")
    
    # Load model with base configuration
    text_model = LLM2Vec.from_pretrained(
        base_model_name_or_path='microsoft/LLM2CLIP-Llama-3.2-1B-Instruct-CC-Finetuned',
        enable_bidirectional=True,
        pooling_mode="latent_attention",
        max_length=512,
        torch_dtype=torch.bfloat16,
    )

    if model_path:
        text_model = LLM2Vec.from_pretrained(
            base_model_name_or_path='microsoft/LLM2CLIP-Llama-3.2-1B-Instruct-CC-Finetuned',
            enable_bidirectional=True,
            pooling_mode="latent_attention",
            max_length=512,
            torch_dtype=torch.bfloat16,
            )
    else:
        text_model = LLM2Vec.from_pretrained(
            base_model_name_or_path='microsoft/LLM2CLIP-Llama-3.2-1B-Instruct-CC-Finetuned',
            enable_bidirectional=True,
            pooling_mode="mean",
            max_length=512,
            torch_dtype=torch.bfloat16,
            )
    
    # Load checkpoint if provided
    if model_path:
        try:
            ckpt = torch.load(model_path, weights_only=True)
            text_model.load_state_dict(ckpt, strict=False)
        except Exception as e:
            print(f"Warning: Failed to load checkpoint: {e}")
    
    # Configure tokenizer
    tokenizer = text_model.tokenizer
    tokenizer.padding_side = 'left'
    
    return text_model, tokenizer

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

def compute_similarities(model, tokenizer, texts, device):
    """
    Compute similarity scores for a given model and texts.
    
    Args:
        model: The LLM2Vec model
        tokenizer: The tokenizer to use
        texts (list): List of texts to compare
        device: The device to run computations on
    
    Returns:
        tuple: (embeddings, similarities)
    """
    with torch.no_grad():
        tokenized = tokenize(texts, tokenizer, 512).to(device)
        tokenized = tokenized.to(torch.bfloat16)
        embeddings = model(tokenized)
        similarities = F.cosine_similarity(embeddings[0], embeddings[1:], dim=1)
    return embeddings, similarities

def main(model_path=None):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load both models and tokenizers
    if model_path is None:
        model_path = '/model/llm2clip/llm2vec/1b_full/supervised4/checkpoint-4896/pytorch_model.bin'
    llm2vec4cxr_model, llm2vec4cxr_tokenizer = load_model(model_path=model_path)
    baseline_model, baseline_tokenizer = load_model()
    
    # Move models to device and set precision
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
        'Costophrenic angle is blunting',
        'Pleural effusion is improving',
        'Pleural effusion is stable',
        'Pleural effusion is worsening'
    ]
    
    all_texts = [text] + comparison_options
    
    # Compute similarities for both models
    _, similarities_llm2vec4cxr = compute_similarities(llm2vec4cxr_model, llm2vec4cxr_tokenizer, all_texts, device)
    _, similarities_baseline = compute_similarities(baseline_model, baseline_tokenizer, all_texts, device)
    
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run LLM2Vec model comparison')
    parser.add_argument('--model_path', type=str, help='Path to the model checkpoint', default=None)
    args = parser.parse_args()
    main(model_path=args.model_path)
    