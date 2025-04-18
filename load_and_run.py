import torch
from torch.nn import functional as F
from llm2vec_wrapper import LLM2VecWrapper as LLM2Vec

# Load the model from the saved directory
model_dir = './huggingface'
text_model = LLM2Vec.from_pretrained(
    base_model_name_or_path=model_dir,
    enable_bidirectional=True,
    pooling_mode="latent_attention",
    max_length=512,
    torch_dtype=torch.bfloat16,
)

# Move model to the appropriate device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
text_model.to(device).to(torch.bfloat16)
text_model.eval()

def tokenize(texts, tokenizer, max_length):
    texts_2 = []
    original_texts = []
    for text in texts:
        t = text.split('!@#$%^&*()')
        texts_2.append(t[1] if len(t) > 1 else "")
        original_texts.append("".join(t))

    original = tokenizer(
        original_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    if 1:
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
            if embed_mask is None:
                e_m = torch.zeros_like(original["attention_mask"][t_i])
                if len(ids["input_ids"][0]) > 0:
                    e_m[-len(ids["input_ids"][0]) :] = torch.ones(
                        len(ids["input_ids"][0])
                    )
                embed_mask = e_m.unsqueeze(0)
            else:
                e_m = torch.zeros_like(original["attention_mask"][t_i])
                if len(ids["input_ids"][0]) > 0:
                    e_m[-len(ids["input_ids"][0]) :] = torch.ones(
                        len(ids["input_ids"][0])
                    )
                embed_mask = torch.cat((embed_mask, e_m.unsqueeze(0)), dim=0)

        original["embed_mask"] = embed_mask
        return original

# Run the same example as in tutorial.py
separator = '!@#$%^&*()'
instruction = 'Determine the change or the status of the pleural effusion.; '
report = 'There has been some interval increase in the left-sided effusion. There continues to be volume loss at both bases. The right-sided PICC line tip is in the distal SVC.'
text = instruction + separator + report

all_keys = [text] + [
    'No pleural effusion',
    'Pleural effusion',
    'Pleural effusion is improving',
    'Pleural effusion is stable',
    'Pleural effusion is worsening'
]

# Tokenize and get embeddings
tokenized = tokenize(all_keys, text_model.tokenizer, 512).to(device)
tokenized = {k: v.to(torch.bfloat16) for k, v in tokenized.items()}
emb = text_model(tokenized)

# Calculate similarity
similarity = F.cosine_similarity(emb[0], emb[1:], dim=1)
print("Similarity scores:")
for label, score in zip(all_keys[1:], similarity):
    print(f"{label}: {score.item():.4f}") 