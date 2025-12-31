#!/usr/bin/env python3
import os
import argparse
import hashlib
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import torch
import torch.nn.functional as F
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)

# -------------------------
# Canonicalization + stable id (same idea as your training)
# -------------------------
def canonicalize_answer(s: str) -> str:
    return " ".join(str(s).strip().split())

def stable_int64_hash(text: str) -> int:
    h = hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(h, "little", signed=True)

# -------------------------
# Query formatting
# -------------------------
def qwen_query_template(instruction: str, query: str) -> str:
    instruction = str(instruction).strip()
    query = str(query).strip()
    # Must contain "Instruct:" and "\nQuery:" for Qwen3-Embedding style
    return f"Instruct: {instruction}\nQuery:{query}"

def concat_query(instruction: str, query: str, sep: str = "\n") -> str:
    instruction = str(instruction).strip()
    query = str(query).strip()
    return f"{instruction}{sep}{query}"

def llm2vec_query(instruction: str, query: str, sep: str = "!@#$%^&*()") -> str:
    instruction = str(instruction).strip()
    query = str(query).strip()
    # Format: "instruction; separator query" (matches training format)
    return f"{instruction}; {sep} {query}"

# -------------------------
# Pooling helpers for BERT-like encoders
# -------------------------
def mean_pool(last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    # last_hidden: [B,T,H], mask: [B,T]
    mask = attention_mask.unsqueeze(-1).to(last_hidden.dtype)  # [B,T,1]
    summed = (last_hidden * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp_min(1e-6)
    return summed / denom

# -------------------------
# Qwen3 embedding (CausalLM) helpers (same as your notebook approach)
# -------------------------
def last_token_pool_qwen(last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    idx = attention_mask.sum(dim=1) - 1
    return last_hidden_states[
        torch.arange(last_hidden_states.size(0), device=last_hidden_states.device),
        idx
    ]

def get_last_hidden_state_qwen(model, input_ids, attention_mask):
    """
    OOM-safe-ish: fetch final hidden state without requesting all hidden_states.
    Works with Qwen-like wrappers.
    """
    m = model.module if hasattr(model, "module") else model

    # unwrap PEFT if present (harmless if not)
    if hasattr(m, "base_model"):
        m = m.base_model
    if hasattr(m, "model") and hasattr(m.model, "model"):
        m = m.model

    base = getattr(m, "model", None) or getattr(m, "transformer", None)
    if base is not None:
        out = base(input_ids=input_ids, attention_mask=attention_mask, use_cache=False, return_dict=True)
        if hasattr(out, "last_hidden_state"):
            return out.last_hidden_state
        if hasattr(out, "hidden_states") and out.hidden_states is not None:
            return out.hidden_states[-1]

    # fallback
    out = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
        use_cache=False,
        return_dict=True,
    )
    return out.hidden_states[-1]

def choose_attn_impl(prefer: str = "flash_attention_2") -> str:
    try:
        from transformers.modeling_flash_attention_utils import is_flash_attn_greater_or_equal_2
        if prefer == "flash_attention_2" and is_flash_attn_greater_or_equal_2():
            return "flash_attention_2"
    except Exception:
        pass
    return "sdpa"

# -------------------------
# Chunked top-k retrieval (float32 running scores; avoids fp16 overflow)
# -------------------------
@torch.inference_mode()
def topk_retrieve_indices(
    q_emb_cpu: torch.Tensor,  # [Nq,H] CPU
    d_emb_cpu: torch.Tensor,  # [Nd,H] CPU
    k: int,
    *,
    query_batch_size: int = 256,
    doc_chunk_size: int = 8192,
    device: str = "cuda",
) -> Tuple[torch.Tensor, torch.Tensor]:
    device_t = torch.device(device)
    if device_t.type == "cuda":
        compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        compute_dtype = torch.float32

    Nq = q_emb_cpu.size(0)
    Nd = d_emb_cpu.size(0)
    k = min(int(k), Nd)

    top_scores_all = torch.empty((Nq, k), dtype=torch.float32)
    top_indices_all = torch.empty((Nq, k), dtype=torch.long)

    for qs in tqdm(range(0, Nq, query_batch_size), desc="Retrieval (query batches)", unit="batch"):
        qe = q_emb_cpu[qs : qs + query_batch_size].to(device_t, dtype=compute_dtype, non_blocking=True)
        bq = qe.size(0)

        # float32 running buffers
        top_scores = torch.full((bq, k), -1e9, device=device_t, dtype=torch.float32)
        top_indices = torch.full((bq, k), -1, device=device_t, dtype=torch.long)

        for ds in tqdm(range(0, Nd, doc_chunk_size), desc="Doc chunks", unit="chunk", leave=False):
            de = d_emb_cpu[ds : ds + doc_chunk_size].to(device_t, dtype=compute_dtype, non_blocking=True)
            scores = (qe @ de.T).float()  # [bq, chunk], float32 for stable topk

            chunk = scores.size(1)
            idx_chunk = torch.arange(ds, ds + chunk, device=device_t, dtype=torch.long).unsqueeze(0).expand(bq, -1)

            comb_scores = torch.cat([top_scores, scores], dim=1)
            comb_idx = torch.cat([top_indices, idx_chunk], dim=1)

            new_scores, new_pos = torch.topk(comb_scores, k, dim=1)
            new_idx = comb_idx.gather(1, new_pos)

            top_scores, top_indices = new_scores, new_idx

        top_scores_all[qs : qs + bq] = top_scores.cpu()
        top_indices_all[qs : qs + bq] = top_indices.cpu()

    return top_scores_all, top_indices_all

# -------------------------
# Data handling: doc bank + duplicate-positive ids
# -------------------------
def build_doc_bank(
    answers: List[str],
    *,
    dedup_answers: bool,
) -> Tuple[List[str], np.ndarray]:
    """
    Returns:
      doc_texts: list[str]
      doc_ids:   np.int64 hash id for each doc
    """
    if not dedup_answers:
        doc_texts = answers
        doc_ids = np.array([stable_int64_hash(canonicalize_answer(a)) for a in doc_texts], dtype=np.int64)
        return doc_texts, doc_ids

    seen = {}
    doc_texts = []
    doc_canon = []
    for a in answers:
        c = canonicalize_answer(a)
        if c not in seen:
            seen[c] = len(doc_texts)
            doc_texts.append(a)
            doc_canon.append(c)
    doc_ids = np.array([stable_int64_hash(c) for c in doc_canon], dtype=np.int64)
    return doc_texts, doc_ids

# -------------------------
# Encoders
# -------------------------
class BaseEncoder:
    name: str
    def encode(self, texts: List[str]) -> torch.Tensor:
        """Return CPU float32 embeddings [N,H], L2-normalized."""
        raise NotImplementedError

    def cleanup(self):
        pass

class BertEncoder(BaseEncoder):
    def __init__(
        self,
        name: str,
        model_id: str,
        *,
        pooling: str = "mean",   # "mean" | "cls"
        max_len: int = 256,
        batch_size: int = 64,
        trust_remote_code: bool = False,
        device: Optional[str] = None,
    ):
        self.name = name
        self.model_id = model_id
        self.pooling = pooling
        self.max_len = int(max_len)
        self.batch_size = int(batch_size)

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code)
        # BERT base models typically use right-padding; keep default.
        dtype = torch.bfloat16 if (self.device.type == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16
        if self.device.type == "cpu":
            dtype = torch.float32

        self.model = AutoModel.from_pretrained(
            model_id,
            trust_remote_code=trust_remote_code,
            torch_dtype=dtype if self.device.type == "cuda" else None,
        ).to(self.device)
        self.model.eval()

        if pooling not in ("mean", "cls"):
            raise ValueError("pooling must be 'mean' or 'cls'")

    @torch.inference_mode()
    def encode(self, texts: List[str]) -> torch.Tensor:
        embs = []
        use_amp = self.device.type == "cuda"
        amp_dtype = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16

        for i in tqdm(range(0, len(texts), self.batch_size), desc=f"Encoding ({self.name})", unit="batch"):
            chunk = texts[i : i + self.batch_size]
            enc = self.tokenizer(
                chunk,
                padding=True,
                truncation=True,
                max_length=self.max_len,
                return_tensors="pt",
            )
            input_ids = enc["input_ids"].to(self.device, non_blocking=True)
            attention_mask = enc["attention_mask"].to(self.device, non_blocking=True)

            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                out = self.model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
                last_hidden = out.last_hidden_state  # [B,T,H]

                if self.pooling == "cls":
                    rep = last_hidden[:, 0]  # CLS
                else:
                    rep = mean_pool(last_hidden, attention_mask)

                rep = F.normalize(rep, p=2, dim=1)

            embs.append(rep.float().cpu())

        return torch.cat(embs, dim=0)

    def cleanup(self):
        try:
            del self.model
            del self.tokenizer
        except Exception:
            pass

class Qwen3EmbeddingEncoder(BaseEncoder):
    def __init__(
        self,
        name: str,
        model_id: str,
        *,
        max_len: int = 2048,
        batch_size: int = 16,
        use_4bit: bool = False,
        device_map: str = "auto",
    ):
        self.name = name
        self.model_id = model_id
        self.max_len = int(max_len)
        self.batch_size = int(batch_size)

        attn_impl = choose_attn_impl()
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left", trust_remote_code=True, use_fast=False)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if use_4bit:
            qconf = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                trust_remote_code=True,
                attn_implementation=attn_impl,
                quantization_config=qconf,
                device_map=device_map,
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                trust_remote_code=True,
                attn_implementation=attn_impl,
                torch_dtype=torch.bfloat16,
                device_map=device_map,
            )
        self.model.eval()

        # determine “input device” (works with device_map="auto")
        try:
            self.input_device = self.model.get_input_embeddings().weight.device
        except Exception:
            m = self.model.module if hasattr(self.model, "module") else self.model
            self.input_device = next(m.parameters()).device

    @torch.inference_mode()
    def encode(self, texts: List[str]) -> torch.Tensor:
        embs = []
        use_amp = self.input_device.type == "cuda"
        amp_dtype = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16

        for i in tqdm(range(0, len(texts), self.batch_size), desc=f"Encoding ({self.name})", unit="batch"):
            chunk = texts[i : i + self.batch_size]
            enc = self.tokenizer(
                chunk,
                padding=True,
                truncation=True,
                max_length=self.max_len,
                return_tensors="pt",
            )
            input_ids = enc["input_ids"].to(self.input_device, non_blocking=True)
            attention_mask = enc["attention_mask"].to(self.input_device, non_blocking=True)

            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                h = get_last_hidden_state_qwen(self.model, input_ids, attention_mask)  # [B,T,H]
                rep = last_token_pool_qwen(h, attention_mask)                          # [B,H]
                rep = F.normalize(rep, p=2, dim=1)

            embs.append(rep.float().cpu())

        return torch.cat(embs, dim=0)

    def cleanup(self):
        try:
            del self.model
            del self.tokenizer
        except Exception:
            pass

class LLM2Vec4CXR_Encoder(BaseEncoder):
    """
    Encoder using LLM2Vec4CXR with LLM2VecWrapper.
    Uses mean pooling with embed_mask (matching the working evaluation script).
    """
    def __init__(
        self,
        name: str = "llm2vec4cxr",
        model_id: str = "lukeingawesome/llm2vec4cxr",
        *,
        max_len: int = 512,
        batch_size: int = 32,
        device: Optional[str] = None,
        separator: str = "!@#$%^&*()",
    ):
        self.name = name
        self.model_id = model_id
        self.max_len = int(max_len)
        self.batch_size = int(batch_size)
        self.separator = separator

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        # Import LLM2VecWrapper
        import sys
        sys.path.insert(0, "/opt/project/chexembed/llm2vec4cxr")
        from llm2vec_wrapper import LLM2VecWrapper as LLM2Vec

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        # Load model with mean pooling (matching working script)
        dtype = torch.bfloat16 if (self.device.type == "cuda") else torch.float32
        self.model = LLM2Vec.from_pretrained(
            base_model_name_or_path=model_id,
            enable_bidirectional=True,
            pooling_mode="mean",
            max_length=max_len,
            torch_dtype=dtype,
            use_safetensors=True,
        )
        self.model = self.model.to(dtype).to(self.device).eval()

    def _tokenize_with_embed_mask(self, texts: List[str], use_separator: bool = False) -> dict:
        """Tokenize texts and create embed_mask for separator-based masking."""
        if use_separator:
            # Split by separator and create embed_mask for text after separator
            texts_after_sep = []
            original_texts = []
            for text in texts:
                parts = text.split(self.separator)
                texts_after_sep.append(parts[1].strip() if len(parts) > 1 else text)
                original_texts.append("".join(parts))
        else:
            original_texts = texts
            texts_after_sep = texts  # embed everything

        # Tokenize original texts
        encoding = self.tokenizer(
            original_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_len,
        )

        # Build embed_mask
        embed_mask = None
        for i, t in enumerate(texts_after_sep):
            sub = self.tokenizer(
                [t],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_len,
                add_special_tokens=False,
            )
            m = torch.zeros_like(encoding["attention_mask"][i])
            if len(sub["input_ids"][0]) > 0:
                m[-len(sub["input_ids"][0]):] = 1
            else:
                m = encoding["attention_mask"][i].clone()
            
            if embed_mask is None:
                embed_mask = m.unsqueeze(0)
            else:
                embed_mask = torch.cat([embed_mask, m.unsqueeze(0)], dim=0)

        encoding["embed_mask"] = embed_mask
        return encoding

    @torch.inference_mode()
    def encode(self, texts: List[str]) -> torch.Tensor:
        """Encode plain documents (no instruction separator, embed all tokens)."""
        embs = []
        for i in tqdm(range(0, len(texts), self.batch_size), desc=f"Encoding docs ({self.name})", unit="batch"):
            chunk = texts[i : i + self.batch_size]
            encoding = self._tokenize_with_embed_mask(chunk, use_separator=False)
            encoding = {k: v.to(self.device) for k, v in encoding.items()}
            rep = self.model.forward(encoding)  # [B, H]
            rep = F.normalize(rep, p=2, dim=1)
            embs.append(rep.float().cpu())
        return torch.cat(embs, dim=0)

    @torch.inference_mode()
    def encode_queries(self, texts: List[str]) -> torch.Tensor:
        """Encode queries with instruction separator (embed only text after separator)."""
        embs = []
        for i in tqdm(range(0, len(texts), self.batch_size), desc=f"Encoding queries ({self.name})", unit="batch"):
            chunk = texts[i : i + self.batch_size]
            encoding = self._tokenize_with_embed_mask(chunk, use_separator=True)
            encoding = {k: v.to(self.device) for k, v in encoding.items()}
            rep = self.model.forward(encoding)  # [B, H]
            rep = F.normalize(rep, p=2, dim=1)
            embs.append(rep.float().cpu())
        return torch.cat(embs, dim=0)

    def cleanup(self):
        try:
            del self.model
            del self.tokenizer
        except Exception:
            pass

class BiomedCLIP_TextEncoder(BaseEncoder):
    def __init__(
        self,
        name: str = "biomedclip_pubmedbert",
        model_id: str = "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
        *,
        context_length: int = 256,
        batch_size: int = 64,
        device: Optional[str] = None,
    ):
        self.name = name
        self.model_id = model_id
        self.context_length = int(context_length)
        self.batch_size = int(batch_size)

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        # Model card uses open_clip create_model_from_pretrained + get_tokenizer :contentReference[oaicite:6]{index=6}
        try:
            from open_clip import create_model_from_pretrained, get_tokenizer
        except Exception as e:
            raise RuntimeError(
                "open_clip is required for BiomedCLIP. Install with: pip install open_clip_torch"
            ) from e

        # open_clip HF hub prefix per model card
        hf_id = f"hf-hub:{model_id}"
        self.model, _preprocess = create_model_from_pretrained(hf_id)
        self.tokenizer = get_tokenizer(hf_id)

        self.model = self.model.to(self.device).eval()

    @torch.inference_mode()
    def encode(self, texts: List[str]) -> torch.Tensor:
        embs = []
        use_amp = self.device.type == "cuda"
        amp_dtype = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16

        for i in tqdm(range(0, len(texts), self.batch_size), desc=f"Encoding ({self.name})", unit="batch"):
            chunk = texts[i : i + self.batch_size]
            tokens = self.tokenizer(chunk, context_length=self.context_length).to(self.device)

            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                # open_clip models expose encode_text in most versions
                if hasattr(self.model, "encode_text"):
                    rep = self.model.encode_text(tokens)
                else:
                    # fallback: forward signature may differ; try calling model(None, tokens)
                    # (kept as safety net)
                    _img = None
                    out = self.model(_img, tokens)
                    # out might be tuple; try common patterns
                    rep = out[1] if isinstance(out, (tuple, list)) and len(out) > 1 else out

                rep = F.normalize(rep, p=2, dim=-1)

            embs.append(rep.float().cpu())

        return torch.cat(embs, dim=0)

    def cleanup(self):
        try:
            del self.model
            del self.tokenizer
        except Exception:
            pass

# -------------------------
# Evaluation core
# -------------------------
def compute_recall_at_k(
    q_emb: torch.Tensor,  # [N,H] CPU, normalized
    d_emb: torch.Tensor,  # [M,H] CPU, normalized
    target_ids: np.ndarray,  # [N] int64
    doc_ids: np.ndarray,     # [M] int64
    ks: Tuple[int, ...],
    *,
    retrieval_device: str,
    query_batch_size: int,
    doc_chunk_size: int,
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
    kmax = min(max(ks), d_emb.size(0))
    top_scores, top_indices = topk_retrieve_indices(
        q_emb,
        d_emb,
        kmax,
        query_batch_size=query_batch_size,
        doc_chunk_size=doc_chunk_size,
        device=retrieval_device,
    )

    top_idx_np = top_indices.numpy()          # [N,kmax]
    retrieved_ids = doc_ids[top_idx_np]       # [N,kmax]

    metrics: Dict[str, float] = {}
    for k in ks:
        k_eff = min(int(k), kmax)
        hits = (retrieved_ids[:, :k_eff] == target_ids[:, None]).any(axis=1)
        metrics[f"recall@{k_eff}"] = float(hits.mean())

    return metrics, top_idx_np, top_scores.numpy()

# -------------------------
# IO
# -------------------------
def read_dataframe(path: str) -> pd.DataFrame:
    p = path.lower()
    if p.endswith(".parquet"):
        return pd.read_parquet(path)
    if p.endswith(".csv"):
        return pd.read_csv(path)
    if p.endswith(".jsonl") or p.endswith(".json"):
        return pd.read_json(path, lines=True)
    raise ValueError("Unsupported file. Use .parquet, .csv, or .jsonl")

def parse_ks(s: str) -> Tuple[int, ...]:
    ks = tuple(int(x.strip()) for x in s.split(",") if x.strip())
    if not ks:
        raise ValueError("Empty --ks, example: --ks 1,5,10")
    return ks

# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Dataset with columns: instruction, query, answer. (.parquet/.csv/.jsonl)")
    ap.add_argument("--ks", type=str, default="1,5,10")
    ap.add_argument("--dedup_answers", action="store_true", help="Use unique canonical answers as doc bank (faster).")
    ap.add_argument("--limit", type=int, default=0, help="If >0, evaluate first N rows only.")
    ap.add_argument("--retrieval_device", choices=["cuda", "cpu"], default="cuda")
    ap.add_argument("--query_batch_size", type=int, default=256)
    ap.add_argument("--doc_chunk_size", type=int, default=8192)
    ap.add_argument("--save_results_csv", type=str, default="", help="Optional: save metrics table to CSV.")
    ap.add_argument("--save_debug_dir", type=str, default="", help="Optional: save per-model debug CSVs here.")
    ap.add_argument("--hf_home", type=str, default="")
    ap.add_argument("--cuda_visible_devices", type=str, default="")

    # Qwen baseline
    ap.add_argument("--qwen_model_id", type=str, default="Qwen/Qwen3-Embedding-0.6B")
    ap.add_argument("--qwen_max_len", type=int, default=2048)
    ap.add_argument("--qwen_bs", type=int, default=16)
    ap.add_argument("--qwen_use_4bit", action="store_true")

    # BERT baselines
    ap.add_argument("--bert_max_len", type=int, default=256)
    ap.add_argument("--bert_bs", type=int, default=64)
    ap.add_argument("--bert_no_instruction", action="store_true",
                    help="If set, BERT/BiomedCLIP models use only query text without instruction prefix.")

    # LLM2Vec4CXR
    ap.add_argument("--llm2vec_max_len", type=int, default=512)
    ap.add_argument("--llm2vec_bs", type=int, default=32)
    ap.add_argument("--llm2vec_sep", type=str, default="!@#$%^&*()")

    # BiomedCLIP text encoder
    ap.add_argument("--biomedclip_context_length", type=int, default=256)
    ap.add_argument("--biomedclip_bs", type=int, default=64)

    # Model selection
    ap.add_argument("--only_models", type=str, default="",
                    help="Comma-separated list of model kinds to run (e.g., 'llm2vec' or 'qwen,bert'). Empty = all models.")

    args = ap.parse_args()
    ks = parse_ks(args.ks)

    if args.hf_home:
        os.environ["HF_HOME"] = args.hf_home
    if args.cuda_visible_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    # Load data
    df = read_dataframe(args.data)
    for c in ["instruction", "query", "answer"]:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")

    if args.limit and args.limit > 0:
        df = df.iloc[: args.limit].reset_index(drop=True)

    df["instruction"] = df["instruction"].fillna("").astype(str)
    df["query"] = df["query"].fillna("").astype(str)
    df["answer"] = df["answer"].fillna("").astype(str)

    instructions = df["instruction"].tolist()
    queries = df["query"].tolist()
    answers = df["answer"].tolist()

    # Target ids per query (duplicates become positives by id)
    target_ids = np.array([stable_int64_hash(canonicalize_answer(a)) for a in answers], dtype=np.int64)

    # Doc bank
    doc_texts, doc_ids = build_doc_bank(answers, dedup_answers=args.dedup_answers)

    # Model list (one-by-one evaluation)
    model_specs = [
        dict(
            name=f"qwen3_embedding_base::{args.qwen_model_id}",
            kind="qwen",
        ),
        dict(
            name="BioClinicalBERT::emilyalsentzer/Bio_ClinicalBERT",
            kind="bert",
            model_id="emilyalsentzer/Bio_ClinicalBERT",
            pooling="mean",
            trust_remote_code=False,
        ),
        dict(
            name="CXR-BERT-specialized::microsoft/BiomedVLP-CXR-BERT-specialized",
            kind="bert",
            model_id="microsoft/BiomedVLP-CXR-BERT-specialized",
            pooling="cls",  # model card: CLS used to align text/image embeddings :contentReference[oaicite:7]{index=7}
            trust_remote_code=True,
        ),
        dict(
            name="BiomedCLIP_text::microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
            kind="biomedclip",
            model_id="microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
        ),
        dict(
            name="llm2vec4cxr::lukeingawesome/llm2vec4cxr",
            kind="llm2vec",
            model_id="lukeingawesome/llm2vec4cxr",
        ),
    ]

    # Filter models if --only_models is specified
    if args.only_models:
        only_kinds = set(k.strip().lower() for k in args.only_models.split(","))
        model_specs = [s for s in model_specs if s["kind"].lower() in only_kinds]
        print(f"Running only: {[s['name'] for s in model_specs]}")

    results_rows = []

    for spec in model_specs:
        name = spec["name"]
        kind = spec["kind"]
        print("\n" + "=" * 80)
        print(f"Evaluating: {name}")
        print("=" * 80)

        # Build query texts per model (important!)
        if kind == "qwen":
            query_texts = [qwen_query_template(i, q) for i, q in zip(instructions, queries)]
        elif kind == "llm2vec":
            query_texts = [llm2vec_query(i, q, sep=args.llm2vec_sep) for i, q in zip(instructions, queries)]
        elif kind in ("bert", "biomedclip"):
            # BERT / BiomedCLIP: optionally include instruction
            if args.bert_no_instruction:
                query_texts = queries  # query only, no instruction
            else:
                query_texts = [concat_query(i, q, sep="\n") for i, q in zip(instructions, queries)]
        else:
            # Fallback: plain concat
            query_texts = [concat_query(i, q, sep="\n") for i, q in zip(instructions, queries)]

        # Load encoder
        encoder: BaseEncoder
        if kind == "qwen":
            encoder = Qwen3EmbeddingEncoder(
                name=name,
                model_id=args.qwen_model_id,
                max_len=args.qwen_max_len,
                batch_size=args.qwen_bs,
                use_4bit=args.qwen_use_4bit,
                device_map="auto",
            )
            q_max_len = args.qwen_max_len
        elif kind == "bert":
            encoder = BertEncoder(
                name=name,
                model_id=spec["model_id"],
                pooling=spec["pooling"],
                max_len=args.bert_max_len,
                batch_size=args.bert_bs,
                trust_remote_code=spec.get("trust_remote_code", False),
            )
            q_max_len = args.bert_max_len
        elif kind == "biomedclip":
            encoder = BiomedCLIP_TextEncoder(
                name=name,
                model_id=spec["model_id"],
                context_length=args.biomedclip_context_length,
                batch_size=args.biomedclip_bs,
            )
            q_max_len = args.biomedclip_context_length
        elif kind == "llm2vec":
            encoder = LLM2Vec4CXR_Encoder(
                name=name,
                model_id=spec["model_id"],
                max_len=args.llm2vec_max_len,
                batch_size=args.llm2vec_bs,
                separator=args.llm2vec_sep,
            )
            q_max_len = args.llm2vec_max_len
        else:
            raise ValueError(f"Unknown kind: {kind}")

        # Encode queries + docs
        # For llm2vec: use encode_queries() which handles instruction separator
        if kind == "llm2vec":
            q_emb = encoder.encode_queries(query_texts)
        else:
            q_emb = encoder.encode(query_texts)
        d_emb = encoder.encode(doc_texts)

        # Compute recall@k
        metrics, top_idx_np, top_scores_np = compute_recall_at_k(
            q_emb=q_emb,
            d_emb=d_emb,
            target_ids=target_ids,
            doc_ids=doc_ids,
            ks=ks,
            retrieval_device=args.retrieval_device,
            query_batch_size=args.query_batch_size,
            doc_chunk_size=args.doc_chunk_size,
        )

        # Record results
        row = {"model": name, "n_queries": len(query_texts), "n_docs": len(doc_texts)}
        row.update(metrics)
        results_rows.append(row)

        print("Metrics:")
        for k in sorted(metrics.keys(), key=lambda x: int(x.split("@")[1])):
            print(f"  {k}: {metrics[k]:.6f}")

        # Optional debug dump
        if args.save_debug_dir:
            os.makedirs(args.save_debug_dir, exist_ok=True)
            pred_top1 = [doc_texts[j] for j in top_idx_np[:, 0]]
            dbg = df.copy()
            dbg["pred_doc_index@1"] = top_idx_np[:, 0]
            dbg["pred_answer@1"] = pred_top1
            dbg["score@1"] = top_scores_np[:, 0]
            # hits for each K
            for k in ks:
                k_eff = min(k, top_idx_np.shape[1])
                retrieved_ids = doc_ids[top_idx_np[:, :k_eff]]
                hits = (retrieved_ids == target_ids[:, None]).any(axis=1)
                dbg[f"hit@{k_eff}"] = hits

            out_path = os.path.join(
                args.save_debug_dir,
                name.replace("/", "_").replace("::", "__") + "_debug.csv"
            )
            dbg.to_csv(out_path, index=False)
            print(f"Saved debug CSV → {out_path}")

        # Cleanup GPU memory before next model
        encoder.cleanup()
        del encoder, q_emb, d_emb
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Final results table
    results_df = pd.DataFrame(results_rows)
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print(results_df)

    if args.save_results_csv:
        results_df.to_csv(args.save_results_csv, index=False)
        print(f"Saved results CSV → {args.save_results_csv}")

if __name__ == "__main__":
    main()
