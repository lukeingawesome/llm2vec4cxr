#!/usr/bin/env python3
import os
import argparse
import hashlib
from typing import Dict, List, Tuple, Optional, Iterable

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
# Canonicalization + stable id (kept for compatibility; not required for MC4)
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
    return f"Instruct: {instruction}\nQuery:{query}"

def concat_query(instruction: str, query: str, sep: str = "\n") -> str:
    instruction = str(instruction).strip()
    query = str(query).strip()
    return f"{instruction}{sep}{query}"

def llm2vec_query(instruction: str, query: str, sep: str = "!@#$%^&*()") -> str:
    instruction = str(instruction).strip()
    query = str(query).strip()
    # Matches your current script's format
    return f"{instruction}; {sep} {query}"

# -------------------------
# Small tqdm helper (so we can disable inner progress)
# -------------------------
def maybe_tqdm(it: Iterable, *, enabled: bool, desc: str, unit: str):
    return tqdm(it, desc=desc, unit=unit) if enabled else it

# -------------------------
# Pooling helpers for BERT-like encoders
# -------------------------
def mean_pool(last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).to(last_hidden.dtype)  # [B,T,1]
    summed = (last_hidden * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp_min(1e-6)
    return summed / denom

# -------------------------
# Qwen3 embedding (CausalLM) helpers
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
# Encoders
# -------------------------
class BaseEncoder:
    name: str
    def encode(self, texts: List[str], *, show_progress: bool = False, desc: str = "") -> torch.Tensor:
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
    def encode(self, texts: List[str], *, show_progress: bool = False, desc: str = "") -> torch.Tensor:
        embs = []
        use_amp = self.device.type == "cuda"
        amp_dtype = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16

        it = maybe_tqdm(
            range(0, len(texts), self.batch_size),
            enabled=show_progress,
            desc=desc or f"Encoding ({self.name})",
            unit="batch",
        )
        for i in it:
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
                rep = last_hidden[:, 0] if self.pooling == "cls" else mean_pool(last_hidden, attention_mask)
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
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id, padding_side="left", trust_remote_code=True, use_fast=False
        )
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
    def encode(self, texts: List[str], *, show_progress: bool = False, desc: str = "") -> torch.Tensor:
        embs = []
        use_amp = self.input_device.type == "cuda"
        amp_dtype = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16

        it = maybe_tqdm(
            range(0, len(texts), self.batch_size),
            enabled=show_progress,
            desc=desc or f"Encoding ({self.name})",
            unit="batch",
        )
        for i in it:
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

class BiomedCLIP_TextEncoder(BaseEncoder):
    """
    BiomedCLIP text encoder using open_clip.
    """
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

        try:
            from open_clip import create_model_from_pretrained, get_tokenizer
        except Exception as e:
            raise RuntimeError(
                "open_clip is required for BiomedCLIP. Install with: pip install open_clip_torch"
            ) from e

        hf_id = f"hf-hub:{model_id}"
        self.model, _preprocess = create_model_from_pretrained(hf_id)
        self.tokenizer = get_tokenizer(hf_id)

        self.model = self.model.to(self.device).eval()

    @torch.inference_mode()
    def encode(self, texts: List[str], *, show_progress: bool = False, desc: str = "") -> torch.Tensor:
        embs = []
        use_amp = self.device.type == "cuda"
        amp_dtype = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16

        it = maybe_tqdm(
            range(0, len(texts), self.batch_size),
            enabled=show_progress,
            desc=desc or f"Encoding ({self.name})",
            unit="batch",
        )
        for i in it:
            chunk = texts[i : i + self.batch_size]
            tokens = self.tokenizer(chunk, context_length=self.context_length).to(self.device)

            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                if hasattr(self.model, "encode_text"):
                    rep = self.model.encode_text(tokens)
                else:
                    _img = None
                    out = self.model(_img, tokens)
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


class LLM2Vec4CXR_Encoder(BaseEncoder):
    """
    Your wrapper-based LLM2Vec4CXR encoder.
    - encode(): docs (embed all tokens)
    - encode_queries(): queries (embed only text after separator)
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
        wrapper_dir: str = "/opt/project/chexembed/llm2vec4cxr",
    ):
        self.name = name
        self.model_id = model_id
        self.max_len = int(max_len)
        self.batch_size = int(batch_size)
        self.separator = separator

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        # Import wrapper
        import sys
        if wrapper_dir and wrapper_dir not in sys.path:
            sys.path.insert(0, wrapper_dir)
        from llm2vec_wrapper import LLM2VecWrapper as LLM2Vec

        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        # Model
        dtype = torch.bfloat16 if (self.device.type == "cuda") else torch.float32
        self.model = LLM2Vec.from_pretrained(
            base_model_name_or_path=model_id,
            enable_bidirectional=True,
            pooling_mode="mean",
            max_length=max_len,
            torch_dtype=dtype,
            use_safetensors=True,
        )
        # NOTE: keep original behavior
        self.model = self.model.to(dtype).to(self.device).eval()

    def _tokenize_with_embed_mask(self, texts: List[str], use_separator: bool) -> dict:
        if use_separator:
            texts_after_sep = []
            original_texts = []
            for text in texts:
                parts = text.split(self.separator)
                texts_after_sep.append(parts[1].strip() if len(parts) > 1 else text)
                original_texts.append("".join(parts))  # remove separator, keep query at end
        else:
            original_texts = texts
            texts_after_sep = texts

        encoding = self.tokenizer(
            original_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_len,
        )

        embed_mask = []
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
            embed_mask.append(m.unsqueeze(0))

        encoding["embed_mask"] = torch.cat(embed_mask, dim=0)
        return encoding

    @torch.inference_mode()
    def encode(self, texts: List[str], *, show_progress: bool = False, desc: str = "") -> torch.Tensor:
        embs = []
        it = maybe_tqdm(
            range(0, len(texts), self.batch_size),
            enabled=show_progress,
            desc=desc or f"Encoding docs ({self.name})",
            unit="batch",
        )
        for i in it:
            chunk = texts[i : i + self.batch_size]
            encoding = self._tokenize_with_embed_mask(chunk, use_separator=False)
            encoding = {k: v.to(self.device) for k, v in encoding.items()}
            rep = self.model.forward(encoding)  # [B,H]
            rep = F.normalize(rep, p=2, dim=1)
            embs.append(rep.float().cpu())
        return torch.cat(embs, dim=0)

    @torch.inference_mode()
    def encode_queries(self, texts: List[str], *, show_progress: bool = False, desc: str = "") -> torch.Tensor:
        embs = []
        it = maybe_tqdm(
            range(0, len(texts), self.batch_size),
            enabled=show_progress,
            desc=desc or f"Encoding queries ({self.name})",
            unit="batch",
        )
        for i in it:
            chunk = texts[i : i + self.batch_size]
            encoding = self._tokenize_with_embed_mask(chunk, use_separator=True)
            encoding = {k: v.to(self.device) for k, v in encoding.items()}
            rep = self.model.forward(encoding)  # [B,H]
            rep = F.normalize(rep, p=2, dim=1)
            embs.append(rep.float().cpu())
        return torch.cat(embs, dim=0)

    def cleanup(self):
        try:
            del self.model
            del self.tokenizer
        except Exception:
            pass

# -------------------------
# MC4 evaluation (candidate0 must be top)
# -------------------------
def eval_mc4_accuracy(
    df: pd.DataFrame,
    *,
    encoder: BaseEncoder,
    kind: str,
    llm2vec_sep: str,
    bert_no_instruction: bool,
    row_batch_size: int,
    tol: float,
    save_debug_csv: str = "",
) -> Dict[str, float]:
    required = {"instruction", "query", "candidate0", "candidate1", "candidate2", "candidate3"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"df missing required columns: {sorted(missing)}")

    df_eval = df.copy()
    for c in list(required):
        df_eval[c] = df_eval[c].fillna("").astype(str)

    n_total = 0
    n_correct_strict = 0
    n_correct_tie = 0
    recall_hits = {1: 0, 2: 0, 3: 0, 4: 0}
    margin_sum = 0.0
    rank_sum = 0.0

    debug_header_written = False
    if save_debug_csv:
        # overwrite
        open(save_debug_csv, "w").write("")

    outer = tqdm(range(0, len(df_eval), row_batch_size), desc=f"Eval MC4 ({encoder.name})", unit="rows")
    for start in outer:
        batch = df_eval.iloc[start : start + row_batch_size]
        B = len(batch)
        if B == 0:
            continue

        inst = batch["instruction"].tolist()
        qtxt = batch["query"].tolist()

        # Build query strings per model kind
        if kind == "qwen":
            query_texts = [qwen_query_template(i, q) for i, q in zip(inst, qtxt)]
        elif kind == "llm2vec":
            query_texts = [llm2vec_query(i, q, sep=llm2vec_sep) for i, q in zip(inst, qtxt)]
        else:  # bert / biomedclip / default
            if bert_no_instruction:
                query_texts = qtxt
            else:
                query_texts = [concat_query(i, q, sep="\n") for i, q in zip(inst, qtxt)]

        # Flatten candidates [c0,c1,c2,c3, c0,c1,c2,c3, ...]
        c0 = batch["candidate0"].tolist()
        c1 = batch["candidate1"].tolist()
        c2 = batch["candidate2"].tolist()
        c3 = batch["candidate3"].tolist()
        cand_texts = []
        for i in range(B):
            cand_texts.extend([c0[i], c1[i], c2[i], c3[i]])

        # Encode
        if kind == "llm2vec" and hasattr(encoder, "encode_queries"):
            q_emb = encoder.encode_queries(query_texts, show_progress=False)  # [B,H] CPU
        else:
            q_emb = encoder.encode(query_texts, show_progress=False)          # [B,H] CPU

        c_emb = encoder.encode(cand_texts, show_progress=False)              # [4B,H] CPU
        H = q_emb.size(1)
        c_emb = c_emb.view(B, 4, H)

        # sims: [B,4] (normalized dot product = cosine)
        sims = torch.einsum("bh,bkh->bk", q_emb.float(), c_emb.float())

        pred = sims.argmax(dim=1)  # [B]
        correct_strict = (pred == 0)

        maxv = sims.max(dim=1).values
        correct_tie = (sims[:, 0] >= (maxv - tol))

        # recall@k within 4
        for k in (1, 2, 3, 4):
            topk = sims.topk(k, dim=1).indices
            hit = (topk == 0).any(dim=1)
            recall_hits[k] += int(hit.sum().item())

        other_max = sims[:, 1:].max(dim=1).values
        margin = sims[:, 0] - other_max
        margin_sum += float(margin.sum().item())

        rank = 1 + (sims[:, 1:] > sims[:, 0:1]).sum(dim=1)
        rank_sum += float(rank.sum().item())

        n_total += B
        n_correct_strict += int(correct_strict.sum().item())
        n_correct_tie += int(correct_tie.sum().item())

        if save_debug_csv:
            dbg = batch.copy()
            dbg["sim0"] = sims[:, 0].numpy()
            dbg["sim1"] = sims[:, 1].numpy()
            dbg["sim2"] = sims[:, 2].numpy()
            dbg["sim3"] = sims[:, 3].numpy()
            dbg["pred"] = pred.numpy()
            dbg["correct_strict"] = correct_strict.numpy()
            dbg["correct_tie"] = correct_tie.numpy()
            dbg["margin0_minus_best_other"] = margin.numpy()
            dbg["rank_of_candidate0"] = rank.numpy()

            dbg.to_csv(
                save_debug_csv,
                mode="a",
                index=False,
                header=(not debug_header_written),
            )
            debug_header_written = True

    if n_total == 0:
        raise RuntimeError("No rows evaluated (n_total=0). Check your dataframe.")

    metrics: Dict[str, float] = {
        "n_total": float(n_total),
        "chance_level": 0.25,
        "accuracy_top1_strict": float(n_correct_strict / n_total),
        "accuracy_top1_tie_aware": float(n_correct_tie / n_total),
        "mean_margin(sim0 - best_other)": float(margin_sum / n_total),
        "mean_rank_candidate0": float(rank_sum / n_total),
        "recall@1_within4": float(recall_hits[1] / n_total),
        "recall@2_within4": float(recall_hits[2] / n_total),
        "recall@3_within4": float(recall_hits[3] / n_total),
        "recall@4_within4": float(recall_hits[4] / n_total),
    }
    return metrics

# -------------------------
# IO helpers
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

# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Dataset with columns: instruction, query, candidate0..3 (.parquet/.csv/.jsonl)")
    ap.add_argument("--limit", type=int, default=0, help="If >0, evaluate first N rows only.")
    ap.add_argument("--row_batch_size", type=int, default=256, help="Rows per evaluation chunk.")
    ap.add_argument("--tol", type=float, default=1e-6, help="Tie-aware tolerance.")

    ap.add_argument("--save_results_csv", type=str, default="", help="Optional: save summary metrics table to CSV.")
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
                    help="If set, BERT models use only query text without instruction prefix.")

    # LLM2Vec4CXR
    ap.add_argument("--llm2vec_max_len", type=int, default=512)
    ap.add_argument("--llm2vec_bs", type=int, default=32)
    ap.add_argument("--llm2vec_sep", type=str, default="!@#$%^&*()")
    ap.add_argument("--llm2vec_wrapper_dir", type=str, default="/opt/project/chexembed/llm2vec4cxr")

    # BiomedCLIP config
    ap.add_argument("--biomedclip_model_id", default="microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224")
    ap.add_argument("--biomedclip_context_length", type=int, default=256)
    ap.add_argument("--biomedclip_bs", type=int, default=64)

    # Model selection
    ap.add_argument("--only_models", type=str, default="",
                    help="Comma-separated model kinds to run: qwen,bert,biomedclip,llm2vec. Empty = all.")
    args = ap.parse_args()

    if args.hf_home:
        os.environ["HF_HOME"] = args.hf_home
    if args.cuda_visible_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    print("Loading data...")
    df = read_dataframe(args.data)
    required_cols = ["instruction", "query", "candidate0", "candidate1", "candidate2", "candidate3"]
    for c in required_cols:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")

    if args.limit and args.limit > 0:
        df = df.iloc[: args.limit].reset_index(drop=True)

    # Ensure string columns
    for c in required_cols:
        df[c] = df[c].fillna("").astype(str)

    # Model specs (BERT variants included under kind="bert")
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
            pooling="cls",
            trust_remote_code=True,
        ),
        dict(
            name=f"BiomedCLIP_text::{args.biomedclip_model_id}",
            kind="biomedclip",
            model_id=args.biomedclip_model_id,
        ),
        dict(
            name="llm2vec4cxr::lukeingawesome/llm2vec4cxr",
            kind="llm2vec",
            model_id="lukeingawesome/llm2vec4cxr",
        ),
    ]

    # Filter models if requested
    if args.only_models:
        only_kinds = set(k.strip().lower() for k in args.only_models.split(",") if k.strip())
        model_specs = [s for s in model_specs if s["kind"].lower() in only_kinds]
        print(f"Running only: {[s['name'] for s in model_specs]}")

    results_rows = []

    for spec in model_specs:
        name = spec["name"]
        kind = spec["kind"]
        print("\n" + "=" * 80)
        print(f"Evaluating MC4 error-detection: {name}")
        print("=" * 80)

        # Instantiate encoder
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
        elif kind == "bert":
            encoder = BertEncoder(
                name=name,
                model_id=spec["model_id"],
                pooling=spec["pooling"],
                max_len=args.bert_max_len,
                batch_size=args.bert_bs,
                trust_remote_code=spec.get("trust_remote_code", False),
            )
        elif kind == "biomedclip":
            encoder = BiomedCLIP_TextEncoder(
                name=name,
                model_id=spec["model_id"],
                context_length=args.biomedclip_context_length,
                batch_size=args.biomedclip_bs,
            )
        elif kind == "llm2vec":
            encoder = LLM2Vec4CXR_Encoder(
                name=name,
                model_id=spec["model_id"],
                max_len=args.llm2vec_max_len,
                batch_size=args.llm2vec_bs,
                separator=args.llm2vec_sep,
                wrapper_dir=args.llm2vec_wrapper_dir,
            )
        else:
            raise ValueError(f"Unknown model kind: {kind}")

        # Debug path per model
        debug_path = ""
        if args.save_debug_dir:
            os.makedirs(args.save_debug_dir, exist_ok=True)
            debug_path = os.path.join(
                args.save_debug_dir,
                name.replace("/", "_").replace("::", "__") + "_mc4_debug.csv"
            )

        # Evaluate
        metrics = eval_mc4_accuracy(
            df,
            encoder=encoder,
            kind=kind,
            llm2vec_sep=args.llm2vec_sep,
            bert_no_instruction=args.bert_no_instruction,
            row_batch_size=args.row_batch_size,
            tol=args.tol,
            save_debug_csv=debug_path,
        )

        row = {"model": name}
        row.update(metrics)
        results_rows.append(row)

        print("Metrics:")
        for k, v in row.items():
            if k == "model":
                continue
            if k == "n_total":
                print(f"  {k}: {int(v)}")
            else:
                print(f"  {k}: {v:.6f}")

        if debug_path:
            print(f"Saved debug CSV → {debug_path}")

        # Cleanup
        encoder.cleanup()
        del encoder
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    results_df = pd.DataFrame(results_rows)
    print("\n" + "=" * 80)
    print("Summary (MC4 / error detection)")
    print("=" * 80)
    print(results_df)

    if args.save_results_csv:
        results_df.to_csv(args.save_results_csv, index=False)
        print(f"Saved results CSV → {args.save_results_csv}")

if __name__ == "__main__":
    main()
