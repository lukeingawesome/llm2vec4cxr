from llm2vec import LLM2Vec
from peft import PeftModel
from transformers import (
    AutoConfig,
    PretrainedConfig,
    AutoTokenizer,

)
import logging
import json
import os
logger = logging.getLogger(__name__)
class LLM2VecWrapper(LLM2Vec):
    def __init__(self, *args, **kwargs):
        super(LLM2VecWrapper, self).__init__(*args, **kwargs)

    def prepare_for_tokenization(self, text):
        text = (
            "<|start_header_id|>user<|end_header_id|>\n\n"
            + text.strip()
            + "<|eot_id|>"
        )
        return text
    
    def encode_text(self, text, max_length=None):
        """
        Encode text to embeddings with proper embed_mask handling.
        
        Args:
            text (str or list): Text(s) to encode
            max_length (int, optional): Maximum sequence length
        
        Returns:
            torch.Tensor: Text embeddings
        """
        if max_length is None:
            max_length = getattr(self, 'max_length', 512)
        
        inputs = self.tokenizer(
            text, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=max_length
        )
        
        # Add embed_mask (same as attention_mask for simple text encoding)
        inputs["embed_mask"] = inputs["attention_mask"].clone()
        
        # Move to same device as model and ensure proper dtype
        import torch
        if hasattr(self, 'device') and self.device is not None:
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Ensure proper dtype for floating point tensors
        model_dtype = next(self.parameters()).dtype
        for key in inputs:
            if inputs[key].dtype.is_floating_point:
                inputs[key] = inputs[key].to(model_dtype)
        
        with torch.no_grad():
            embeddings = self(inputs)
        
        return embeddings
    
    def tokenize_with_separator(self, texts, max_length=None, separator='!@#$%^&*()'):
        """
        Tokenize texts with special handling for separator-based splitting.
        This is useful for instruction-following tasks.
        
        Args:
            texts (list): List of texts to tokenize
            max_length (int, optional): Maximum sequence length  
            separator (str): Separator to split instruction from text
        
        Returns:
            dict: Tokenized inputs with attention masks and embed masks
        """
        if max_length is None:
            max_length = getattr(self, 'max_length', 512)
            
        texts_2 = []
        original_texts = []
        
        for text in texts:
            parts = text.split(separator)
            texts_2.append(parts[1] if len(parts) > 1 else "")
            original_texts.append("".join(parts))

        # Tokenize original texts
        tokenized = self.tokenizer(
            original_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        
        # Create embedding masks for the separated parts
        import torch
        embed_mask = None
        for t_i, t in enumerate(texts_2):
            ids = self.tokenizer(
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
    
    def encode_with_instruction(self, texts, max_length=None, separator='!@#$%^&*()'):
        """
        Encode texts with instruction-following using separator-based processing.
        
        Args:
            texts (list): List of texts with instructions separated by separator
            max_length (int, optional): Maximum sequence length
            separator (str): Separator between instruction and text
        
        Returns:
            torch.Tensor: Text embeddings
        """
        tokenized = self.tokenize_with_separator(texts, max_length, separator)
        
        # Move to same device as model and ensure proper dtype
        import torch
        if hasattr(self, 'device') and self.device is not None:
            tokenized = {k: v.to(self.device) for k, v in tokenized.items()}
        
        # Ensure proper dtype for floating point tensors
        model_dtype = next(self.parameters()).dtype
        for key in tokenized:
            if tokenized[key].dtype.is_floating_point:
                tokenized[key] = tokenized[key].to(model_dtype)
        
        with torch.no_grad():
            embeddings = self(tokenized)
        
        return embeddings

    @classmethod
    def from_pretrained(
        cls,
        base_model_name_or_path,
        peft_model_name_or_path=None,
        merge_peft=False,
        enable_bidirectional=True,
        extra_model_name_or_path=None,
        **kwargs,
    ):
        # pop out encoder args
        keys = ["pooling_mode", "max_length", "doc_max_length", "skip_instruction"]
        encoder_args = {
            key: kwargs.pop(key, None) for key in keys if kwargs.get(key) is not None
        }

        tokenizer = AutoTokenizer.from_pretrained(base_model_name_or_path)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

        config = AutoConfig.from_pretrained(base_model_name_or_path)
        config_class_name = config.__class__.__name__

        model_class = cls._get_model_class(
            config_class_name, enable_bidirectional=enable_bidirectional
        )
        model = model_class.from_pretrained(base_model_name_or_path, **kwargs)

        if os.path.isdir(base_model_name_or_path) and os.path.exists(
            f"{base_model_name_or_path}/config.json"
        ):
            with open(f"{base_model_name_or_path}/config.json", "r") as fIn:
                config_dict = json.load(fIn)
            config = PretrainedConfig.from_dict(config_dict)
            model.config._name_or_path = config._name_or_path

        # For special case where config.json and adapter weights are in the same directory
        if hasattr(model, "peft_config"):
            model = PeftModel.from_pretrained(
                model,
                base_model_name_or_path,
            )
            model = model.merge_and_unload()

        if peft_model_name_or_path is not None:
            model = PeftModel.from_pretrained(
                model,
                peft_model_name_or_path,
            )
            if merge_peft:
                model = model.merge_and_unload()
        if extra_model_name_or_path is not None:
            logger.info(f"Loading extra model from {extra_model_name_or_path}")
            if not merge_peft:
                model = model.merge_and_unload()
            if isinstance(extra_model_name_or_path, str):
                model = PeftModel.from_pretrained(
                    model,
                    extra_model_name_or_path,
                )
                model = model.merge_and_unload()
            elif isinstance(extra_model_name_or_path, list):
                for extra_model in extra_model_name_or_path:
                    model = PeftModel.from_pretrained(
                        model,
                        extra_model,
                    )
                    peft_model_name_or_path = extra_model
                    model = model.merge_and_unload()
            else:
                raise ValueError(
                    f"extra_model_name_or_path should be a string or a list of strings."
                )
        config = {}
        config_addr = (
            peft_model_name_or_path
            if peft_model_name_or_path is not None
            else base_model_name_or_path
        )
        if os.path.exists(f"{config_addr}/llm2vec_config.json"):
            with open(f"{config_addr}/llm2vec_config.json", "r") as fIn:
                llm2vec_config = json.load(fIn)
            config.update(llm2vec_config)

        for key, value in encoder_args.items():
            config[key] = value

        return cls(model=model, tokenizer=tokenizer, **config)