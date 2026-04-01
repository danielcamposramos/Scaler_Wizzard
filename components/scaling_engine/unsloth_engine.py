"""Unsloth-powered training engine for Scaler Wizard.

Provides 2x speed and 70% memory reduction for TRMC model training (continuation pre-training).
"""

try:
    from unsloth import FastLanguageModel # type: ignore
    from transformers import TrainingArguments, Trainer, DataCollatorWithPadding # type: ignore
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:
    FastLanguageModel = None

class TRMCDataCollator(DataCollatorWithPadding):
    """Custom collator that prepares both positive and negative signals for the Two-Way loss."""
    def __call__(self, features):
        # Separate features into positive and negative sets
        pos_features = [{"input_ids": f["input_ids"], "attention_mask": f["attention_mask"]} for f in features]
        batch = super().__call__(pos_features)
        
        # Handle negative signals if they exist in the batch
        if "neg_input_ids" in features[0] and features[0]["neg_input_ids"] is not None:
            neg_features = [{"input_ids": f["neg_input_ids"], "attention_mask": f["neg_attention_mask"]} for f in features]
            neg_batch = super().__call__(neg_features)
            batch["neg_input_ids"] = neg_batch["input_ids"]
            batch["neg_attention_mask"] = neg_batch["attention_mask"]
        
        return batch

class TRMCTrainer(Trainer):
    """Custom Trainer implementing Two-Way Contrastive Loss for ground-up pre-training."""
    def compute_loss(self, model, inputs, return_outputs=False):
        # 1. Positive Signal (Standard Causal LM)
        outputs = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], labels=inputs["input_ids"])
        pos_loss = outputs.loss
        pos_logits = outputs.logits

        # 2. Negative Signal (Contrastive Aspect)
        if "neg_input_ids" in inputs:
            # Get logits for the negative sequence
            neg_outputs = model(input_ids=inputs["neg_input_ids"], attention_mask=inputs["neg_attention_mask"])
            neg_logits = neg_outputs.logits
            
            # Calculate log-probabilities for comparison (Two-Way Statistics)
            # We compare the average log-prob of the correct sequence vs incorrect sequence
            pos_logps = F.log_softmax(pos_logits, dim=-1).max(dim=-1).values.mean()
            neg_logps = F.log_softmax(neg_logits, dim=-1).max(dim=-1).values.mean()
            
            # Contrastive Margin Loss: push pos log_prob up, neg log_prob down
            # This mirrors the "statistics for true vs signals for falses" logic
            margin = 0.1
            contrastive_loss = F.relu(margin - (pos_logps - neg_logps))
            
            total_loss = pos_loss + contrastive_loss
            
            # Log the two-way statistics for the Cockpit
            self.log({"pos_signal": pos_logps.item(), "neg_signal": neg_logps.item()})
        else:
            total_loss = pos_loss
            
        return (total_loss, outputs) if return_outputs else total_loss

class UnslothEngine:
    """Integrates Unsloth optimized kernels for RoPE and LoRA scaling."""
    dataset_curator = None # Will be initialized externally or passed in

    def __init__(self, model_name: str, max_seq_length: int = 4096):
        if FastLanguageModel is None:
            raise RuntimeError("Unsloth not installed. Please install for speed training.")
        
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name = model_name,
            max_seq_length = max_seq_length,
            load_in_4bit = True, # Recommended for consumer hardware
            dtype = None,        # Auto-detect (Float16 or Bfloat16)
        )

    def apply_fast_lora(self, r: int = 16, alpha: int = 32):
        """Applies Unsloth's optimized LoRA adapters. Typically for fine-tuning, less common for pre-training."""
        if FastLanguageModel: # Ensure Unsloth is loaded
            self.model = FastLanguageModel.get_peft_model(
                self.model,
                r = r,
                target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                                 "gate_proj", "up_proj", "down_proj"],
                lora_alpha = alpha,
                lora_dropout = 0, # Optimized to 0 for Unsloth
                bias = "none",    # Optimized to "none" for Unsloth
                use_gradient_checkpointing = "unsloth", # 0 memory overhead
                random_state = 3407,
                use_rslora = False,
                loftq_config = None,
            )
        return self.model

    def get_trainable_model(self):
        """Returns the patched model ready for high-speed training."""
        return self.model, self.tokenizer

    def run_pretraining(self, dataset_configs: list[dict], args_dict: dict):
        """
        Executes continuation pre-training for the TRMC model.
        Uses a Two-Way Contrastive Loss to distinguish between True and False signals.
        """

        if UnslothEngine.dataset_curator is None:
            from Scaler_Wizzard.components.scaling.dataset_curator import DatasetCurator
            UnslothEngine.dataset_curator = DatasetCurator()
        
        mixed_dataset = UnslothEngine.dataset_curator.create_mixed_dataset(
            {f"ds_{i}": config for i, config in enumerate(dataset_configs)}
        )

        def tokenize_function(examples):
            # Tokenize Positive (True) signal
            tokenized = self.tokenizer(examples["text"], truncation=True, max_length=1024)
            
            # Tokenize Negative (False) signal if present
            if "negative_text" in examples and examples["negative_text"][0] is not None:
                neg_tokenized = self.tokenizer(examples["negative_text"], truncation=True, max_length=1024)
                tokenized["neg_input_ids"] = neg_tokenized["input_ids"]
                tokenized["neg_attention_mask"] = neg_tokenized["attention_mask"]
            else:
                tokenized["neg_input_ids"] = [None] * len(examples["text"])
                tokenized["neg_attention_mask"] = [None] * len(examples["text"])
                
            return tokenized
        
        tokenized_dataset = mixed_dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=["text", "negative_text"]
        )

        training_args = TrainingArguments(
            output_dir = args_dict.get("output_dir", "outputs"),
            learning_rate = args_dict.get("learning_rate", 5e-6),
            lr_scheduler_type = "linear",
            per_device_train_batch_size = args_dict.get("batch_size", 2),
            gradient_accumulation_steps = 4,
            num_train_epochs = args_dict.get("epochs", 50), # Increased for pre-training
            fp16 = not torch.cuda.is_bf16_supported(),
            bf16 = torch.cuda.is_bf16_supported(),
            optim = "adamw_8bit",
            logging_steps = 10,
            save_steps = 500,
            save_total_limit = 3, # Keep only the last 3 checkpoints
        )
        
        trainer = TRMCTrainer(
            model = self.model,
            train_dataset = tokenized_dataset,
            tokenizer = self.tokenizer,
            args = training_args,
            data_collator = TRMCDataCollator(self.tokenizer),
        )
        return trainer.train()