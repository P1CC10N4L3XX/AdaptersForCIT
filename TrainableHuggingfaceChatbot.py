from agents import HuggingfaceChatbot
from peft import LoraConfig, get_peft_model, PeftModel
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
import torch


class TrainableHuggingfaceChatbot(HuggingfaceChatbot):
    def __init__(self, model, adapter_path = None, lora_r = 8, lora_alpha = 32, lora_dropout = 0.05, target_modules = ("q_proj", "v_proj") , max_mem_per_gpu='80Gib'):
        super().__init__(model, max_mem_per_gpu)
        if adapter_path is not None:
            self.model = PeftModel.from_pretrained(
                self.model,
                adapter_path
            )
        else:
            lora_config = LoraConfig(
                r = loar_r,
                lora_alpha = lora_alpha,
                lora_dropout = lora_dropout,
                target_modules = list(target_modules),
                bias = "none",
                task_type = "CAUSAL_LM"
            )
            self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()
        self.model.to(self.device)
        self.model.train(False)
    
        
        