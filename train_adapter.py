from transformers import Trainer, TrainingArguments
from TrainableHuggingfaceChatbot import TrainableHuggingfaceChatbot

MODEL_NAME = "meta-llama/Llama-2-7b-hf"
DOMAIN = "GDPR"
N_SPLITS = 5
MAX_LENGTH = 1024

LAMBDA_CI = 0.6
LAMBDA_VERDICT = 0.4

SPECIAL_TOKENS = ["<CI>", "</CI>", "<VERDICT>", "</VERDICT>"]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

if __name__ == '__main__':
    chatbot = TrainableHuggingfaceChatbot(
        model="TinyLlama/TinyLlama-1.1B-Chat"
    )
    

