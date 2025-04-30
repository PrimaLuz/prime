from dataclasses import dataclass
from zeroband.config import Config



@dataclass
class TokenizerInfo:
    hf_name: str  # TODO: REMOVE

    vocab_size: int
    bot_token: int
    eot_token: int


def get_tokenizer_info(config: Config) -> TokenizerInfo:
    if config.data.fake and config.model_name == "debugmodel":
        from zeroband.data import DEBUG_VOCAB_SIZE
        return TokenizerInfo(
            hf_name="debugmodel",
            vocab_size=DEBUG_VOCAB_SIZE,
            bot_token=1,
            eot_token=2,
        )
    
    elif config.model_type == "llama2":
        return TokenizerInfo(
            hf_name="mistralai/Mistral-7B-v0.1",
            # print(len(AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1", use_fast=True)))
            vocab_size=32000,
            # print(AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1", use_fast=True).bos_token_id)
            bot_token=1,
            # print(AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1", use_fast=True).eos_token_id)
            eot_token=2,
        )
    elif config.model_type == "llama3":
        return TokenizerInfo(
            hf_name="meta-llama/Meta-Llama-3-8B",
            # print(len(AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B", use_fast=True)))
            vocab_size=128256,
            # print(AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1", use_fast=True).bos_token_id)
            bot_token=128000,
            # print(AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B", use_fast=True).eos_token_id)
            eot_token=128001,
        )
    else:
        raise ValueError(f"Model type {config.model_type} not supported")
