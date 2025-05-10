"""
Description:
    Llama model loader for Hugging Face Transformers.
"""

def load_llama_model(model_name="meta-llama/Llama-3.2-1B", device="cpu"):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    config = model.config
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print("Loaded model:", model_name)
    
    return model, config, tokenizer