def tokenize_data(tokenizer, dataset, model_type):
    """Tokenizes dataset using the provided tokenizer, adjusting based on the model type."""
    
    def bert_tokenize(examples):
        # BERT typically requires CLS and SEP tokens
        return tokenizer(["[CLS] " + text + " [SEP]" for text in examples['text']], truncation=True, padding="max_length")

    def gpt_tokenize(examples):
        # GPT doesn't use special CLS or SEP tokens
        return tokenizer(examples['text'], truncation=True, padding="max_length")

    def t5_tokenize(examples):
        # T5 uses a prefix indicating the task
        return tokenizer(["summarize: " + text for text in examples['text']], truncation=True, padding="max_length")

    def llama_tokenize(examples):
        # LLaMA tokenization can be similar to BERT or customized
        return tokenizer(examples['text'], truncation=True, padding="max_length")

    # Map the appropriate tokenization function based on the model type
    if model_type == "bert":
        return dataset.map(bert_tokenize, batched=True)
    elif model_type == "gpt":
        return dataset.map(gpt_tokenize, batched=True)
    elif model_type == "t5":
        return dataset.map(t5_tokenize, batched=True)
    elif model_type == "llama":
        return dataset.map(llama_tokenize, batched=True)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
