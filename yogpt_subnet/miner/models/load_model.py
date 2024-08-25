from transformers import (AutoModel, AutoModelForCausalLM,
                          AutoModelForSeq2SeqLM,
                          AutoModelForSequenceClassification, AutoTokenizer)


def load_model(model_id, model_type="default"):
    """
    Loads a model and its tokenizer from Hugging Face based on the model type.

    Args:
    model_id (str): Identifier for the model on Hugging Face Hub.
    model_type (str): Type of the model to load (e.g., 'causal', 'seq2seq', 'classification', 'default').

    Returns:
    tuple: A tuple containing the model and its tokenizer.
    """
    # Select the model type
    if model_type == "causal":
        model = AutoModelForCausalLM.from_pretrained(model_id)
    elif model_type == "seq2seq":
        model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
    elif model_type == "classification":
        model = AutoModelForSequenceClassification.from_pretrained(model_id)
    else:
        model = AutoModel.from_pretrained(model_id)

    # Always load the tokenizer for the model
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    return model, tokenizer
