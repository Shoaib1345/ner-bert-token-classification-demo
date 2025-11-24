import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, AutoConfig

def perform_ner(text, model_name="dbmdz/bert-large-cased-finetuned-conll03-english"):
    try:
        # Load model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForTokenClassification.from_pretrained(model_name)
        config = AutoConfig.from_pretrained(model_name)

        # Tokenize input
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)

        # Get predictions
        with torch.no_grad():
            outputs = model(**inputs)
        predictions = torch.argmax(outputs.logits, dim=2)

        # Get tokens and their predictions
        tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        token_labels = [config.id2label[p.item()] for p in predictions[0]]

        # Process results, handling special tokens and subwords
        results = []
        current_entity = []
        current_label = None

        # Skip [CLS] and [SEP]
        for token, label in zip(tokens[1:-1], token_labels[1:-1]):
            # Handle subwords
            if token.startswith("##"):
                if current_entity:
                    current_entity[-1] += token[2:]
                continue

            # Handle entity continuation
            if label.startswith("B-") or label == "O":
                if current_entity:
                    results.append((" ".join(current_entity), current_label))
                    current_entity = []
                # Remove B- prefix
                if label != "O":
                    current_entity = [token]
                    current_label = label[2:]
            elif label.startswith("I-"):
                if not current_entity:
                    current_entity = [token]
                    current_label = label[2:]
                else:
                    current_entity.append(token)

        # Add final entity if exists
        if current_entity:
            results.append((" ".join(current_entity), current_label))

        return results

    except Exception as e:
        print(f"Error performing NER: {str(e)}")
        return []

text = """OpenAI, headquartered in San Francisco, California, released its first major language model series in 2018. Over the years, it has collaborated with organizations such as Microsoft to advance research in artificial intelligence. In 2023, CEO Sam Altman announced several breakthroughs in multimodal AI systems, positioning the company as a global leader in the field. The organization continues to invest heavily in safety, ethics, and responsible AI development."""

results = perform_ner(text)
print(results)
