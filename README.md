Named Entity Recognition (NER) using BERT (Transformers)

This repository contains a Python-based Named Entity Recognition (NER) tool built using HuggingFace Transformers, specifically the model:

dbmdz/bert-large-cased-finetuned-conll03-english


The project performs token classification to identify entities such as:

PERSON

ORGANIZATION

LOCATION

MISC

It also includes logic to merge BERT subword tokens (e.g., San + ##Francisco → San Francisco) and extract clean entity strings.

🚀 Features

Performs NER on any user-provided text

Uses a fine-tuned BERT model trained on CoNLL-2003 dataset

Handles:

Subword merging (## tokens)

BIO tagging

Extraction of full entity phrases

Single function: perform_ner()

Supports custom model names

📦 Installation
1. Clone the repository
git clone https://github.com/your-username/bert-ner-extractor.git
cd bert-ner-extractor

Contributing

Pull requests are welcome! Feel free to submit improvements, bug fixes, or new features.

📜 License

This project is licensed under the MIT License.

3. Install dependencies
pip install torch transformers
