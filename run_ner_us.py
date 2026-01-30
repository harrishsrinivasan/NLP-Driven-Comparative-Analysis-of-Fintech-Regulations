import os
import sys
import pandas as pd

# Ensure 'finreg-nlp-app' is on path so we can import the modules package
sys.path.append(os.path.join(os.path.dirname(__file__), "finreg-nlp-app"))

from modules.ner_helper import extract_ner_entities
from modules.pdf_processor import load_spacy_model


def main():
    csv_in = os.path.join("cleaned_law", "cleaned_us_law.csv")
    if not os.path.exists(csv_in):
        print(f"Input file not found: {csv_in}")
        return

    df = pd.read_csv(csv_in)
    if 'cleaned_text' not in df.columns:
        print("cleaned_text column not found in CSV")
        return

    text = " ".join(df['cleaned_text'].dropna().astype(str).tolist())

    # Preload model once and pass into extractor
    nlp = load_spacy_model()
    ents = extract_ner_entities(text, top_n=50, nlp=nlp)

    print(ents.head(50))

    out = os.path.join("cleaned_law", "us_entities.csv")
    ents.to_csv(out, index=False)
    print(f"Saved US entities to {out}")


if __name__ == '__main__':
    main()
