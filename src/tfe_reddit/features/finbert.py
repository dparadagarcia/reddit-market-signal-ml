from __future__ import annotations

from functools import lru_cache

import pandas as pd


@lru_cache(maxsize=2)
def _load_finbert_artifacts(model_name: str):
    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as exc:
        raise ImportError(
            "FinBERT requiere dependencias opcionales. Instala el proyecto con "
            "`python -m pip install -e .[nlp]`."
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()
    return torch, tokenizer, model


def score_financial_sentiment(
    texts: list[str],
    model_name: str = "ProsusAI/finbert",
    batch_size: int = 16,
    max_length: int = 256,
) -> pd.DataFrame:
    """Obtiene probabilidades positiva/negativa/neutral con FinBERT."""
    if not texts:
        return pd.DataFrame(
            columns=[
                "finbert_positive",
                "finbert_negative",
                "finbert_neutral",
                "finbert_sentiment",
            ]
        )

    torch, tokenizer, model = _load_finbert_artifacts(model_name=model_name)

    rows: list[dict[str, float]] = []
    for start in range(0, len(texts), batch_size):
        batch = [text if isinstance(text, str) and text.strip() else "[empty]" for text in texts[start : start + batch_size]]
        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        with torch.no_grad():
            logits = model(**encoded).logits
            probs = torch.softmax(logits, dim=1).cpu().numpy()

        label_map = {int(idx): str(label).lower() for idx, label in model.config.id2label.items()}
        for prob_row in probs:
            values = {"positive": 0.0, "negative": 0.0, "neutral": 0.0}
            for idx, value in enumerate(prob_row):
                label = label_map.get(idx, "")
                if label in values:
                    values[label] = float(value)

            rows.append(
                {
                    "finbert_positive": values["positive"],
                    "finbert_negative": values["negative"],
                    "finbert_neutral": values["neutral"],
                    "finbert_sentiment": values["positive"] - values["negative"],
                }
            )

    return pd.DataFrame(rows)
