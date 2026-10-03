"""Spanish-friendly text normalization and tokenization."""

import re
import unicodedata

STOPWORDS = set(
    """a al algo algun alguna alguno ante antes como con contra cual cuales cuando cuanto cuantos cuantas
    de del desde donde dos el ella ellos en entre era es esa ese eso esta este esto estos hay la las le les
    lo los mas me mi mis mucho muy no nos o para pero por que quien se segun ser si sin sobre su sus
    tambien te tiene tienen todo un una uno unos y ya hoy cada puede pueden""".split()
)


def strip_accents(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn")


def _stem(token: str) -> str:
    """Very light Spanish stemming, enough for keyword matching.

    "cotizaciones", "cotiza" -> "cotiz"; "guantes", "guante" -> "guant".
    """
    if token.isdigit():
        return token
    for suffix in ("aciones", "acion", "mente", "es", "s"):
        if len(token) > len(suffix) + 3 and token.endswith(suffix):
            token = token[: -len(suffix)]
            break
    if len(token) > 4 and token[-1] in "aeo":
        token = token[:-1]
    return token


def tokenize(text: str) -> list[str]:
    text = strip_accents(text.lower())
    tokens = re.findall(r"[a-z0-9]+", text)
    return [_stem(t) for t in tokens if t not in STOPWORDS and len(t) > 1]
