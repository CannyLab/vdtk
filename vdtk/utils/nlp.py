import logging

import spacy

_SPACY_MODEL_PATHS = {
    "en_core_web_lg": "https://github.com/explosion/spacy-models/releases/download/en_core_web_lg-3.4.1/en_core_web_lg-3.4.1-py3-none-any.whl",
    "en_core_web_sm": "https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.4.1/en_core_web_sm-3.4.1-py3-none-any.whl",
    "en_core_web_md": "https://github.com/explosion/spacy-models/releases/download/en_core_web_md-3.4.1/en_core_web_md-3.4.1-py3-none-any.whl",
    "en_core_web_trf": "https://github.com/explosion/spacy-models/releases/download/en_core_web_trf-3.4.1/en_core_web_trf-3.4.1-py3-none-any.whl",
}


def get_or_download_spacy_model(model: str) -> spacy.language.Language:
    """Get a spacy model, or download it if it doesn't exist"""
    try:
        return spacy.load(model)
    except OSError:
        if model not in _SPACY_MODEL_PATHS:
            raise ValueError(f"Unknown spacy model {model}")
        logging.info(f"Model not found: {model}. Downloading spacy model {model}")
        from pip._internal.cli.main import main as pip_main

        pip_main(["install", _SPACY_MODEL_PATHS[model]])
        return spacy.load(model)
