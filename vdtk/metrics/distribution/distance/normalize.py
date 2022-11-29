

from nltk.tokenize.treebank import TreebankWordTokenizer

_TOKENIZER = TreebankWordTokenizer()
_PUNCT_TRANSLATIONS = ''.join([
    '’', # Replace curly apostrophe
    '‘', # Replace curly apostrophe
    '“', # Replace curly quotation mark
    '”', # Replace curly quotation mark
    '…', # Replace ellipsis
    '—', # Replace em dash
    '–', # Replace en dash
    '−', # Replace minus sign
    '‐', # Replace hyphen
    '‑', # Replace non-breaking hyphen
    '‒', # Replace figure dash
    '⁃', # Replace hyphen bullet
    '⁻', # Replace reversed hyphen bullet
    '₋', # Replace hyphen-minus
    '−', # Replace minus sign
    '﹣', # Replace fullwidth hyphen-minus
    '－', # Replace fullwidth hyphen-minus
    '‐', # Replace hyphen
    '-', # Replace hyphen
    '.', # Replace period
    ',', # Replace comma
    ';', # Replace semicolon
    ':', # Replace colon
    '!', # Replace exclamation mark
    '?', # Replace question mark
])
_FILTER = ['-lrb-', '-rrb-', '-lcb-', '-rcb-', '-lsb-', '-rsb-']

def coco_normalize(string: str) -> str:
    # Normalize and tokenize string using COCO approach
    tokens = _TOKENIZER.tokenize(string, convert_parentheses=True)

    # Lower-Case
    tokens = [s.lower() for s in tokens]
    # Filter LRB/RRB, LCB/RCB, LSB/RSB
    tokens = [t for t in tokens if t and t not in _FILTER]
    # Remove punctuation
    tokens = [t.translate(str.maketrans('', '', _PUNCT_TRANSLATIONS)) for t in tokens]
    # Remove empty tokens
    tokens = [t for t in tokens if t]

    return ' '.join(tokens)


if __name__ == '__main__':
    print(coco_normalize("This is a (test) string . ?!"))
