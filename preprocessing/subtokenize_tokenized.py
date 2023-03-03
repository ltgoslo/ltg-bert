from sacremoses import MosesPunctNormalizer
import ftfy

mpn = MosesPunctNormalizer(lang="en", penn=False)


def normalize_ptb(token):
    if token == "-LRB-":
        return '('
    elif token == "-RRB-":
        return ')'
    elif token in ["''", "``"]:
        return '"'

    return token


def normalize_abbreviations(text):
    text = text.replace(" n't ", "n't ")
    text = text.replace(" N'T ", "N'T ")
    text = text.replace(" 'll ", "'ll ")
    text = text.replace(" 'LL ", "'LL ")
    text = text.replace(" 're ", "'re ")
    text = text.replace(" 'RE ", "'RE ")
    text = text.replace(" 've ", "'ve ")
    text = text.replace(" 'VE ", "'VE ")
    text = text.replace(" 'm ", "'m ")
    text = text.replace(" 'M ", "'M ")
    text = text.replace(" 's ", "'s ")
    text = text.replace(" 'S ", "'S ")
    text = text.replace(" 'd ", "'d ")
    text = text.replace(" 'D ", "'D ")
    return text


def normalize(token):
    token = token.strip()
    token = normalize_ptb(token)
    token = mpn.normalize(token)
    token = ftfy.fix_text(token)
    token = token.strip()
    return token


def fix_quotes(text, quote_symbol='"'):
    n_quotes = text.count(f" {quote_symbol}") + text.count(f"{quote_symbol} ") - text.count(f" {quote_symbol} ")
    if (
        n_quotes == 0
        or (n_quotes % 2) == 1
        or f"{quote_symbol}{quote_symbol}" in text
        or f"{quote_symbol} {quote_symbol}" in text
    ):
        return text

    i, i_quote, n_changes = 0, 0, 0
    while i < len(text):
        if text[i] != quote_symbol or (i - 1 >= 0 and text[i - 1] != ' ' and i + 1 < len(text) and text[i + 1] != ' '):
            i += 1
            continue

        if (i_quote % 2) == 0:
            if i > 0 and text[i - 1] != ' ':
                text = text[:i] + ' ' + text[i:]
                i += 1
                n_changes += 1
            if i + 1 < len(text) and text[i + 1] == ' ':
                text = text[:i + 1] + text[i + 2:]
                n_changes += 1
        else:
            if i > 0 and text[i - 1] == ' ':
                text = text[:i - 1] + text[i:]
                i -= 1
                n_changes += 1
            if i + 1 < len(text) and text[i + 1].isalnum():
                text = text[:i + 1] + ' ' + text[i + 1:]
                n_changes += 1

        i_quote += 1
        i += 1

    return text


def detokenize(tokens, compact_dashes=False):
    text = ' '.join(tokens)
    text = normalize_abbreviations(text)

    if compact_dashes:
        text = text.replace(' - ', '-')

    for i in range(len(text) - 2, -1, -1):
        if text[i] == '.' and (text[i + 1].isupper() or text[i + 1] in ['‘', '(', '[', '{']):
            text = text[:i+1] + ' ' + text[i+1:]
        elif text[i] in ['?', '!', '…', '’'] and (text[i + 1].isalnum() or text[i + 1] in ['‘', '(', '[', '{']):
            text = text[:i+1] + ' ' + text[i+1:]
        elif i > 2 and text[i] == '.' and text[i - 1] == '.' and text[i - 2] == '.' and text[i + 1] != ' ':
            text = text[:i+1] + ' ' + text[i+1:]
        elif i > 2 and text[i] == '.' and text[i - 1] == '.' and text[i - 2] == '.' and text[i + 1] != ' ':
            text = text[:i+1] + ' ' + text[i+1:]
        elif text[i] == ',' and (text[i + 1].isalpha() or text[i + 1] in ['‘', '(', '[', '{']):
            text = text[:i+1] + ' ' + text[i+1:]
        elif text[i] in [';', ')', ']', '}', '%'] and (text[i + 1].isalnum() or text[i + 1] in ['‘', '(', '[', '{']):
            text = text[:i+1] + ' ' + text[i+1:]
        elif text[i] == ':' and (text[i + 1] in ['‘', '(', '[', '{'] or (text[i + 1].isalnum() and (not text[i + 1].isnumeric() or i - 1 < 0 or not text[i - 1].isnumeric()))):
            text = text[:i+1] + ' ' + text[i+1:]
        elif text[i] in ['(', '[', '{'] and text[i + 1] == ' ':
            text = text[:i+1] + text[i+2:]
        elif text[i] == ' ' and text[i+1] in ['.', ';', ':', '?', '!', '…', ',', '’', ')', ']']:
            text = text[:i] + text[i+1:]
        elif i > 0 and text[i] == ' ' and text[i - 1] in ['$', '£', '€'] and text[i + 1].isnumeric():
            text = text[:i] + text[i+1:]
        elif i > 0 and text[i] == ' ' and text[i - 1].isnumeric() and text[i + 1] == '%':
            text = text[:i] + text[i+1:]

    text = fix_quotes(text, '"')
    text = fix_quotes(text, "'")

    spans = []
    word_offset, char_offset = 0, 0
    for i, ch in enumerate(text):
        if ch == ' ':
            if tokens[word_offset][char_offset] == ' ':
                char_offset += 1
            continue

        assert ch == tokens[word_offset][char_offset], f"{text}\n{' '.join(tokens)}\n{tokens[word_offset]}\n{char_offset} {ch}"

        if char_offset == 0:
            start = i

        if char_offset == len(tokens[word_offset]) - 1:
            end = i + 1
            spans.append((start, end))
            word_offset += 1
            char_offset = 0
        else:
            char_offset += 1

    return text, spans


def calculate_spans(original_spans, encoding_offsets):
    span_id = 0
    subword_spans = [[] for _ in original_spans]
    for i, (_, end) in enumerate(encoding_offsets):
        subword_spans[span_id].append(i)

        while original_spans[span_id][1] <= end:
            span_id += 1
            if span_id < len(original_spans) and end > original_spans[span_id][0]:
                subword_spans[span_id].append(i)

            if span_id == len(original_spans):
                return subword_spans

    return subword_spans


def subtokenize(tokens, tokenizer, compact_dashes=False):
    tokens = [normalize(token) for token in tokens]
    text, spans = detokenize(tokens, compact_dashes=compact_dashes)

    encoding = tokenizer.encode(text, add_special_tokens=False)
    spans = calculate_spans(spans, encoding.offsets)
    subwords = encoding.ids

    return subwords, spans


# tokens = ["(", "Chang", "Chiung", "-", "fang", "/", "tr", ".", "by", "David", "Mayer", ")", "I", "'m", "very", "nercous", "."]
# tokens = [normalize(token) for token in tokens]
# text, spans = detokenize(tokens)
# print(text)
