from sacremoses import MosesPunctNormalizer
import ftfy


mpn = MosesPunctNormalizer(lang="en", penn=False)


# try to fix, but be gentle and try not to cause more harm
def fix_double_quotes(text):
    n_quotes = text.count('"')
    if n_quotes == 0 or (n_quotes % 2) == 1 or '""' in text or '" "' in text:
        return text

    original_text = text

    i, i_quote, n_changes = 0, 0, 0
    while i < len(text):
        if text[i] != '"':
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

    # too much changes, let's return the original text to play it safe
    if n_changes > 2 and n_changes > n_quotes * 2 / 3:
        return original_text

    return text


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

    text = text.replace(" n't,", "n't,")
    text = text.replace(" N'T,", "N'T,")
    text = text.replace(" 'll,", "'ll,")
    text = text.replace(" 'LL,", "'LL,")
    text = text.replace(" 're,", "'re,")
    text = text.replace(" 'RE,", "'RE,")
    text = text.replace(" 've,", "'ve,")
    text = text.replace(" 'VE,", "'VE,")
    text = text.replace(" 'm,", "'m,")
    text = text.replace(" 'M,", "'M,")
    text = text.replace(" 's,", "'s,")
    text = text.replace(" 'S,", "'S,")
    text = text.replace(" 'd,", "'d,")
    text = text.replace(" 'D,", "'D,")

    text = text.replace(" n't.", "n't.")
    text = text.replace(" N'T.", "N'T.")
    text = text.replace(" 'll.", "'ll.")
    text = text.replace(" 'LL.", "'LL.")
    text = text.replace(" 're.", "'re.")
    text = text.replace(" 'RE.", "'RE.")
    text = text.replace(" 've.", "'ve.")
    text = text.replace(" 'VE.", "'VE.")
    text = text.replace(" 'm.", "'m.")
    text = text.replace(" 'M.", "'M.")
    text = text.replace(" 's.", "'s.")
    text = text.replace(" 'S.", "'S.")
    text = text.replace(" 'd.", "'d.")
    text = text.replace(" 'D.", "'D.")
    return text


def clean(text, minimal=False):
    if not minimal:
        text = add_whitespace(text)
        text = normalize_abbreviations(text)
        text = fix_double_quotes(text)
        text = mpn.normalize(text)

    text = ftfy.fix_text(text)
    text = text.strip()
    return text


def get_sentence(sentence):
    words = word_iterator(sentence)
    if words is None:
        return ''

    text = ''.join(words)
    return text


def add_whitespace(text):
    text = ' '.join(text.replace('\n', "<<NEWLINE/>>").split()).replace("<<NEWLINE/>>", '\n')  # remove excess whitespace
    for i in range(len(text)-2, -1, -1):
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
        elif text[i] == ' ' and text[i+1] in ['.', ';', ':', '?', '!', '…', ',', '’', ')', ']', '}']:
            text = text[:i] + text[i+1:]
        elif i > 0 and text[i] == ' ' and text[i - 1] in ['$', '£', '€'] and text[i + 1].isnumeric():
            text = text[:i] + text[i+1:]
        elif i > 0 and text[i] == ' ' and text[i - 1].isnumeric() and text[i + 1] == '%':
            text = text[:i] + text[i+1:]

    return text


def is_caption(item):
    return item.tag == "p" and "type" in item.attrib and item.attrib["type"].startswith("caption:")


def get_chapter_name(chapter, subchapter=None):
    if "n" in chapter.attrib:
        n = chapter.attrib["n"]
        if subchapter is not None:
            subchapter = subchapter + '.' + str(n)
        else:
            subchapter = str(n)

    if chapter[0].tag == "head":
        return clean(get_sentence(chapter[0])), subchapter

    if is_caption(chapter[0]):
        return clean(get_sentence(chapter[0])), subchapter

    if "type" in chapter.attrib:
        name = chapter.attrib["type"]
        name = name[0].upper() + name[1:]
    else:
        name = ""

    if "n" in chapter.attrib:
        name += ' ' + subchapter

    name = name.strip()
    return name, subchapter


def process_written_part(part, f, prefix="", suffix="", subchapter=None):
    def process_root(part, f, prefix="", suffix="", subchapter=None):
        for i, child in enumerate(part):
            if i == 0 and is_caption(child):
                continue
            process_written_part(child, f, prefix=prefix, suffix=suffix, subchapter=subchapter)
        f.write(prefix + suffix + '\n\n')

    def process_paragraph(part, f, prefix="", suffix="", subchapter=None):
        sentences = []
        for child in part:
            if child.tag == "s":
                sentence = get_sentence(child)
                if len(sentence) > 0:
                    sentences.append(sentence)
            else:
                sentences = clean('\n'.join(sentences))
                if len(sentences) > 0:
                    f.write(prefix + sentences + suffix + '\n\n')
                sentences = []
                process_written_part(child, f, prefix=prefix, suffix=suffix, subchapter=subchapter)

        sentences = clean('\n'.join(sentences))
        if len(sentences) > 0:
            f.write(prefix + sentences + suffix + '\n\n')

    def process_speech(part, f, prefix="", suffix="", subchapter=None):
        name = None
        for i, child in enumerate(part):
            if child.tag == "speaker":
                assert i == 0
                name = clean(get_sentence(child))
            elif child.tag == "stage":
                new_prefix = prefix if name is None else (prefix + name + ": ")
                process_paragraph(child, f, prefix=new_prefix, suffix=suffix, subchapter=subchapter)
            elif child.tag in ["l", "p"]:
                new_prefix = (prefix + "'") if name is None else (prefix + name + ": '")
                process_paragraph(child, f, prefix=new_prefix, suffix="'" + suffix, subchapter=subchapter)
            elif child.tag in ["note", "bibl", "pb", "gap"]:
                pass
            else:
                raise Exception("Unexpected tag in a speech act: " + child.tag)

    def process_list(part, f, prefix="", suffix="", subchapter=None):
        label = None
        for i, child in enumerate(part):
            if child.tag == "head":
                assert i == 0
                name = clean(get_sentence(child))
                f.write(prefix + name + ": " + suffix + '\n\n')
            elif child.tag == "item":
                process_paragraph(child, f, prefix="- " + (label + ' ' if label else '') + prefix, suffix=suffix, subchapter=subchapter)
                label = None
            elif child.tag == "label":
                label = clean(get_sentence(child))
            elif child.tag in ["note", "bibl", "pb", "gap"]:
                pass
            else:
                raise Exception("Unexpected tag in a list: " + child.tag)

    if part.tag == "wtext":
        process_root(part, f, prefix=prefix, suffix=suffix, subchapter=subchapter)

    # chapter
    elif part.tag == "div":
        level = int(part.attrib["level"])
        chapter_name, subchapter = get_chapter_name(part, subchapter)
        f.write(prefix + '#' * (level + 1) + " " + chapter_name + suffix + '\n\n')

        process_root(part, f, prefix=prefix, suffix=suffix, subchapter=subchapter)

    # chapter head, skip it, we've already processed that
    elif part.tag == "head":
        pass

    # paragraph
    elif part.tag == "p":
        process_paragraph(part, f, prefix=prefix, suffix=suffix, subchapter=subchapter)

    # quote
    elif part.tag == "quote":
        prefix = "> " + prefix
        process_paragraph(part, f, prefix=prefix, suffix=suffix, subchapter=subchapter)

    # speech
    elif part.tag == "sp":
        process_speech(part, f, prefix=prefix, suffix=suffix, subchapter=subchapter)

    # line group (group of verses)
    elif part.tag == "lg":
        process_paragraph(part, f, prefix=prefix, suffix=suffix, subchapter=subchapter)

    # line (verse)
    elif part.tag == "l":
        process_paragraph(part, f, prefix=prefix, suffix=suffix, subchapter=subchapter)

    # list
    elif part.tag == "list":
        process_list(part, f, prefix=prefix, suffix=suffix, subchapter=subchapter)

    elif part.tag == "stage":
        print("Stage act shouldn't be here, skipping this sentence and moving on")

    # note, bibliographic reference, page break, ommited part
    elif part.tag in ["note", "bibl", "pb", "gap"]:
        pass

    else:
        raise Exception("Unexpected type of tag: " + part.tag)


def process_spoken_part(part, f, persons):
    def process_root(part, f, persons):
        for i, child in enumerate(part):
            process_spoken_part(child, f, persons)
        f.write('\n\n')

    def process_utterance(part, f, persons):
        if part.attrib["who"] not in persons:
            persons[part.attrib["who"]] = random.choice(first_names)
        speaker = persons[part.attrib["who"]]

        sentences = []
        for child in part:
            if child.tag == "s":
                sentence = get_sentence(child)
                if len(sentence) > 0:
                    sentences.append(sentence)
            elif child.tag in ["event", "pause", "shift", "trunc", "unclear", "vocal", "align", "gap"]:
                pass
            else:
                raise Exception("Unexpected tag in an utterance: " + child.tag)

        sentences = clean('\n'.join(sentences))
        if len(sentences) > 0 and any(ch.isalnum() for ch in sentences):
            f.write(f"{speaker}: '{sentences}'\n\n")

    if part.tag == "stext":
        process_root(part, f, persons)

    # chapter
    elif part.tag == "div":
        level = int(part.attrib["level"]) if "level" in part.attrib else 1

        chapter_name, _ = get_chapter_name(part)
        f.write('#' * (level + 1) + " " + chapter_name + '\n\n')
        process_root(part, f, persons)

    # paragraph
    elif part.tag == "u":
        process_utterance(part, f, persons)

    # note, bibliographic reference, page break, ommited part
    elif part.tag in ["event", "pause", "shift", "trunc", "unclear", "vocal", "align", "gap"]:
        pass

    else:
        raise Exception("Unexpected type of tag: " + part.tag)
