import os
import re
import logging
import unicodedata
from html.parser import HTMLParser
import emoji
import unidecode
from spacy.lang.en import English
from spacy.language import Language

logger = logging.getLogger(__name__)

@Language.component("avoid_sentencizer_hashtags")
def _avoid_sentence_boundary_on_hashtag(doc):
    for token in doc[:-1]:
        if token.text == '#':
            doc[token.i+1].is_sent_start = False
    return doc

# build spacy model
def build_spacy_model():
    nlp = English()
    #sentencizer = nlp.create_pipe("sentencizer")
    #nlp.add_pipe(sentencizer)
    nlp.add_pipe("sentencizer")
    nlp.add_pipe("avoid_sentencizer_hashtags")
    return nlp
nlp = build_spacy_model()
# compile regexes
username_regex = re.compile(r'(^|[^@\w])@(\w{1,15})\b')
url_regex = re.compile(r'((www\.[^\s]+)|(https?://[^\s]+)|(http?://[^\s]+))')
control_char_regex = re.compile(r'[\r\n\t]+')
# translate table for punctuation
transl_table = dict([(ord(x), ord(y)) for x, y in zip(u"‘’´“”–-",  u"'''\"\"--")])
# HTML parser
html_parser = HTMLParser()


def preprocess_bert(text, args, do_lower_case=True):
    """Preprocesses tweet for BERT"""
    # standardize
    text = standardize_text(text)
    # replace usernames/urls
    if args.replace_usernames:
        text = replace_usernames(text, filler=args.username_filler)
    if args.replace_urls:
        text = replace_urls(text, filler=args.url_filler)
    if args.asciify_emojis:
        text = asciify_emojis(text)
    if args.standardize_punctuation:
        text = standardize_punctuation(text)
    if do_lower_case:
        text = text.lower()
    if args.replace_multiple_usernames:
        text = replace_multi_occurrences(text, args.username_filler)
    if args.replace_multiple_urls:
        text = replace_multi_occurrences(text, args.url_filler)
    if args.remove_unicode_symbols:
        text = remove_unicode_symbols(text)
    if args.remove_accented_characters:
        text = remove_accented_characters(text)
    return text

def remove_accented_characters(text):
    text = unidecode.unidecode(text)
    return text

def remove_unicode_symbols(text):
    text = ''.join(ch for ch in text if unicodedata.category(ch)[0] != 'So')
    return text

def replace_multi_occurrences(text, filler):
    """Replaces multiple occurrences of filler with n filler"""
    # only run if we have multiple occurrences of filler
    if text.count(filler) <= 1:
        return text
    # pad fillers with whitespace
    text = text.replace(f'{filler}', f' {filler} ')
    # remove introduced duplicate whitespaces
    text = ' '.join(text.split())
    # find indices of occurrences
    indices = []
    for m in re.finditer(r'{}'.format(filler), text):
        index = m.start()
        indices.append(index)
    # collect merge list
    merge_list = []
    for i, index in enumerate(indices):
        if i > 0 and index - old_index == len(filler) + 1:
            # found two consecutive fillers
            if len(merge_list) > 0 and merge_list[-1][1] == old_index:
                # extend previous item
                merge_list[-1][1] = index
                merge_list[-1][2] += 1
            else:
                # create new item
                merge_list.append([old_index, index, 2])
        old_index = index
    # merge occurrences
    if len(merge_list) > 0:
        new_text = ''
        pos = 0
        for (start, end, count) in merge_list:
            new_text += text[pos:start]
            new_text += f'{count} {filler}'
            pos = end + len(filler)
        new_text += text[pos:]
        text = new_text
    return text

def segment_sentences(text, args):
    """Uses spacy to segment text into sentences. Sentences which only consist of a filler will be merged with previous or following sentences"""
    doc = nlp(text)
    regex_fillers = r'(^\d {username}$)|^{username}$|(^\d {url}$)|^{url}$'.format(username=args.username_filler, url=args.url_filler)
    num_tokens = len(doc)
    sentences = [s.string.strip() for s in doc.sents]
    for i, sentence in enumerate(sentences):
        if re.match(regex_fillers, sentence):
            if i == 0 and len(sentences) > 1:
                # prepend to next sentence
                sentences[i+1] = f'{sentence} {sentences[i+1]}'
            elif i > 0:
                # add sentence to previous
                sentences[i-1] += f' {sentence}'
            # remove current
            del sentences[i]
    return sentences, num_tokens

def asciify_emojis(text):
    """
    Converts emojis into text aliases. E.g. 👍 becomes :thumbs_up:
    For a full list of text aliases see: https://www.webfx.com/tools/emoji-cheat-sheet/
    """
    text = emoji.demojize(text)
    return text

def standardize_text(text):
    """
    1) Escape HTML
    2) Replaces some non-standard punctuation with standard versions. 
    3) Replace \r, \n and \t with white spaces
    4) Removes all other control characters and the NULL byte
    5) Removes duplicate white spaces
    """
    # escape HTML symbols
    text = html_parser.unescape(text)
    # standardize punctuation
    text = text.translate(transl_table)
    text = text.replace('…', '...')
    # replace \t, \n and \r characters by a whitespace
    text = re.sub(control_char_regex, ' ', text)
    # remove all remaining control characters
    text = ''.join(ch for ch in text if unicodedata.category(ch)[0] != 'C')
    # replace multiple spaces with single space
    text = ' '.join(text.split())
    return text.strip()

def standardize_punctuation(text):
    return ''.join([unidecode.unidecode(t) if unicodedata.category(t)[0] == 'P' else t for t in text])

def replace_usernames(text, filler='user'):
    # @<user> is a marker used internally. use filler instead
    text = text.replace('@<user>', f'{filler}')
    # replace other user handles by filler
    text = re.sub(username_regex, filler, text)
    # add spaces between, and remove double spaces again
    text = text.replace(filler, f' {filler} ')
    text = ' '.join(text.split())
    return text

def replace_urls(text, filler='url'):
    # <url> is a marker used internally. use filler instead
    text = text.replace('<url>', filler)
    # replace other urls by filler
    text = re.sub(url_regex, filler, text)
    # add spaces between, and remove double spaces again
    text = text.replace(filler, f' {filler} ')
    text = ' '.join(text.split())
    return text
