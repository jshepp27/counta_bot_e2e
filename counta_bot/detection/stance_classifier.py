import json
import spacy
from spacy.matcher import PhraseMatcher
import random
from pathlib import Path

# import sys
# sys.path.append('./')

from counta_bot.utils.keyphrase_extraction import extract_keyphrase
import re

import spacy
from spacy.matcher import PhraseMatcher
from fuzzywuzzy import fuzz, process

#filepath = Path(__file__).parent
nlp = spacy.load("en_core_web_sm")
phrase_matcher = PhraseMatcher(nlp.vocab)

# TODOs: Package as a Module
# TODOs: Handle Negation (Polarity shifters)
# TODOs: Review Unsuperived Approach; Consider adveanced patterns and common-sence knowledge

### STANCE SCORING ###
# TODOs: https://www.cs.uic.edu/~liub/FBS/opinion-mining-final-WSDM.pdf
# TODOs: Pattern based Negation
# TODOs: Semantic Orientation of an opinion (Claim)
# TODOs:Group synonyms of 'features', 'targets'
import os
print(os.getcwd())

### SENTIMENT LEXICONS ###
pos = [w.replace("\n", "") for w in open("../../data/lexicon/positive_lex.txt")]
neg = [w.replace("\n", "") for w in open("../../data/lexicon/positive_lex.txt")]

### STANCE: ASPECT-SEMANTIC ORIENTATION ###
def extract_aspect(sentence, n_gram):
    aspects = extract_keyphrase(str(sentence))[0]

    return nlp(aspects)

def index_aspect(phrase, aspect, sentence):
    # Init NLP Objects
    sentence = nlp(sentence)
    patterns = [nlp(aspect)]

    phrase_matcher.add(phrase, None, *patterns)

    start = 0
    stop = 0

    matched_phrases = phrase_matcher(sentence)
    for i in matched_phrases:
        _, start, stop = i

    return start, stop

# TODOs: Implement Polarity Shifters, Complex, Verb Patterns
def stance_score(start, stop, sentence):
    sentence = nlp(sentence)

    pos_score = 0.0
    neg_score = 0.0

    score = 0
    for idx, tok in enumerate(sentence):
        if idx == start or idx == stop:
            continue

        # TODOs: Implement Polarity Shift
        # TODOs: Experiement with descriptive term + keyphrase aspects
        # TODOs: ABSA https://www.kaggle.com/code/phiitm/aspect-based-sentiment-analysis
        # Use external libaray: Textblob

        k = 8
        # Negation Rules
        if tok.dep_ == "neg":
            if tok.text in pos:
                # Shift to Negative
                if idx <= k:
                    if idx < start:
                        neg_score += 1 / (start - idx)
                    else:
                        neg_score += 1 / (idx - stop) ** 0.5

            if str(tok.head.text) in neg:
                # Shift to Positive
                if idx < start:
                    pos_score += 1 / (start - idx)
                elif idx > start:
                    pos_score += 1 / (idx - stop) ** 0.5
                else:
                    continue

        # Aspect Sentement Orientation
        if str(tok.text) in pos:
            if idx < start:
                pos_score += 1 / (start - idx)
            else:
                pos_score += 1 / (idx - stop) ** 0.5

        if str(tok.text) in neg:
            if idx <= start:
                neg_score += 1 / (start - idx)
            else:
                neg_score += 1 / (idx - stop) ** 0.5

    score = pos_score - neg_score / (pos_score + neg_score + 1)

    return score


def overlap_score(evidence_kp, adu_kp):
    score = 0

    # Split Keyphrase into components, scoring partial units as overlap
    for i in evidence_kp:
        for j in i.split():
            # Ensure string value, to enact .find
            if " ".join(adu_kp).find(j) != -1:
                score += 1
                token = j

            else:
                continue

    return score


def get_overlapping_token(evidence_kp, adu_kp):
    for i in evidence_kp:
        overlap_tokens = []
        for j in i.split():
            if " ".join(adu_kp).find(j) != -1:
                overlap_tokens.append(j)

        return " ".join(i for i in overlap_tokens)


def sentence_stance(adu, aspect):
    aspect = " ".join(i for i in aspect)

    start, stop = index_aspect("aspects", aspect, adu)
    score = stance_score(start, stop, adu)

    return "PRO" if score > 0 else "CON" if score < 0 else "NEUTRAL"


def fuzzy_match(ev_unit, target):
    overlapping_aspect = process.extractOne(ev_unit, target.split())[0]
    score = overlapping_aspect

    overlapping_aspect = re.sub(r'[^\w]', ' ', overlapping_aspect)
    return overlapping_aspect, score


def compare_stance(ev_unit, adu_target):
    # Get the overlapping evidence aspect-target.
    overlapping_target, score = fuzzy_match(ev_unit=ev_unit, target=adu_target)

    # Get position of the overlapping_target
    start, stop = index_aspect("OVERLAP", overlapping_target, ev_unit)

    # Assert Stance towards evidence aspect
    score = stance_score(start, stop, ev_unit)

    return str("PRO") if score > 0 else str("CON")

### TEST STATEMENTS ###

# TODOs: Unit Tests

id = random.randint(0, 1000)
claim =  nlp("I do not believe abortion should be legal")

test_1 = "The mutual trust and understanding you share with your partner will lead to better sex, but that's not the only reason sex can be better when you're in a relationship."
test_1_aspect = " ".join(i for i in extract_keyphrase(test_1, n_kp=1))

test_2 = 'Hello! Let me preface by saying I dont believe there is a better sex.'
test_2_aspect = " ".join(i for i in extract_keyphrase(test_2, n_kp=1))

test_3 = "These simple ideas and techniques could help both you and your lover enjoy sex. Think beyond the thrust."

print(test_1, sentence_stance(test_1, test_1_aspect), test_1_aspect)
print(test_2, compare_stance(test_2, test_1_aspect), test_2_aspect)


