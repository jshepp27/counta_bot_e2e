import json
import spacy
from spacy.matcher import PhraseMatcher
import random
from pathlib import Path

# TODOs: Handel multiple Keyphrases
# TODOs: Provide KP as input

import utils
from utils.keyphrase_extraction import extract_keyphrase

nlp = spacy.load("en_core_web_sm")

### SENTIMENT LEXICONS ###
pos = [w.replace("\n", "") for w in open("../../data/lexicon/positive_lex.txt")]
neg = [w.replace("\n", "") for w in open("../../data/lexicon/negative_lex.txt")]

### STANCE-CLAIM DETECTION ###
# def sentence_stance(sentence, target):
def sentence_stance(sentence, target=None):
    sentence = nlp(sentence)
    
    if target == None:
        # Extract Aspect
        aspect = extract_keyphrase(str(sentence), n_gram=3)[0]
        aspect = nlp(aspect)

    else: aspect = nlp(target)

    compound_word = ""
    for i in aspect:
        if i.pos_ in ["NOUN", "PROPN"]:
            comps = "".join([str(j) for j in i.children if j.dep_ == "compound"])
            if comps:
                compound_word = comps + " " + str(i)

    pos_score = 0.0
    neg_score = 0.0

    # Pattern Match
    phrase_matcher = PhraseMatcher(nlp.vocab)

    patterns = [nlp(compound_word)]
    phrase_matcher.add("phrases", None, *patterns)

    start = 0
    stop = 0
    matched_phrases = phrase_matcher(sentence)
    for i, j, k in matched_phrases:
        start = j
        stop = k

    for idx, tok in enumerate(sentence):

        if idx == start or idx == stop - 1:
            continue

        # Polarity Shift
        # NEAR parameter, k
        k = 5
        if tok.dep_ == "neg":
            if tok.text in pos:
                # Shift to Negative
                if idx <= k:
                    if idx < start: neg_score += 1/(start - idx)
                    else: neg_score += 1/(idx - stop)**0.5
        
            if str(tok.head.text) in neg:
                # Shift to Positive
                if idx < start: pos_score += 1/(start - idx)
                else: pos_score += 1/(idx - stop)**0.5

        if str(tok.text) in pos:
            if idx < start: pos_score += 1/(start - idx)
            else: pos_score += 1/(idx - stop)**0.5

        if str(tok.text) in neg:
            if idx < start: neg_score += 1/(start - idx)
            else: neg_score += 1/(idx - stop)**0.5

    result = pos_score - neg_score /(pos_score - neg_score + 1)
    stance = ""

    neg_score, pos_score
    stance = {"claim": sentence, "stance": "PRO", "aspect": aspect} if result > 0 else {"claim": sentence, "stance": "CON", "aspect": aspect}

    return stance

### TEST STATEMENT ###

id = random.randint(0, 1000)
claim =  nlp("I do not believe abortion should be legal")

print(sentence_stance(claim))