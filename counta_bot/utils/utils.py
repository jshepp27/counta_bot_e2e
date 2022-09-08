import json
import spacy
import uuid
from time import time
from functools import wraps
import re

nlp = spacy.load("en_core_web_sm")

import os
os.path.join(os.path.dirname(__file__))

def sentences_segment(doc):
    return [i for i in re.split(r'(?<=[^A-Z].[.?]) +(?=[A-Z])', doc)]

def tokeniser(doc):
    return re.findall(r"\w+(?:'\w+)?|[^\w\s]", doc)

def paragraphs(document):
    start = 0
    document = nlp(document)
    passages = []
    for token in document:
        if token.is_space and token.text.count("\n") > 1:
            yield document[start:token.i]
            start = token.i
    yield document[start:]


def get_contents(filename):
    """Parse the contents of a file. Each line is a JSON encoded document."""
    documents = []

    with open(filename) as f:
        for line in f:
            doc = json.loads(line)

            if doc["text"] == "": continue
            if not doc: continue

            passages = [str(i) for i in paragraphs(doc["text"])][0].split("\n")

            for passage in passages:
                if len(passage) < 50:
                    continue

                documents.append((str(uuid.uuid4()).replace('-',''), doc['id'], doc["title"], passage))

    return documents

def get_contents_2(filename):
    """Parse the contents of a file. Each line is a JSON encoded document."""
    documents = []

    with open(filename) as f:
        for line in f:
            doc = json.loads(line)

            if doc["text"] == "": continue
            if not doc: continue

            # passages = [str(i) for i in paragraphs(doc["text"])][0].split("\n")
            #
            # for passage in passages:
            #     if len(passage) < 50:
            #         continue

            #documents.append((doc['id'], doc["title"], doc["text"]))
            yield (doc['id'], doc["title"], doc["text"])

    return documents

# NOTE: COPIED
def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        print('func:%r args:[%r, %r] took: %2.4f sec' % \
          (f.__name__, args, kw, te-ts))

        return result
    return wrap
