import json
#from pycorenlp import StanfordCoreNLP
#import stanfordnlp
import stanza
from pathlib import Path
import os
import logging
# import sys
# sys.path.append('/path/to/dir')

from counta_bot.utils.keyphrase_extraction import extract_keyphrase
filepath = Path(__file__).parent

logging.basicConfig(level=logging.INFO)

# TODOs: Set Logger Name as Configuration
logger = logging.getLogger("COUNTA-BOT")

stanza.download("en")

# TODOs: Rename Module
# TODOs: Implement tqdm progress bar
def run_nlp():
    nlp = stanza.Pipeline("en", processors='tokenize,pos,lemma')
    data = [ln.strip() for ln in open(filepath/"../data/train_cmv_cleaned.jsonl")]

    fout = open(filepath/"../data/lemma.jsonl", 'w')
    count = 0

    logger.info(f"[Lemmatizing arguments ... ]")
    for ln in data[0:100]:
        count += 1
        ln = json.loads(ln)

        doc = nlp(ln["arguments"])
        lemmas_ = [word.lemma for sent in doc.sentences for word in sent.words]

        fout.write(json.dumps({"id": ln["id"], "lemma": lemmas_}))
        fout.write("\n")

    fout.close()
    logger.info(f"[Loaded {count} arguments ... ]")

if __name__ == "__main__":
    run_nlp()