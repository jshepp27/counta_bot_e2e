import json
import logging
import sys
import re
sys.path.append('./')

from counta_bot.utils.elastic_db import ElasticDB
from counta_bot.detection.stance_classifier import sentence_stance, compare_stance
from yake import KeywordExtractor
import tqdm.notebook as tqdm
from tqdm import tqdm

### FAST-NLP REGEX UTILS ###
def sentences_segment(doc):
    return [i for i in re.split(r'(?<=[^A-Z].[.?]) +(?=[A-Z])', doc)]

def tokeniser(doc):
    return re.findall(r"\w+(?:'\w+)?|[^\w\s]", doc)

### INIT OBJECT ###
PORT = "http://localhost:9200"
db = ElasticDB(elastic_port=PORT)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("RETRIEVER_RANKER")
logger.info("[Retriever Initialised]")

### LOAD DATA ###
args = [json.loads(ln) for ln in open("./data/processed_train_cmv.jsonl")]

### RETRIEVER-RANKER ###
kw_extractor = KeywordExtractor(lan="en", n=3, top=5)

# TODOs: Refactor as Class Object
def retrieved_evidence(arg):
    ad_units = sentences_segment(arg["argument"])

    results = []

    for adu in ad_units:
        toks = re.findall(r"\w+(?:'\w+)?|[^\w\s]", adu)

        if len(toks) <= 8:
            continue

        kp = arg["keyphrase"]

        if kp:
            query = ", ".join(i for i in kp)

            titles = [i["_source"]["document"]["title"] for i in db.search(query_=query, k=10)]
            evidence = [i["_source"]["document"]["text"] for i in db.search(query_=query, k=10)]

            results.append({
                "tid": arg["id"],
                "argument_discourse_unit": adu,
                "retrieved_documents_titles": titles,
                "query": query,
                "adu_keyphrases": [i for i in kp],
                "adu_stance": sentence_stance(adu, kp),
                "retrieved_evidence": evidence,
                "merged_evidence": ", ".join(ln for ln in evidence)
            })

        return results

def overlap_score(evidence_kp, adu_kp):
    score = 0
    # TODOs: Robust 'None' Handelling
    if adu_kp == None:
        return score
    # Split Keyphrase into components, scoring partial units as overlap
    else:
        for i in evidence_kp:
            for j in i.split():
                # Ensure string value, to enact .find
                if ", ".join([i for i in adu_kp]).find(j) != -1:
                    score += 1

                else:
                    continue

    return score


def calculate_overlap(merged_ev, adu_kp):
    for ev_unit in sentences_segment(merged_ev):
        toks = tokeniser(ev_unit)
        kp_overlap = 0

        if len(toks) <= 8: continue

        # ev_unit_kp = [i for i in keywords.keywords(ev_unit).split("\n")]
        ev_unit_kp = [i[0] for i in kw_extractor.extract_keywords(ev_unit)]

        if ev_unit_kp:
            kp_overlap = overlap_score(evidence_kp=ev_unit_kp, adu_kp=adu_kp)

        else:
            ev_unit_kp = None
        yield ev_unit, ev_unit_kp, kp_overlap


def score_passages(ev_):
    adu = ev_[0]["argument_discourse_unit"]
    adu_stance = ev_[0]["adu_stance"]
    merged_ev = ev_[0]["merged_evidence"]
    adu_kp = ev_[0]["adu_keyphrases"]

    ### CALCULATE OVERLAP ###
    for ev_unit, ev_unit_kp, kp_overlap in calculate_overlap(merged_ev, adu_kp):
        target = adu_kp[0]

        compared_stace = compare_stance(ev_unit, target)
        if compared_stace != adu_stance:
            yield {
                "adu": adu,
                "adu_kp": adu_kp,
                "evidence_unit": ev_unit,
                "evidence_kps": ev_unit_kp,
                "overlap": kp_overlap,
                "evidence_stance": compare_stance(ev_unit, target),
                "adu_stance": adu_stance
            }

        else:
            continue

### SCORED EVIDENCE ###
def score_evidence(retrieved_evidence):
    for ev_ in retrieved_ev:
        yield [i for i in score_passages(ev_)]

### RANKED EVIDENCE ###
def rank_filter_counter_evidence(retireved_evidence, k=3):
    with tqdm(total=(len(retrieved_ev))) as pbar:
        for i in score_evidence(retrieved_ev):
            yield sorted(i, key=lambda y: y["overlap"], reverse=True)[0:k]

            pbar.update()

if __name__ == "__main__":
    # TODOs: All Data
    logger.info("[Retriever Running ...]")

    logger.info("[Running Queries ...]")
    retrieved_ev = []
    for arg in args[0:100]:
        retrieved_ev.append(retrieved_evidence(arg[0]))

    logger.info("[Ranking Results ...]")
    ranked_sorted_evidence = [i for i in rank_filter_counter_evidence(retrieved_ev)]
    print(ranked_sorted_evidence)

    #args = [json.loads(ln) for ln in open("./data/processed_train_cmv.jsonl")]

    fout = open("./data/rr_counter_evidence.jsonl", "w")
    logger.info("[Writing to Disk ...]")
    for args, adus in zip(args, ranked_sorted_evidence):
        for arg in args:
            adus_ = []
            for unit in adus:
                adus_.append({
                    "adu": unit["adu"],
                    "adu_kp": unit["adu_kp"],
                    "evidence_unit": unit["evidence_unit"],
                    "evidence_kps": unit["evidence_kps"],
                    "overlap": unit["overlap"],
                    "evidence_stance": unit["evidence_stance"],
                    "adu_stance": unit["adu_stance"],
                })

            fout.write(json.dumps([{
                "ids": arg["id"],
                "rank_retrieved_filered": adus_
            }]))

            fout.write("\n")

