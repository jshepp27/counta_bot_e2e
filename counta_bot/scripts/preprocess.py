import json
import pandas as pd
import logging

from counta_bot.detection.stance_classifier import sentence_stance
from yake import KeywordExtractor
from tqdm import tqdm
from counta_bot.utils.utils import sentences_segment

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("PRE-PROCESSOR")


def truncate(data, l=6):
    data = pd.DataFrame(data)

    data_ = []
    for _, i in data.iterrows():

        truncated_args = sentences_segment(i["arguments"])[0:l]
        truncated_counters = sentences_segment(i["counters"])[0:l]

        data_.append({
            "id": i["id"],
            "titles": i["titles"],
            "argument": " ".join(i for i in truncated_args),
            "counters": " ".join(i for i in truncated_counters)
        })

    return data_

def unique_entries(args, key="id"):
    data_ = pd.DataFrame(args)
    unique = data_.drop_duplicates(subset="id")

    unique_ = []
    for _, i in unique.iterrows():
        unique_.append({
            "id": i["id"],
            "titles": i["titles"],
            "argument": i["argument"],
            "counters": i["counters"]
        })

    return unique_


kw_extractor = KeywordExtractor(lan="en", n=3, top=5)
def process_aspects(data, key="argument"):
    aspects = []
    with tqdm(total=(len(data)), position=0, leave=True) as pbar:
        for _ in data:
            # TODOs: Run KeyBERT at Scale
            aspects.append([i[0] for i in kw_extractor.extract_keywords(_[key])])

            pbar.update()

        return aspects


def process_stance(data, aspects):
    stance = []
    with tqdm(total=(len(data))) as pbar:
        for i, j in zip(data, aspects):
            if not j:
                stance.append(" ")

            else:
                aspect = j.pop(0)
                stance.append((sentence_stance(i["titles"], aspect), aspect))

            pbar.update()

        return stance

if __name__ == "__main__":
    ### LOAD DATA ###
    args = [json.loads(ln) for ln in open("../../data/train_cmv_cleaned.jsonl", "r")]
    logger.info(f"[{len(args)} Arguments Processed]")

    logger.info("[Pre-processor Initialised]")

    ### TRUNCATE ###
    args = truncate(args)

    ### EXTRACT UNIQUE ###
    unique_ = unique_entries(args, key="id")
    logger.info(f"[{len(unique_)} Unique Arguments]")

    ### EXTRACT ASPECTS ###
    kw_extractor = KeywordExtractor(lan="en", n=3, top=5)
    arg_aspects = process_aspects(unique_, key="argument")
    counter_aspects = process_aspects(unique_, key="counters")

    logger.info(f"[{len(arg_aspects)} Keyphrases Processed]")

    ### DETERMINE STANCE ###
    arg_stance = process_stance(unique_, arg_aspects)
    logger.info(f"[{len(arg_stance)} Stance Polarities Processed]")

    ### WRITE TO DISK ###
    fout = open("./data/processed_train_cmv.jsonl", "w")

    #TODOs: Sentence Segment
    with tqdm(total=(len(unique_))) as pbar:
        with fout:
            for i, j, k, l in zip(unique_, arg_aspects, arg_stance, counter_aspects):
                fout.write(json.dumps([{
                    "id": i["id"],
                    "title": i["titles"],
                    "argument": {"argument": i["argument"], "arg_keyphrases": j, "arg_stance": k},
                    "counter_tgt": {"counter": i["counters"], "counter_keyphrases": l}
                }]))

                fout.write("\n")
                pbar.update()

    logger.info(f"[{len(unique_)} Data Stored as process_train_cmv.jsonl]")
