import json
import pandas as pd
import logging

from counta_bot.detection.stance_classifier import sentence_stance
from yake import KeywordExtractor
import multiprocessing
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("PRE-PROCESSOR")

import os
print(os.getcwd())

def unique_entries(data, key="id"):
    data_ = pd.DataFrame(data)
    unique = data_.drop_duplicates(subset=[key])

    unique_ = []
    for _, i in unique.iterrows():
        unique_.append({
            "id": i["id"],
            "titles": i["titles"],
            "argument": i["arguments"],
            "counters": i["counters"]
        })

    return unique_


def process_aspects(data):
    aspects = []
    with tqdm(total=(len(data)), position=0, leave=True) as pbar:
        for _ in data:
            # TODOs: Run KeyBERT at Scale
            aspects.append([i[0] for i in kw_extractor.extract_keywords(_["argument"])])

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
    logger.info("[Pre-processor Initialised]")

    ### LOAD DATA ###
    args = [json.loads(ln) for ln in open("../../data/train_cmv_cleaned.jsonl")]
    logger.info(f"[{len(args)} Arguments Processed]")

    ### EXTRACT UNIQUE ###
    unique_ = unique_entries(args, key="id")
    logger.info(f"[{len(unique_)} Unique Arguments]")

    ### EXTRACT ASPECTS ###
    kw_extractor = KeywordExtractor(lan="en", n=3, top=5)
    aspects = process_aspects(unique_)
    logger.info(f"[{len(aspects)} Keyphrases Processed]")

    # ### DETERMINE STANCE ###
    stance = process_stance(unique_, aspects)
    logger.info(f"[{len(stance)} Stance Polarities Processed]")

    ### WRITE TO DISK ###
    fout = open("../../data/processed_train_cmv.jsonl", "w")

    with tqdm(total=(len(unique_))) as pbar:
        print("fuck you")
        with fout:
            for i, j, k in zip(unique_, aspects, stance):
                fout.write(json.dumps([{
                    "id": i["id"],
                    "title": i["titles"],
                    "argument": i["argument"],
                    "keyphrase": j,
                    "stance": k,
                    "counter": i["counters"]
                }]))

                fout.write("\n")
                pbar.update()



    logger.info(f"[{len(unique_)} Data Stored to Disk]")