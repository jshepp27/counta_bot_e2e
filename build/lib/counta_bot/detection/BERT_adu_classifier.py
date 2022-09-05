from transformers import AutoModelForSequenceClassification, DistilBertTokenizerFast, pipeline

# Aleternate "..", "." dependent on ipynb vs .py
saved_path = "./models/BERT_adu_classifier/"
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
BERT_adu = AutoModelForSequenceClassification.from_pretrained(saved_path)

test_1 = "as policemen are a very homogeneous group trained to stick together and the danger of even deepening the pack mentality and escalation of police state"
test_2 = "abortion should be legalised"

pipe = pipeline("text-classification", model=BERT_adu, tokenizer=tokenizer)

def predict(sentence):
    res = pipe(sentence)
    
    return "claim" if res[0]["label"] == "LABEL_0" else "premise"

print(predict(test_2))