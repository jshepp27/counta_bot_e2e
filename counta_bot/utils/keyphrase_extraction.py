from keybert import KeyBERT
from keyphrase_vectorizers import KeyphraseCountVectorizer

kb = KeyBERT()

def extract_keyphrase(doc, n_gram=3, n_kp=3, use_mmr="False", use_maxsum="False"):

    #kp = kb.extract_keywords(doc, keyphrase_ngram_range=(0, n_gram), stop_words="english", diversity=0.2, vectorizer=KeyphraseCountVectorizer())
    kp = kb.extract_keywords(doc, vectorizer=KeyphraseCountVectorizer())

    #kw_model.extract_keywords(doc, vectorizer=KeyphraseCountVectorizer())
    
    return [i[0] for i in kp[0:n_kp]] if kp else None

### TEST STATEMENT ###

# test = "The environmental impact of aviation in the UK is increasing due to new terminals at Heathrow airport"
# print(extract_keyphrase(test))
