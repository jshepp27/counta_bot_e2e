from keybert import KeyBERT

kb = KeyBERT()

def exctract_keyphrase(doc, n_gram=2, n_kp=3):
    #kp = kb.extract_keywords(doc, keyphrase_ngram_range=(1, 3), stop_words=None, use_mmr="True", diversity=0.2)
    #kp = kb.extract_keywords(doc, keyphrase_ngram_range=(1, n_kp), stop_words="english", use_maxsum="True", top_n=5)

    kp = kb.extract_keywords(doc, keyphrase_ngram_range=(0, n_gram), stop_words="english")

    return [i[0] for i in kp[0:n_kp]]