from keybert import KeyBERT
from keyphrase_vectorizers import KeyphraseCountVectorizer

kb = KeyBERT()

def extract_keyphrase(doc, n_gram=3, n_kp=3, use_mmr="False", use_maxsum="False"):
    kp = None
    
    try:
        kp = kb.extract_keywords(doc, vectorizer=KeyphraseCountVectorizer(), stop_words="english", diversity=0.2,)
    
    except:
        print("No Keywords")

    if kp == None:
        return []

    else: return [i[0] for i in kp[0:n_kp]]     

### TEST STATEMENT ###

test_1 = extract_keyphrase("The environmental impact of aviation in the UK is increasing due to new terminals at Heathrow airport")
test_2 = extract_keyphrase("Will plague ever become as widespread as it was in the 1300's.")

print(test_1)
print(test_2)