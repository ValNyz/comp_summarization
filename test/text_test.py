import browse_corpus
import model.text as text

sents_A, s_A = browse_corpus.load_sents('output/comp/C1801', 'C1801-A')
sents_B, s_B = browse_corpus.load_sents('output/comp/C1801', 'C1801-B')

# print(sents_A)
doc_A = text.document(sents_A)
print(doc_A)
doc_B = text.document(sents_B)
