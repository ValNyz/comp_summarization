import browse_corpus
from kmeans import kmeans
import gensim

s_A, sents_A = browse_corpus.load_sents('output/test/C1801', 'C1801-A')
s_B, sents_B = browse_corpus.load_sents('output/test/C1801', 'C1801-B')

model_name = 'testWiki'

sents = []
sents.extend(s_A)
sents.extend(s_B)
model = gensim.models.Word2Vec.load(model_name)
model.min_count = 0
model.build_vocab(sents, update=True)
model.train(sents, total_examples=len(sents),
            epochs=model.epochs)
kmeans(sents, 3, 0.0001, model)
