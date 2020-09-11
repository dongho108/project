import nltk
from nltk.corpus import wordnet as wn

# synonym sets containing "good"

# poses = {'n':'noun', 'v':'verb', 's':'adj (s)', 'a':'adj', 'r':'adv'}
# for synset in wn.synsets("good"):
#     print("{}: {}".format(poses[synset.pos()],
#                           ", ".join([l.name() for l in synset.lemmas()])))

# hypernyms of "panda":

panda = wn.synset("panda.n.01")
hyper=lambda s:s.hypernyms()
print(list(panda.closure(hyper)))