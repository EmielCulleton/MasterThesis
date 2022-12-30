import pickle
from textwash_datafier import JSONDataset


dataset = pickle.load(open('/home/emiel/data/textwash_data.pickle', 'rb'))
tags_tokens = list(zip(dataset.data[0], dataset.labels[0]))

print("IOB tags:", dataset.labels[0])
print("tokens:", dataset.data[0])
print("together:", list(zip(dataset.data[0], dataset.labels[0])))

