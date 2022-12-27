# code Chris Emmery wrote in order to display contents of the COW corpus

from tqdm import tqdm

sent = []
with open('./cow_sents.txt', 'a') as fo:
    for line in tqdm(open('/home/emiel/data/encow16ax01.xml')):
        word = str(line).split('\t')[0]
    if word.startswith('</s>'):
        fo.write(' '.join(sent) + '\n')
        sent = []
    elif not word.startswith('<'):
        sent.append(word)