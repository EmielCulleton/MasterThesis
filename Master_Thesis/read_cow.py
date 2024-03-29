import os
from tqdm import tqdm


def read_cow(file):

    sent = []
    with open('./cow_sents.txt', 'a') as fo:
        for line in tqdm(open(file)):
            word = str(line).split('\t')[0]
            if word.startswith('</s>'):
                fo.write(' '.join(sent) + "\n")
                sent = []
            elif not word.startswith('<'):
                sent.append(word)


    with open('./cow_sents.txt', "r") as cow:
        data = cow.read()
        data_into_list = data.split("\n")

    os.remove('./cow_sents.txt')

    return data_into_list, print("Done!"), print(data_into_list[0:6])

cow_sample = read_cow("/home/emiel/data/encow16ax01_sample.xml")