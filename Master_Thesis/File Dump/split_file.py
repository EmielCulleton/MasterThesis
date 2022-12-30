import textwash_datafier
from sklearn.model_selection import train_test_split
import pickle


class train_and_test():

    def __init__(self, file_path):
        self.dataset = pickle.load(open(file_path, "rb"))


        x, y = self.dataset.data, self.dataset.labels
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=36)

        return x_train, x_test, y_train, y_test

train_and_test('/home/emiel/data/textwash_data.pickle')