import config
from seqeval.metrics import classification_report as classification_report_seqeval


def read_examples_from_file(file_path):
    """Read words and labels from a CoNLL-2002/2003 data file.

    Args:
      file_path (str): path to NER data file.

    Returns:
      examples (dict): a dictionary with two keys: words (list of lists)
        holding words in each sequence, and labels (list of lists) holding
        corresponding labels.
    """

    with open(file_path, encoding="utf-8") as f:
        examples = {"words": [], "labels": []}
        words = []
        labels = []
        for line in f:
            if line.startswith("-DOCSTART-") or line == "" or line == "\n":
                if words:
                    examples["words"].append(words)
                    examples["labels"].append(labels)
                    words = []
                    labels = []
            else:
                splits = line.split(" ")
                words.append(splits[0])
                if len(splits) > 1:
                    labels.append(splits[-1].replace("\n", ""))
                else:
                    # Examples could have no label for mode = "test"
                    labels.append("O")
    return examples


y_true = read_examples_from_file("test.txt")["labels"]
y_pred = read_examples_from_file("spanberta-ner/test_predictions.txt")["labels"]

print(classification_report_seqeval(y_true, y_pred))