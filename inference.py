import torch
import numpy as np
from collections import Counter
from torchtext.legacy.data import Iterator
from model import Model, CharModel

PAD_TOKEN = '<pad>'


def test(model: Model, testing_data: Iterator, output_matrix: bool = False, level: str = 'char') -> float:

    model.eval()
    batch_accuracies = []
    classes = testing_data.dataset.fields['lang'].vocab.itos
    n_classes = len(classes)
    confusion_matrix = np.zeros((n_classes, n_classes))
    sparse_matrix = Counter()

    for j, batch in enumerate(iter(testing_data)):

        if level == 'char':
            sequence = batch.chars[0]
            lengths = batch.chars[1]
            target = batch.lang
        else:
            sequence = batch.paragraph[0]
            lengths = batch.paragraph[1]
            char_lengths = batch.paragraph[2]
            target = batch.lang

        predictions = model.forward(sequence)
        _, predicted_languages = torch.topk(predictions, 1)

        # Save data needed to calculate accuracy for later
        batch_accuracies.extend(target.eq(predicted_languages))

        for p, t in zip(predicted_languages, target):
            if p != t:
                confusion_matrix[p][t] += 1
                sparse_matrix[(classes[p],classes[t])] += 1

    if output_matrix:
        with open("confusion_matrix.txt", 'w') as f:
            f.write("\t")
            f.write("\t".join(classes))
            f.write("\n")
            for i, line in enumerate(confusion_matrix):
                f.write("{}\t".format(classes[i]))
                f.write("\t".join(map(str, line)))
                f.write("\n")

        with open("sparse_matrix.txt", 'w') as f:
            for lan, score in sparse_matrix.most_common():
                f.write("{} - {} : {}\n".format(lan[0], lan[1], score))

    return np.array([sample.item() for sample in batch_accuracies]).mean()