import re
import io
import os
import nltk

import numpy as np
from typing import Iterable, Dict, Union
from collections import Counter

from torch.utils.data import DataLoader
from torchtext.legacy.data import Field, NestedField, Iterator
from torchtext.legacy.data.dataset import Dataset
from torchtext.legacy.data.example import Example


PAD_TOKEN = '<pad>'
START_TOKEN = '<start>'
END_TOKEN = '<end>'

def get_data_fields(fixed_lengths: int) -> Dict:
    """
    Creates torchtext fields for the I/O pipeline
    :param fixed_lengths:
    :return:
    """
    language = Field(
        batch_first=True, init_token=None, eos_token=None, pad_token=None, unk_token=None)

    characters = Field(include_lengths=True, batch_first=True, init_token=None,
                        eos_token=END_TOKEN, pad_token=PAD_TOKEN, fix_length=fixed_lengths)

    nesting_field = Field(tokenize=list, pad_token=PAD_TOKEN, batch_first=True,
                          init_token=None, eos_token=END_TOKEN)
    paragraph = NestedField(nesting_field, pad_token=PAD_TOKEN, eos_token=END_TOKEN,
                            include_lengths=True)

    fields = {
        'chars': ('chars', characters),
        'paragraph': ('paragraph', paragraph),
        'lang': ('lang', language)
    }

    return fields


# data prep
def data_reader(x_data: Iterable,
                y_data: Iterable,
                split_sent: bool,
                level: str,
                train: bool,
                max_chars: int) -> Dict:

    """Return examples as dict"""

    example = {'id': [], 'paragraph': [], 'lang': [], 'chars': []}
    for x, y in zip(x_data, y_data):
        x = x.strip()  # remove whitespace
        y = y.strip()

        examples = []

        if split_sent:
            split_sentences = nltk.tokenize.sent_tokenize(x)
        else:
            split_sentences = [x]

        for x in split_sentences:
            if len(x) == 0: continue  # check len
            example = {'id': [], 'paragraph': [], 'lang': [], 'chars': []}
            # replace all numbers with 0
            x = re.sub('[0-9]+', '0', x)
            paragraph = x.split()
            language = y

            count = 0 # keep count of sentence length
            # check if char level processing is needed
            if level == 'char' or train:
                example['paragraph'] = [word.lower() for word in paragraph[:max_chars]]
            else:

                example['paragraph'] = []
                for word in paragraph:
                    cur_word = word.lower()
                    remaining = max_chars - count
                    count += len(cur_word)
                    if not count > max_chars and len(cur_word) > 0:
                        example['paragraph'].append(cur_word)
                    # handle when the count exceeds max allowed characters but some room is still there
                    elif remaining > 0:
                        # remove added count and add chars as per the room left
                        count -= len(cur_word) + len(''.join(list(cur_word)[:remaining]))
                        example['paragraph'].append(''.join(list(cur_word)[:remaining]))
                        break
                    # if no room left
                    else:
                        count -= len(cur_word)
                        break

                assert count <= max_chars, "Too much chars, max_chars: {}, count: {}".format(max_chars, count)
                if len(example['paragraph'])==0:
                    continue

            example['lang'] = language
            example['chars'] = list(x)[:max_chars]
            examples.append(example)

        yield examples

    # possible last sentence without newline after
    # if len(example['paragraph']) > 0:
    #     yield [example]


class WiliDataset(Dataset):

    def sort_key(self, example):
        if self.level == "char":
            return len(example.chars)
        else:
            return len(example.paragraph)

    """Create Wili dataset from given text and label files"""

    def __init__(self,
                 paragraph_path:str,
                 label_path:str,
                 fields: dict,
                 split_sentences: bool,
                 train: bool,
                 max_chars: int = 1000,
                 level: str = "char",
                 **kwargs):

        self.level = level

        with io.open(os.path.expanduser(paragraph_path), encoding="utf=8") as f_text, \
            io.open(os.path.expanduser(label_path), encoding="utf-8") as f_lab:

            examples = []
            for d in data_reader(f_text, f_lab, split_sentences, level, train,max_chars):
                for sentence in d:
                    examples.extend([Example.fromdict(sentence, fields)])

        if isinstance(fields, dict):
            fields, field_dict = [], fields
            for field in field_dict.values():
                if isinstance(field, list):
                    fields.extend(field)
                else:
                    fields.append(field)

        super(WiliDataset, self).__init__(examples, fields, **kwargs)


def load_data(max_chars_test: int = -1,
              train: bool = True,
              **kwargs
              ):

    if kwargs['fix_lengths']:
        fixed_length = kwargs['max_chars']
    else:
        fixed_length = None

    fields = get_data_fields(fixed_length)
    _paragraph = fields["paragraph"][-1]
    _language = fields["lang"][-1]
    _characters = fields['chars'][-1]

    if train:
        training_data = WiliDataset(kwargs['train_data'],
                                    kwargs['train_labels'],
                                    fields,
                                    kwargs['split_paragraphs'],
                                    kwargs['fix_lengths'],
                                    kwargs['max_chars'],
                                    kwargs['level'])
        validation_data = WiliDataset(kwargs['val_data'],
                                      kwargs['val_labels'],
                                      fields,
                                      False,
                                      False,
                                      kwargs['max_chars'],
                                      kwargs['level'])

        _paragraph.build_vocab(training_data, min_freq=kwargs['min_frequency'])
        _language.build_vocab(training_data)
        _characters.build_vocab(training_data, min_freq=kwargs['min_frequency'])
        return training_data, validation_data

    else:
        if max_chars_test == -1: max_chars_test = kwargs['max_chars']
        testing_data = WiliDataset(kwargs['test_data'],
                                   kwargs['test_labels'],
                                   fields, False, False, max_chars_test, kwargs['level'])
        return testing_data


def create_train_val_set():
    paragraph_path = "data/wili-2018/x_train.txt"
    label_path = "data/wili-2018/y_train.txt"
    data = open(label_path, encoding='utf-8').read().strip().split('\n')
    train, validate = np.split(np.array(data), [int(len(data) * 0.9)])
    # print(len(train), len(validate))
    with open("data/wili-2018/y_train_new.txt", "w", encoding="utf-8") as file1:
        for item in train:
            file1.write("%s\n" % item)
    with open("data/wili-2018/y_val.txt", "w", encoding="utf-8") as file2:
        for item in validate:
            file2.write("%s\n" % item)