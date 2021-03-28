import torch
import os
import time
import numpy as np
from torch.optim.lr_scheduler import LambdaLR
from torchtext.legacy.data import Iterator
from dataloader import load_data
from model import CharCNN
from inference import test


PAD_TOKEN = '<pad>'


# train model
def train(**kwargs):

    training_data, validation_data = load_data(**kwargs)

    n_classes = len(training_data.fields['lang'].vocab)
    char_vocab_size = len(training_data.fields['chars'].vocab)
    padding_idx = training_data.fields['chars'].vocab.stoi[PAD_TOKEN]
    print(n_classes, char_vocab_size, padding_idx)

    gpu = True if torch.cuda.is_available() and kwargs['use_cuda'] else False
    device = torch.device(type='cuda') if gpu else torch.device(type='cpu')

    training_iterator = Iterator(training_data, kwargs['batch_size'], train=True,
                                 sort_within_batch=True, device=device, repeat=False)
    validation_iterator = Iterator(validation_data, kwargs['batch_size'], train=False, sort_within_batch=True,
                                   device=device, repeat=False)

    # our model
    model = CharCNN(char_vocab_size, padding_idx, emb_dim=kwargs['emb_dim'],
                    dropout_p=kwargs['dropout'], n_classes=n_classes, max_seq_length=kwargs['max_chars'])
    model.cuda()

    # optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=kwargs['learning_rate'], momentum=0.9)
    scheduler = LambdaLR(optimizer, lr_lambda=lambda t: 0.8 ** (t / 3))
    num_iter_per_epoch = len(training_iterator)

    best_accuracy = 0
    batch_accuracies, epoch_accuracies = [], []
    if kwargs['output_dir'] is None:
        output_dir = os.path.join(
            "./results",
            f"lid_model_{time.strftime('%Y%m%d_%H%M%S')}",
        )
        os.makedirs(output_dir)
    output_file = open(os.path.join(output_dir, "logs.txt"), "w")
    model.train()
    # training loop
    for epoch in range(kwargs['num_epochs']):
        losses = []
        scheduler.step()  # changed since v1.1.0
        for iter, batch in enumerate(training_iterator):
            # train the model; basically telling on what to train

            optimizer.zero_grad()
            # get the inputs
            if kwargs['level'] == 'char':
                sequence = batch.chars[0]
                lengths = batch.chars[1]
                target = batch.lang
                pad_idx = training_iterator.dataset.fields['chars'].vocab.stoi[PAD_TOKEN]
                criterion = torch.nn.CrossEntropyLoss(ignore_index=pad_idx)
            else:
                sequence = batch.paragraph[0]
                lengths = batch.paragraph[1]
                char_lengths = batch.paragraph[2]
                target = batch.lang
                pad_idx = training_iterator.dataset.fields['paragraph'].vocab.stoi[PAD_TOKEN]
                criterion = torch.nn.CrossEntropyLoss(ignore_index=pad_idx)

            batch_size = sequence.shape[0]

            # forward pass: compute predicted y by passing x to the model
            predictions = model.forward(sequence)

            # compute loss
            loss = criterion(predictions, target.squeeze(1))
            losses.append(loss.item())

            _, predicted_languages = torch.topk(predictions, 1)
            # print(predicted_languages)

            # acuracy calculation
            batch_accuracy = target.eq(predicted_languages).sum().item()/batch_size
            batch_accuracies.append(batch_accuracy)
            epoch_accuracies.append(batch_accuracy)

            # compute gradients
            loss.backward()
            optimizer.step()

            print("Training: Iteration: {}/{} Epoch: {}/{} Loss: {}"
                  " Accuracy: {}, Learning rate: {}".format(iter + 1,
                                                            num_iter_per_epoch,
                                                            epoch + 1, kwargs['num_epochs'],
                                                            round(loss.item(),4),
                                                            round(batch_accuracies[-1], 3),
                                                            round(scheduler.get_last_lr()[0], 5)
                                                            ))

        # evaluation of validation data
        train_accuracy = np.array(batch_accuracies).mean()
        batch_accuracies = []
        print("---------- Validation phase start ---------")
        validation_accuracy = test(model, validation_iterator, level='char')
        print(
            "Epoch: {}/{} \nTraining loss: {} \nTraining accuracy: {} \nValidation accuracy: {} \n"
                .format(
                        epoch + 1,
                        kwargs['num_epochs'],
                        round(np.mean(np.array(losses)), 3),
                        round(train_accuracy, 3),
                        round(validation_accuracy, 3))
        )
        print("---------- Validation phase end ---------")

        output_file.write(
            "Epoch: {}/{} \nTraining loss: {} \nTraining accuracy: {} \nValidation accuracy: {} \n".format(
                epoch + 1,
                kwargs['num_epochs'],
                round(np.mean(np.array(losses)), 3),
                round(train_accuracy, 3),
                round(validation_accuracy, 3))
        )

        # saving the model with best accuracy
        if validation_accuracy > best_accuracy:
            best_accuracy = validation_accuracy
            # one way to save everything but bound to class
            # torch.save(model, "lid_model.pt")
            # recommended way
            torch.save(model.state_dict(), os.path.join(output_dir, "lid_model.pt"))


if __name__ == '__main__':
    # train(batch_size=128, num_epochs=10, learning_rate=0.01)
    pass