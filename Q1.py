
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
from torch import nn


QUANTILES = 10
FRAC = 0.8
BATCH_SIZE = 10
NUM_OF_EPOCHS = 10
table_columns = ['AGE', 'SEX', 'BMI', 'BP', 'S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'Y', 'Class']


class DiabetesDataset(Dataset):

    def __init__(self, data_file, img_cols, target_col):
        if not isinstance(data_file, pd.DataFrame):
            df = pd.read_csv(data_file, sep='\t')
            df['Class'] = pd.qcut(df['Y'], q=QUANTILES, labels=np.arange(QUANTILES)+1)  # labels: [1, 2, 3,...,QUANTILES+1]  # Sections C and D
            #df['Class_div'] = pd.qcut(df['Y'], q=QUANTILES)  # the distribution ranges that the Y values are in.

        else:  # data_file is instance of DataFrame because we split the data for training and testing.
            df = data_file

        self.df = df
        self.img_cols = img_cols
        self.target_col = target_col
        #print(df.head(50))

    def __getitem__(self, idx):
        img = self.df[self.img_cols].iloc[idx]
        img = torch.tensor(img.values, dtype=torch.float32)
        label = self.df[self.target_col].iloc[idx]
        label = torch.tensor(label, dtype=torch.long)
        return img, label

    def __len__(self):
        return self.df.shape[0]


def print_single_batch(features, target):
    data_table = DiabetesDataset("diabetes.csv", features, target)  # Section E
    train_dataloader = DataLoader(data_table, batch_size=BATCH_SIZE, shuffle=True)  # Section F
    imgs, labels = next(iter(train_dataloader))
    print(imgs, '\n', labels)


def split_data(data_file, features, target):
    df = pd.read_csv(data_file, sep='\t')
    df['Class'] = pd.qcut(df['Y'], q=QUANTILES, labels=np.arange(QUANTILES) + 1)
    training_df = df.sample(frac=FRAC)
    test_df = df.drop(training_df.index)
    train_data = DiabetesDataset(training_df, features, target)
    test_data = DiabetesDataset(test_df, features, target)
    return train_data, test_data


def train_network(features, target, split=False, several_epochs=True):
    if split is True:
        train_data, test_data = split_data("diabetes.csv", features, target)
        test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True)

    else:
        train_data = DiabetesDataset("diabetes.csv", features, target)  # Section E

    train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)  # Section F
    num_of_features = len(features)

    model = nn.Sequential(
        nn.Linear(num_of_features, 110),
        nn.ReLU(),
        nn.Linear(110, 110),
        nn.ReLU(),
        nn.Linear(110, QUANTILES+1),
        nn.LogSoftmax(dim=1)
    )  # return a Probabilities Vector
    #print(model)


    def reset_model_params():
        for layer in model.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

    # Running NLLLoss() immediately after LogSoftmax yields the cross entropy, which we want to use.
    CE_loss = nn.NLLLoss()  # Cross Entorpy = Negative Log Likelihood Loss .
    reset_model_params()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)  # lr is the learning rate used for both models' optimizers


    def iterate_batch(imgs, labels):  # Iteration of the training loop
        # imgs = imgs.flatten(start_dim=1)   # imgs already is a one-dimensional tensor
        optimizer.zero_grad()  # Reset the gradient of the parameters
        y_model = model(imgs)  # Entering the data into the network and calculating predicted probabilities
        loss = CE_loss(y_model, labels)  # Calculate the loss function on current minibatch.
        loss.backward()  # Calculation of the random gradient
        optimizer.step()  # update values of (Weights, bias)

        """ Let's convert the vector of probabilities into an unequivocal classification:
            We will use argmax(vector_probabilities) and assume that the model classifies each given sample to
            the class with the highest probability in its output """
        predicted_labels = y_model.argmax(dim=1)
        acc = (predicted_labels == labels).sum() / len(labels)  # acc = the proportion of samples that the model classifies correctly.
        return loss.detach(), acc.detach()


    def iterate_epoch():
        num_of_train_batches = len(train_dataloader)  # total number of batches = math.ceil(len(data_table)/BATCH_SIZE)
        running_loss = 0.
        running_acc = 0.
        for imgs, labels in train_dataloader:
            batch_loss, batch_acc = iterate_batch(imgs, labels)
            running_loss += batch_loss
            running_acc += batch_acc
        loss = running_loss / num_of_train_batches
        acc = running_acc / num_of_train_batches
        return loss, acc


    def eval_model_test():
        loss = 0.
        acc = 0.
        num_of_test_batches = len(test_dataloader)
        for imgs, labels in test_dataloader:
            imgs = imgs.flatten(start_dim=1)
            with torch.no_grad():
                y_model = model(imgs)
                loss += CE_loss(y_model, labels)
                predicted_labels = y_model.argmax(dim=1)
                acc += (predicted_labels == labels).sum() / len(labels)
        return loss / num_of_test_batches, acc / num_of_test_batches


    def train_by_several_epochs():
        loss_train = torch.zeros(NUM_OF_EPOCHS)  # Tensor of 0's in size = num_of_epochs
        acc_train = torch.zeros(NUM_OF_EPOCHS)  # Tensor of 0's in size = num_of_epochs
        if split is True:
            loss_test = torch.zeros(NUM_OF_EPOCHS)
            acc_test = torch.zeros(NUM_OF_EPOCHS)

        model[0].reset_parameters()
        for epoch_idx in tqdm(range(NUM_OF_EPOCHS)):
            loss_train[epoch_idx], acc_train[epoch_idx] = iterate_epoch()
            if split is True:
                loss_test[epoch_idx], acc_test[epoch_idx] = eval_model_test()

        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(range(1, NUM_OF_EPOCHS + 1), loss_train);
        plt.title("CE loss");
        plt.xlabel("Epoch Number");
        plt.subplot(1, 2, 2)
        plt.plot(range(1, NUM_OF_EPOCHS + 1), acc_train);
        plt.title("Accuracy");
        plt.xlabel("Epoch Number");
        plt.show()

        if split is True:
            plt.figure(figsize=(12, 4))
            plt.subplot(1, 2, 1)
            plt.plot(range(1, NUM_OF_EPOCHS + 1), loss_train, label="train loss");
            plt.plot(range(1, NUM_OF_EPOCHS + 1), loss_test, label="test loss");
            plt.title("CE loss");
            plt.legend()
            plt.xlabel("Epoch Number");
            plt.subplot(1, 2, 2)
            plt.plot(range(1, NUM_OF_EPOCHS + 1), acc_train, label="train acc");
            plt.plot(range(1, NUM_OF_EPOCHS + 1), acc_test, label="test acc");
            plt.title("Accuracy");
            plt.legend()
            plt.xlabel("Epoch Number");
            plt.show()


    def train_by_single_epoch():
        num_of_batches = len(train_dataloader)
        loss = torch.zeros(num_of_batches)  # Tensor of 0's in size = num_of_batches
        acc = torch.zeros(num_of_batches)   # Tensor of 0's in size = num_of_batches
        for idx, (imgs, labels) in enumerate(train_dataloader):
            loss[idx], acc[idx] = iterate_batch(imgs, labels)

        plt.figure(figsize=(12,4))
        plt.subplot(1,2,1)
        plt.plot(range(num_of_batches), loss);
        plt.title("CE loss");
        plt.xlabel("Batch Number");
        plt.subplot(1,2,2)
        plt.plot(range(num_of_batches), acc);
        plt.title("Accuracy");
        plt.xlabel("Batch Number");
        plt.show()

    if several_epochs is True:
        train_by_several_epochs()
    else:
        train_by_single_epoch()


#print_single_batch(table_columns[:-1], table_columns[-1])   # Section G

#train_network(table_columns[:-1], table_columns[-1], several_epochs=False)  # Section H: Network training according to all features, single epoch
#train_network(table_columns[:-2], table_columns[-1], several_epochs=False)  # Section H: Network training according to all features, single epoch

#train_network(table_columns[:-1], table_columns[-1])  # Section H: Network training according to all features
#train_network(table_columns[:-2], table_columns[-1])  # Section I: Network training according to all features except 'Y'
#train_network(table_columns[:-1], table_columns[-1], split=True)  # Section J1 : Train and Test Network training according to all features
#train_network(table_columns[:-2], table_columns[-1], split=True)  # Section J2 : Train and Test Network training according to all features except 'Y'


""" Section M:  Repeat sections H, I, J with the updated 'Class' variable"""

QUANTILES = 100
train_network(table_columns[:-1], table_columns[-1])  # Section H: Network training according to all features
train_network(table_columns[:-2], table_columns[-1])  # Section I: Network training according to all features except 'Y'
train_network(table_columns[:-1], table_columns[-1], split=True)  # Section J1 : Train and Test Network training according to all features
train_network(table_columns[:-2], table_columns[-1], split=True)  # Section J2 : Train and Test Network training according to all features except 'Y'
