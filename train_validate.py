# Imports
import warnings
import argparse

import numpy as np
import pandas as pd

from tqdm import tqdm
from typing import Callable

# Scikit-learn
import sklearn
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# PyTorch
import torch
from torch.utils.data import DataLoader
import torch.nn as nn

# Imports from other local python files
from models import FNN_Simple, SimpleCNN, construct_resnet_x, save_models_encoder
from data import CNNDataset 

# Ignore warnings
warnings.filterwarnings('ignore')


############################################################

####################    RESNETX/ CNN    ####################

############################################################


def training_step(
        network: torch.nn.Module, 
        optimizer: torch.optim, 
        data: torch.Tensor, 
        targets: torch.Tensor, 
        loss_fn: Callable
    ):
    """
    Performs a single training step.

    Purpose:
        - Runs forward pass, computes loss, backpropagates, and updates weights.

    Args:
        network - model to train (nn.Module)
        optimizer - optimizer instance (torch.optim)
        data - input batch (torch.Tensor)
        targets - ground truth labels (torch.Tensor)
        loss_fn - loss function

    Returns:
        float:
            loss value for the batch
    """
    optimizer.zero_grad()
    output = network(data)
    labels_processed = targets.flatten().long()
    loss = loss_fn(output, labels_processed)
    loss.backward()
    optimizer.step()
    return loss.item()

    
def get_metric(
        model: torch.nn.Module, 
        test_dataloader: DataLoader, 
        loss_fn: Callable
    ):
    """
    Evaluates loss, accuracy, F1 score, and balanced accuracy given test dataloader.

    Note: no returns, just prints. Used for a validity check.    

    Args:
        model - trained model (nn.Module)
        test_dataloader - dataloader for evaluation
        loss_fn - loss function
    """
    model.eval().to('cuda')
    running_loss = 0.
    accuracy = 0.
    counter = 0
    baccuracy = 0.
    f1_scores = 0.
    for i, data in tqdm(enumerate(test_dataloader)):
        inputs, true_labels = data
        inputs = inputs.to('cuda')
        true_labels = true_labels.to('cuda')
        outputs = model(inputs)
        labels_processed = true_labels.flatten().long()
        loss = loss_fn(outputs, labels_processed)
        running_loss += loss.item()
        outputs = np.argmax(outputs.detach().cpu().numpy(), axis=1)
        labels_processed = labels_processed.detach().cpu().numpy()
        acc = accuracy_score(labels_processed, outputs)
        f1 = sklearn.metrics.f1_score(labels_processed, outputs,average="weighted")
        bacc = sklearn.metrics.balanced_accuracy_score(labels_processed, outputs)
        f1_scores += f1
        accuracy += acc
        baccuracy += bacc
        counter += 1

    print('Loss =', running_loss / counter)
    print('Accuracy = ', 100 * (accuracy / counter))
    print('F1_score = ', 100 * (f1_scores / counter))
    print('Baccuracy = ', 100 * (baccuracy / counter))



def training_loop(
        network: torch.nn.Module,
        train_dataloader: DataLoader, 
        test_dataloader: DataLoader, 
        num_epochs: int,
        loss_fn: Callable,
        learning_rate: float,
        show_progress: bool = True
    ) -> tuple[torch.nn.Module, list]:
    """
    Training loop for a single model.

    Args:
        network - model to train (nn.Module)
        train_dataloader - training data loader (DataLoader)
        test_dataloader - evaluation data loader (DataLoader)
        num_epochs - epochs number (int)
        loss_fn - loss function (Callable)
        learning_rate - optimizer learning rate (float)
        show_progress - if True, enables progress bars (bool, default = True)

    Returns:
        Tuple[nn.Module, list]:
            - trained model
            - list of loss values per epoch
    """
    device = "cuda"
    device = torch.device(device)
    if not torch.cuda.is_available():
        print("CUDA IS NOT AVAILABLE")
        device = torch.device("cpu")
    losses = []

    optimizer = torch.optim.AdamW(network.parameters(), lr=learning_rate)

    for _ in tqdm(range(num_epochs), desc="Epoch", position=0, disable= (not show_progress)):
        network.train().to('cuda')
        running_loss = 0.
        last_loss = 0.
        for i, data in tqdm(enumerate(train_dataloader), desc="Minibatch", position=1, leave=False, disable= (not show_progress)):
            inputs, targets = data
            inputs = inputs.to(device=device)
            targets = targets.to(device=device)
            loss = training_step(network, optimizer, inputs, targets,loss_fn)
            running_loss += loss
            if i % 100 == 99:
                last_loss = running_loss / 99 # loss per batch
                print('  batch {} loss: {}'.format(i + 1, last_loss))
                running_loss = 0.

        losses.append(last_loss)
        print('\n')
        get_metric(network, test_dataloader, loss_fn)
        print('\n\n\n')
            
    return network, losses



############################################################

##############     Multi-Input Classifier     ##############

############################################################


def training_step_class(
        network_1: torch.nn.Module, 
        network_2: torch.nn.Module, 
        network_class: torch.nn.Module, 
        optimizer, data_1, data_2, 
        targets, loss_fn: Callable
    ):
    """
    Performs a single training step for a multi-input classifier.

    Multi-Input Classifier:
    The outputs of the two networks are concatenated and put through a classifier (FNN).

    Args:
        network_1, network_2 - feature extraction models (nn.Module)
        network_class - classifier model (nn.Module)
        optimizer - optimizer instance
        data_1, data_2 - input batches
        targets - ground truth labels
        loss_fn - loss function

    Returns:
        float:
            Loss value for the batch
    """
    optimizer.zero_grad()
    output_1 = network_1(data_1)
    output_2 = network_2(data_2)
    output_1_2 = torch.cat((output_1, output_2), 1)
    output_class = network_class(output_1_2)
    labels_processed = targets.flatten().long()
    loss = loss_fn(output_class, labels_processed)
    loss.backward()
    optimizer.step()
    return loss.item()
    
def get_metric_class(
        network_1: torch.nn.Module, 
        network_2: torch.nn.Module, 
        network_class: torch.nn.Module, 
        loss_fn: Callable,
        test_dataloader: DataLoader
    ):
    """
    Evaluates (loss, accuracy, F1 score, and balanced accuracy) multi-input model performance.

    Multi-Input Classifier:
    The outputs of the two networks are concatenated and put through a classifier (FNN).

    Note: no returns, just prints. Used for a validity check.  

    Args:
        network_1, network_2 - feature extractors (nn.Module)
        network_class - classifier model (nn.Module)
        loss_fn - loss function (Callable)
        test_dataloader - evaluation dataloader (DataLoader)
    """
    
    network_class.eval().to('cuda')
    running_loss = 0.
    accuracy = 0.
    counter = 0
    baccuracy = 0.
    f1_scores = 0.

    print(torch.cuda.get_device_name(0))
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')

    for i, data in tqdm(enumerate(test_dataloader)):

        inputs_1, inputs_2, true_labels = data
        inputs_1 = inputs_1.to('cuda')
        inputs_2 = inputs_2.to('cuda')
        true_labels = true_labels.to('cuda')
        output_1 = network_1(inputs_1)
        output_2 = network_2(inputs_2)
        output_1_2 = torch.cat((output_1, output_2), 1)
        outputs = network_class(output_1_2)
        labels_processed = true_labels.flatten().long()
        loss = loss_fn(outputs, labels_processed)
        running_loss += loss.item()
        outputs = np.argmax(outputs.detach().cpu().numpy(), axis=1)
        labels_processed = labels_processed.detach().cpu().numpy()
        
        acc = accuracy_score(labels_processed, outputs)
        f1 = sklearn.metrics.f1_score(labels_processed, outputs,average="weighted")
        bacc = sklearn.metrics.balanced_accuracy_score(labels_processed, outputs)

        f1_scores += f1
        accuracy += acc
        baccuracy += bacc
        counter += 1
        
    print('Loss =', running_loss / counter)
    print('Accuracy = ', 100 * (accuracy / counter))
    print('F1_score = ', 100 * (f1_scores / counter))
    print('Baccuracy = ', 100 * (baccuracy / counter))


def training_loop_class(
        network_1: torch.nn.Module,
        network_2: torch.nn.Module,
        network_class: torch.nn.Module,
        train_dataloader: DataLoader,
        test_dataloader: DataLoader,
        num_epochs: int,
        loss_fn: Callable,
        learning_rate: float,
        show_progress: bool = True
    ) -> tuple[torch.nn.Module, torch.nn.Module, torch.nn.Module, list]:
    """
    Training loop for a multi-input classification system.

    Multi-Input Classifier:
    The outputs of the two networks are concatenated and put through a classifier (FNN).

    Args:
        network_1, network_2 - feature extraction models (nn.Module)
        network_class - classifier model (nn.Module)
        train_dataloader - training data loader (DataLoader)
        test_dataloader - evaluation data loader (DataLoader)
        num_epochs - number of epochs (int)
        loss_fn - loss function (Callable)
        learning_rate - optimizer learning rate (float)
        show_progress - if True, enables progress bars (bool)

    Returns:
        Tuple[nn.Module, nn.Module, nn.Module, list]:
            - trained networks
            - list of loss values per epoch
    """
    device = "cuda"
    device = torch.device(device)
    if not torch.cuda.is_available():
        print("CUDA IS NOT AVAILABLE")
        device = torch.device("cpu")
    losses = []

    optimizer = torch.optim.AdamW(network_class.parameters(), lr=learning_rate)

    for epoch in tqdm(range(num_epochs), desc="Epoch", position=0, disable= (not show_progress)):

        network_class.train().to('cuda')
        running_loss = 0.
        last_loss = 0.
        for i, data in tqdm(enumerate(train_dataloader), desc="Minibatch", position=1, leave=False, disable=(not show_progress)):
            inputs_1, inputs_2, targets = data
            inputs_1 = inputs_1.to(device=device)
            inputs_2 = inputs_2.to(device=device)
            targets = targets.to(device=device)
            loss = training_step_class(network_1, network_2, network_class, optimizer, inputs_1, inputs_2, targets, loss_fn)
            running_loss += loss
            if i % 100 == 99:
                last_loss = running_loss / 99 # loss per batch
                print('  batch {} loss: {}'.format(i + 1, last_loss))
                running_loss = 0.

        losses.append(last_loss)
        print('\n')
        get_metric_class(network_1, network_2, network_class, loss_fn, test_dataloader)
        print('\n\n\n')
            
    return network_1, network_2, network_class, losses


############################################################

##############      Execute Python Code       ##############

############################################################



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Advanced Sliding Window Word Classification Models")

    # ======== 1 Required     ========
    parser.add_argument("--model_name", type=str, required=True, help="Model name")

    # ======== 2 Data         ========
    parser.add_argument("--data_path", type=str, default="data\\development.npy", help="Path to input numpy data")
    parser.add_argument("--labels_path", type=str, default="data\\development.csv", help="Path to labels csv")
    parser.add_argument("--test_size", type=float, default=0.15, help="Test split size")
    parser.add_argument("--random_state", type=int, default=89, help="Random seed")
    parser.add_argument("--useful_words", type=str, nargs='+',
        default=['Licht','Radio','Ofen','Fernseher','Lüftung','Heizung','Staubsauger','Alarm','other','an','aus'],
        help="Words to keep")
    parser.add_argument("--device", type=str, default="cuda", help="Device")

    # ======== 3 Parameters   ========
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--cnn_hidden_channels", type=int, nargs='+', default=[32,64,128,256,128,64,32], help="custom CNN channels")
    parser.add_argument("--cnn_kernel_size", type=int, default=3, help="CNN kernel size")
    parser.add_argument("--fnn_hidden_dim", type=int, default=512, help="Hidden dim for classifier")

    parser.add_argument("--lr_resnetx", type=float, default=0.0001, help="Learning rate for ResNetX")
    parser.add_argument("--lr_cnn", type=float, default=0.0001, help="Learning rate for custom CNN")
    parser.add_argument("--lr_classifier", type=float, default=0.0001, help="Learning rate for classifier")

    parser.add_argument("--epochs_cnn", type=int, default=2)
    parser.add_argument("--epochs_resnetx", type=int, default=3)
    parser.add_argument("--epochs_classifier", type=int, default=2)

    args = parser.parse_args()

    # ======== Load data inputs and utilities ======== #
    labels = pd.read_csv(args.labels_path)

    useful_words = args.useful_words
    labels = labels[labels['word'].isin(useful_words)]
    encoder = LabelEncoder()
    encoder.fit(labels['word'])
    encoded_labels = encoder.transform(labels['word'])
    labels['word_labels'] = encoded_labels

    # ======== Load and prepare data ======== #
    train_ids, test_ids = train_test_split(labels['speaker_id'].unique(), test_size=args.test_size,
                                           random_state=args.random_state)
    train = labels[labels['speaker_id'].isin(train_ids)]
    test = labels[labels['speaker_id'].isin(test_ids)]
    train_dataset = CNNDataset(args.data_path, train, 0)
    test_dataset = CNNDataset(args.data_path, test, 0)
    train_dataset.set_to_additional_values(0)
    test_dataset.set_to_additional_values(0)

    # #### Train Models #### #

    # ======== 1 - ResNetX ======== #

    model_melspect, _ = construct_resnet_x()
    model_melspect = model_melspect.to(args.device)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    loss_fn = torch.nn.CrossEntropyLoss()
    model_melspect, n_inputs_resnet_x = training_loop(
        model_melspect,train_dataloader, test_dataloader,
        args.epochs_resnetx, loss_fn, args.lr_resnetx, True)

    # ======== 2 - Custom CNN ======== #

    model_mfcc = SimpleCNN(
        1, args.cnn_hidden_channels,
        1, True, args.cnn_kernel_size 
    ).to(args.device) 

    # it requires different input
    train_dataset.set_to_additional_values(1)
    test_dataset.set_to_additional_values(1)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False) 

    loss_fn = torch.nn.CrossEntropyLoss()
    model_mfcc, _ = training_loop(
        model_mfcc, train_dataloader,test_dataloader, 
        args.epochs_cnn, loss_fn, args.lr_cnn, True)

    # ======== 3 - Final Classifier ======== #

    classifier = FNN_Simple(n_inputs_resnet_x, args.fnn_hidden_dim, len(useful_words)).to(args.device)

    model_mfcc.fc = nn.Flatten()
    model_melspect.fc = torch.nn.Identity()

    # Now we change the data for the classifier training and build new datasets
    train_dataset.set_to_additional_values(2)
    test_dataset.set_to_additional_values(2)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False) 

    loss_fn = torch.nn.CrossEntropyLoss()
    model_mfcc, model_melspect, classifier, _ = training_loop_class( 
        model_mfcc,model_melspect, classifier, 
        train_dataloader, test_dataloader, args.epochs_classifier, 
        loss_fn, args.lr_classifier, True
    )

    # ======== SAVE ALL MODELS ======== # 
    save_models_encoder(model_mfcc, model_melspect, classifier, encoder, args)
