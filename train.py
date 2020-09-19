from tqdm.auto import tqdm
import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
from collections import OrderedDict
import argparse
import numpy as np

from dataloaders import get_loaders

def train_net(n_epochs, arch, train_loader, valid_loader, learning_rate, save_path, freezing=False):

    model = eval('models.{}(pretrained=True)'.format(arch))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if freezing == True:
        for param in model.parameters():
            param.requires_grad = False

    model.classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(25088, 4096)),
        ('relu1', nn.ReLU()),
        ('dropout1', nn.Dropout(0.5)),
        ('fc2', nn.Linear(4096, 4096)),
        ('relu2', nn.ReLU()),
        ('dropout2', nn.Dropout(0.5)),
        ('fc3', nn.Linear(4096, 1))]))

    criterion = nn.MSELoss()

    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.1, verbose=True)

    model.to(device)

    # prepare the net for training
    model.train()
    step = 0
    valid_loss_min = np.Inf
    # running_loss = 0.0
    # print_every = print_every

    training_losses, validation_losses = [], []
    for epoch in tqdm(range(n_epochs)):

        training_loss = 0
        model.train()

        for images, labels in train_loader:
            step += 1

            # flatten pts
            labels = labels.view(labels.size()[0], -1)

            # convert variables to floats for regression loss
            labels = labels.type(torch.FloatTensor).to(device)
            images = images.type(torch.FloatTensor).to(device)

            # forward pass to get outputs
            predicted_labels = model.forward(images)

            # calculate the loss between predicted and target keypoints
            loss = criterion(predicted_labels, labels)

            # zero the parameter (weight) gradients
            optimizer.zero_grad()

            # backward pass to calculate the weight gradients
            loss.backward()

            # update the weights
            optimizer.step()

            # print loss statistics
            training_loss += loss.item()
            # if step % print_every == 0:

        validation_loss = 0
        model.eval()

        with torch.no_grad():
            for images, labels in valid_loader:
                labels = labels.view(labels.size()[0], -1)

                labels = labels.type(torch.FloatTensor).to(device)
                images = images.type(torch.FloatTensor).to(device)

                predicted_labels = model.forward(images)
                batch_loss = criterion(predicted_labels, labels)

                validation_loss += batch_loss.item()

            error = [abs(predicted_labels[i] - labels[i]) / labels[i] for i in range(len(predicted_labels))]
            error = (sum(error) / len(predicted_labels)).cpu().numpy()[0] * 100

        train_loss = int(training_loss / len(train_loader))
        validation_loss = int(validation_loss / len(valid_loader))

        scheduler.step(validation_loss)

        training_losses.append(train_loss)
        validation_losses.append(validation_loss)

        print(f"Epoch {epoch + 1}/{n_epochs}.."
              f"Train Loss: {train_loss}.."
              f"Validation Loss: {validation_loss}.."
              f"%Error: {error:.2f}%..")

        # save the model if validation loss decreases
        num_decrease = 0
        if validation_loss <= valid_loss_min:
            print('Validation loss decreased ({:.3f} --> {:.3f}).  Saving model ...'.format(
                valid_loss_min,
                validation_loss))
            torch.save(model.state_dict(), save_path)
            valid_loss_min = validation_loss
            num_decrease += 1

        '''if num_decrease == 3:
            break'''

    print()
    print('Finished Training')

    return model



def main():
    # Some initial parameters
    test_loaders=False

    # Command line arguments
    parser = argparse.ArgumentParser(description='Train a new network on a data set')

    parser.add_argument('data_directory', type=str,
        help='Path of the Image Dataset (with train, valid and test folders)')
    parser.add_argument('--arch', type=str, \
        help='Please choose a pretrained architecture, default is VGG16')
    parser.add_argument('--learning_rate', type=float, \
        help='Learning rate. Default is 0.001')
    parser.add_argument('--epochs', type=int, \
        help='Number of epochs. Default is 75')
    parser.add_argument('--save_path', type=str, \
        help='Name of the model save files. Default is best_model.pth')
    parser.add_argument('--freezing', type=bool, \
        help='Specify whether to freeze model parameters in pretrained model. Default is False')
    parser.add_argument('--batch_size', type=int, \
        help='number of batches for dataloader. Default is 64')
    parser.add_argument('--train_csv_path', type=str, \
        help='Specify the csv file containing the training data label. Default is train_clean.csv')
    parser.add_argument('--test_csv_path', type=str, \
        help='Specify the csv file containing the testing data label. Default is test_clean.csv')
    parser.add_argument('--root_dir', type=str, \
        help='Specify the csv folder containing image data. Default is sneaker_image_data_cropped/')


    args, _ = parser.parse_known_args()


    arch = 'vgg16'
    if args.arch:
        arch = args.arch

    learning_rate = 0.001
    if args.learning_rate:
        learning_rate = args.learning_rate

    n_epochs = 75
    if args.epochs:
        epochs = args.epochs

    freezing = False
    if args.freezing:
        freezing = args.freezing

    save_path = 'best_model.pth'
    if args.save_path:
        save_path = args.save_path

    batch_size = 64
    if args.batch_size:
        batch_size = args.batch_size

    train_csv_path='train_clean.csv'
    if args.train_csv_path:
        train_csv_path = args.train_csv_path

    test_csv_path='test_clean.csv'
    if args.train_csv_path:
        test_csv_path = args.test_csv_path

    root_dir='sneaker_image_data_cropped/'
    if args.root_dir:
        root_dir = args.root_dir

    trainloader, validloader, testloader, train_data = get_loaders(batch_size=batch_size, train_csv_path=train_csv_path, test_csv_path=test_csv_path, root_dir=root_dir)

    train_net(train_loader=trainloader, valid_loader=validloader, arch=arch, learning_rate=learning_rate, n_epochs=n_epochs, freezing=freezing, save_path=save_path)

if __name__ == '__main__':
    main()
