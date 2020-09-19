# # TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import numpy as np


def train_net(n_epochs, model, train_loader, valid_loader, criterion, optimizer, scheduler, device, save_path):
    # prepare the net for training
    model.train()
    step = 0
    valid_loss_min = np.Inf

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
            #if step % print_every == 0:
        
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
                
            # calculates the average %error between the predicted and actual prices of the validation batch
            error = [abs(predicted_labels[i] - labels[i]) / labels[i] for i in range(len(predicted_labels))]
            error = (sum(error) / len(predicted_labels)).cpu().numpy()[0] * 100
        
        # quantify the train and validation losses
        train_loss = int(training_loss / len(train_loader))
        validation_loss = int(validation_loss / len(valid_loader))
        
        # LR scheduler
        scheduler.step(validation_loss)
        
        # save the training and validation losses for plotting learning curves
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
        
        # uncomment this line if you want the model to stop after 3 consecutive decreases in LR (sign of convergence)
        '''if num_decrease == 3:
            break'''

    print()
    print('Finished Training')
    
    return model, training_losses, validation_losses
