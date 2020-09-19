import urllib.request
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import models
from torch import nn
from collections import OrderedDict

# function to visualize a random image from a dataloader
def loader_show_image(dataloader):
    image, label = next(iter(dataloader))

    image = np.array(image[0])
    label = np.array(label[0])

    plt.title('The true price of this shoe is {}'.format(label))
    plt.imshow(np.transpose(image, (1, 2, 0)))
    plt.axis('off');

# function to visual 5 image/label pairs after passing through the model once
def visualize_output(test_images, pred_outputs, actual_outputs=None, batch_size=5):
    for i in range(batch_size):
        plt.figure(figsize=(30, 20))
        ax = plt.subplot(1, batch_size, i + 1)

        # un-transform the image data
        unnorm = UnNormalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        image = test_images[i].cpu()  # get the image from it's Variable wrapper
        image = unnorm(image)
        image = image.numpy()  # convert to numpy array from a Tensor
        image = np.transpose(image, (1, 2, 0))  # transpose to go from torch to numpy image

        # un-transform the predicted price data
        predicted_output = pred_outputs[i].data.cpu()
        predicted_output = predicted_output.numpy()[0]
        predicted_output = abs(float(predicted_output))

        # plot ground truth points for comparison, if they exist
        if actual_outputs is not None:
            actual_output_label = actual_outputs[i][0].data.cpu()

        # call show_all_keypoints
        plt.title('The predicted price for this shoe is \${}, its actual price is \${}'.format(np.round(predicted_output, 2),
                                                                                          np.round(actual_output_label, 2)))
        plt.imshow(np.squeeze(image))

        plt.axis('off');

    plt.show()

# pipeline function to pass a batch of data from the test_loader and visualizing the results
def model_sample_output(test_loader, model, device):
    # iterate through the test dataset
    for i, sample in enumerate(test_loader):

        # get sample data: images and ground truth keypoints
        images = sample[0]
        labels = sample[1]
        # convert images to FloatTensors
        images = images.type(torch.FloatTensor).to(device)

        # forward pass to get net output
        output_preds = model.forward(images)

        output_preds = output_preds.view(output_preds.size()[0], -1).to(device)
        labels = labels.view(output_preds.size()[0], -1).to(device)

        # break after first image is tested
        if i == 0:
            visualize_output(images, output_preds, labels)

#class used to unormalize images for display            
class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor

#function used to load a saved checkpoint during model inference
def load_checkpoint(path):
    checkpoint = torch.load(path, map_location=torch.device('cpu'))
    model = models.vgg16(pretrained=True)

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

    print('loading model...')
    model.load_state_dict(checkpoint)
    print('model loaded!')

    return model

# function to test the loss and error of a trained model on the test_loader
def test_model(model, test_loader, device, criterion):
    test_loss = 0
    model.eval()

    with torch.no_grad():
        print('testing...')
        for images, labels in tqdm(test_loader):

            labels = labels.view(labels.size()[0], -1)

            labels = labels.type(torch.FloatTensor).to(device)
            images = images.type(torch.FloatTensor).to(device)

            predicted_labels = model.forward(images)
            batch_loss = criterion(predicted_labels, labels)

            test_loss += batch_loss.item()  

        error = [abs(predicted_labels[i] - labels[i]) / labels[i] for i in range(len(predicted_labels))]
        error = (sum(error) / len(predicted_labels)).cpu().numpy()[0] * 100       

        print(f"Test Loss: {test_loss/len(test_loader):.3f}.."
              "\n"
              f"Test Error: {error:.2f}..")