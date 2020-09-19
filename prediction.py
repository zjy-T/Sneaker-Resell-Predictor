from torchvision.transforms import transforms
from PIL import Image
import numpy as np
from helper import load_checkpoint
import torch
import argparse

def predict(model_path, image_path):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_checkpoint(model_path).to(device)

    img_transforms = transforms.Compose([transforms.Resize((224, 336)),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406],
                                                              [0.229, 0.224, 0.225])])

    image = img_transforms(Image.open(image_path)).unsqueeze(0).to(device)

    model.eval()
    # forward pass to get net output
    output_preds = model.forward(image)

    output_pred = output_preds.view(output_preds.size()[0], -1).to(device)

    # un-transform the predicted key_pts data
    predicted_output = output_pred.data.cpu()
    predicted_output = predicted_output.numpy()[0]
    predicted_output = abs(float(predicted_output))

    print('The predicted price of this shoe is ${} USD'.format(np.round(predicted_output, 2)))


def main():
    parser = argparse.ArgumentParser(description='Predicting a new network on a data set')

    parser.add_argument('--image_path', type=str, help='Path of the image to be predicted')
    parser.add_argument('--model_path', type=str, help='Path of the trained model')

    args, _ = parser.parse_known_args()

    image_path = 'test2.jpg'
    if args.image_path:
        image_path = args.image_path

    model_path = 'best_model_vgg_nf.pth'
    if args.model_path:
        checkpoint_path = args.model_path


    predict(image_path=image_path, model_path=model_path)

if __name__ == '__main__':
    main()
