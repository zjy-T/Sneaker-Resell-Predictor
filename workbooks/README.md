# Jupyter Workbooks

Book 1: Data loading and cleaning (you can skip this step if you are just using my data)
* reading in price data csv files from StockX and Farfetch
* merging the two csv files to create the raw data file
* removing entries that either don't have a price or don't have an image
* removing entries with prices deemed as outliers
* separating the dataset into train and test/split files

Book 2: Model training
* setting up the dataloaders and data pipelines in preparation for training
* visualizing the dataloader
* instantiate transfer learning model and editing the last classifier layer to fit our problem
* set-up hyperparameters
* single pass through of data to the model before training to visualize performance and to ensure pipeline/model are properly set-up
* train
* visualize the train/validation learning curves
* pass testing data to visualize performance

Book 3: Model prediction
* Load trained model (.pth file)
* set-up data pipeline for prediction
* make a prediction on your desired sneaker image

make sure training.py, helper.py, and custom_loader.py are present or else code will not work
