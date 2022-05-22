
import os
import tensorflow as tf
from model import CRNN
from util import *
import wandb
from wandb.keras import WandbCallback
import argparse
import json


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compute scaling parameters from the training set')

    # Paths
    parser.add_argument('-d', '--data_dir', type=str, help="Data directory", default = "../../data/")
    parser.add_argument('-i', '--input_dir', type=str, help="Input directory", default = "dataset/")
    parser.add_argument('-sf', '--scaling_file', type=str, help="Scaling parameters file", default = "scaling_parameters.json")
    parser.add_argument('-m', '--model_dir', type=str, help="Directory where model weights are stored", default = "../models/")

    parser.add_argument('-lr', '--learning_rate', type=float, help="Learning rate", default = 1e-3)
    parser.add_argument('-e', '--epochs', type=int, help="Number of epochs", default = 50)
    parser.add_argument('-bs', '--batch_size', type=int, help="Training batch size", default = 32)
    parser.add_argument('-k', '--max_k', type=int, help="Maximum number of speakers", default = 10)
    parser.add_argument('-s', '--seconds', type=int, help="Durationn of sample", default = 5)

    args = parser.parse_args()

    DATASET_DIR = os.path.join(args.data_dir, args.input_dir)
    SCALING_FILE = os.path.join(args.data_dir, args.scaling_file)

    # Weights & Biases
    wandb.init(project="test-project", entity="speaker-estimation")
    wandb.config = {
        "learning_rate": args.learning_rate,
        "epochs": args.epochs,
        "batch_size": args.batch_size
    }
    
    # Get Scaling Parameters
    with open(SCALING_FILE, 'r') as f:
        scaling_parameters = json.load(f)

    mean = scaling_parameters["mean"]
    std = scaling_parameters["std"]

    # Data Generators
    train_files = [f for f in os.listdir(os.path.join(DATASET_DIR, "train/")) if f.endswith(".wav")]
    valid_files = [f for f in os.listdir(os.path.join(DATASET_DIR, "valid/")) if f.endswith(".wav")]

    train_generator = CustomDataGenerator(os.path.join(DATASET_DIR, "train/"), 
                                        train_files, dim=(500, 201), 
                                        max_k=args.max_k, batch_size=args.batch_size, 
                                        mean=mean, std=std, s=args.seconds, 
                                        train=True, shuffle=True)

    valid_generator = CustomDataGenerator(os.path.join(DATASET_DIR, "valid/"), 
                                        valid_files, dim=(500, 201), 
                                        max_k=args.max_k, batch_size=1, 
                                        mean=mean, std=std, s=args.seconds,
                                        train=False, shuffle=False)

    # Model
    model = CRNN((500, 201, 1), args.max_k)
    print(model.summary())

    # Compile
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-8),
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=tf.keras.metrics.CategoricalAccuracy(),
    )

    # Callbacks
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(args.model_dir, 'weights.{epoch:02d}-{val_loss:.2f}.h5',),
        save_weights_only=True,
        monitor='val_categorical_accuracy',
        mode='max',
        save_best_only=True)

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=10,
    )

    wandb_callback = WandbCallback(save_model=False)

    # Model directory
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)

    # Train
    model.fit(train_generator, validation_data=valid_generator, epochs=args.epochs, callbacks=[checkpoint, early_stopping, wandb_callback])
    model.save_weights(os.path.join(args.model_dir, 'final_weights.h5'))