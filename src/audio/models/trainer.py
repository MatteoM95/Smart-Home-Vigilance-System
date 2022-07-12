import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import tensorflow_model_optimization as tfmot

from tensorflow import keras
from sklearn.metrics import accuracy_score

from src.audio.utils.signal_generator import SignalGenerator
from models.models import set_model


class Trainer():

    def __init__(self, model_name=None, alpha=1, n_classes=None, pruning=False, input_shape=None):

        self.model_name = model_name
        self.n_classes = n_classes
        self.input_shape = (input_shape[1],input_shape[2],input_shape[3])

        #initialize optimization
        self.pruning = pruning
        self.alpha = alpha 

        self.model = set_model(self.model_name, self.n_classes, self.input_shape, self.alpha)

        # initialize
        self.tflite_path = None
        self.model_path = None


    def train_model(self, train_ds, val_ds, learning_rate, input_shape, num_epochs):

        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath='models_tflite', monitor='val_sparse_categorical_accuracy', save_best_only=True)
        lr_callback = tf.keras.callbacks.LearningRateScheduler(self.scheduler)
        callbacks = [cp_callback, lr_callback]
            
        pruning_params = {'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(initial_sparsity=0.30,
                                                                               final_sparsity=0.40,
                                                                               begin_step=len(train_ds) * 5,
                                                                               end_step=len(train_ds) * 15)}
        if self.pruning:
            
            prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude
            model = prune_low_magnitude(self.model,**pruning_params)
            pr_callback = tfmot.sparsity.keras.UpdatePruningStep()
            callbacks.append(pr_callback)
        
            self.model.build(input_shape=input_shape)
            self.model.compile(optimizer=tf.optimizers.Adam(learning_rate=learning_rate),
                        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                        metrics=keras.metrics.SparseCategoricalAccuracy())

            self.model.fit(train_ds, epochs=num_epochs, validation_data=val_ds, callbacks=callbacks)   
            model = tfmot.sparsity.keras.strip_pruning(model)
        
        else:
            self.model.build(input_shape=input_shape)
            self.model.compile(optimizer=tf.optimizers.Adam(learning_rate=learning_rate),
                        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                        metrics=keras.metrics.SparseCategoricalAccuracy())

            self.model.fit(train_ds, epochs=num_epochs, validation_data=val_ds)   


    def save_tf(self, path=''):

        self.model_path = path

        if not os.path.isdir(self.model_path):
            os.mkdir(self.model_path)

        self.model.save(self.model_path)


    def save_tflite(self, path='', optimization=None):
        
        self.tflite_path = path

        converter = tf.lite.TFLiteConverter.from_saved_model(self.model_path)
        converter.experimental_enable_resource_variables = True

        if optimization:
            converter.optimizations = optimization

        tflite_m = converter.convert()

        with open(self.tflite_path, 'wb') as fp:
            fp.write(tflite_m)


    def make_inference(self, ds):
        """Used in order to evaluate the performances for each class and eventual
        confusion among pair of classes
        """
        interpreter = tf.lite.Interpreter(model_path=self.tflite_path)
        interpreter.allocate_tensors()

        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        ds = ds.unbatch().batch(1)

        predictions, labels = [], []

        for x, y in ds:
            # give the input
            interpreter.set_tensor(input_details[0]["index"], x)
            interpreter.invoke()

            # predict and get the current ground truth
            curr_prediction_logits = interpreter.get_tensor(output_details[0]['index']).squeeze()
            curr_label = y.numpy().squeeze()

            curr_prediction = np.argmax(curr_prediction_logits)

            predictions.append(curr_prediction)
            labels.append(curr_label)

        confusion_matrix = tf.math.confusion_matrix(labels, predictions)  # add names!

        # validation set is balanced
        accuracy = accuracy_score(labels, predictions)

        return confusion_matrix, accuracy


    def plot_stats(self, cm, labels):

        print("\n ==== STATS ====")

        print("Model size = {} kB".format(round(os.path.getsize(self.tflite_path) / 1024, 2)))

        # confusion matrix
        ax = plt.subplot()
        sns.heatmap(cm, annot=True, fmt='g', ax=ax)
        ax.xaxis.set_ticklabels(labels)
        ax.yaxis.set_ticklabels(labels)
        plt.xticks(rotation=90)
        plt.yticks(rotation=0)
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.title('Heatmap')
        plt.tight_layout()
        plt.savefig('heatmap_{}.png')