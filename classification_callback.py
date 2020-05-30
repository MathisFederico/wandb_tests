import numpy as np
from sklearn.metrics import confusion_matrix

import wandb
from wandb.keras import WandbCallback

import plotly.express as px
import plotly.graph_objs as go

class WandbClassificationCallback(WandbCallback):

    def __init__(self, monitor='val_loss', verbose=0, mode='auto',
                 save_weights_only=False, log_weights=False, log_gradients=False,
                 save_model=True, training_data=None, validation_data=None,
                 labels=[], data_type=None, predictions=1, generator=None,
                 input_type=None, output_type=None, log_evaluation=False,
                 validation_steps=None, class_colors=None, log_batch_frequency=None,
                 log_best_prefix="best_", log_confusion_matrix=False, wrong_predictions=0):
        
        super().__init__(monitor=monitor,
                        verbose=verbose, 
                        mode=mode,
                        save_weights_only=save_weights_only,
                        log_weights=log_weights,
                        log_gradients=log_gradients,
                        save_model=save_model,
                        training_data=training_data,
                        validation_data=validation_data,
                        labels=labels,
                        data_type=data_type,
                        predictions=predictions,
                        generator=generator,
                        input_type=input_type,
                        output_type=output_type,
                        log_evaluation=log_evaluation,
                        validation_steps=validation_steps,
                        class_colors=class_colors,
                        log_batch_frequency=log_batch_frequency,
                        log_best_prefix=log_best_prefix)
                        
        self.log_confusion_matrix = log_confusion_matrix
        self.wrong_predictions = wrong_predictions
               
    def on_epoch_end(self, epoch, logs={}):
        if self.generator:
            self.validation_data = next(self.generator)

        if self.log_weights:
            wandb.log(self._log_weights(), commit=False)

        if self.log_gradients:
            wandb.log(self._log_gradients(), commit=False)
        
        if self.log_confusion_matrix:
            if self.validation_data is None:
                wandb.termwarn(
                    "No validation_data set, pass a generator to the callback.")
            elif self.validation_data and len(self.validation_data) > 0:
                wandb.log(self._log_confusion_matrix(), commit=False)

        if self.input_type in ("image", "images", "segmentation_mask") or self.output_type in ("image", "images", "segmentation_mask"):
            if self.validation_data is None:
                wandb.termwarn(
                    "No validation_data set, pass a generator to the callback.")
            elif self.validation_data and len(self.validation_data) > 0:
                wandb.log({"examples": self._log_images(
                    num_images=self.predictions)}, commit=False)

        wandb.log({'epoch': epoch}, commit=False)
        wandb.log(logs, commit=True)

        self.current = logs.get(self.monitor)
        if self.current and self.monitor_op(self.current, self.best):
            if self.log_best_prefix:
                wandb.run.summary["%s%s" % (self.log_best_prefix, self.monitor)] = self.current
                wandb.run.summary["%s%s" % (self.log_best_prefix, "epoch")] = epoch
                if self.verbose and not self.save_model:
                    print('Epoch %05d: %s improved from %0.5f to %0.5f' % (
                        epoch, self.monitor, self.best, self.current))
            if self.save_model:
                self._save_model(epoch)
            self.best = self.current
        
    def _log_confusion_matrix(self):
        x_val = self.validation_data[0]
        y_val = self.validation_data[1]
        y_val = np.argmax(y_val, axis=1)
        y_pred = np.argmax(self.model.predict(x_val), axis=1)

        confmatrix = confusion_matrix(y_pred, y_val, labels=range(len(self.labels)))
        confdiag = np.eye(len(confmatrix)) * confmatrix
        np.fill_diagonal(confmatrix, 0)

        confmatrix = confmatrix.astype('float')
        confmatrix[confmatrix == 0] = np.nan
        confmatrix = go.Heatmap({'coloraxis': 'coloraxis1', 'x': self.labels, 'y': self.labels, 'z': confmatrix, 'hoverongaps':False, 'hovertemplate': 'y_true: %{x}<br>y_pred: %{y}<br>Count: %{z}<extra></extra>'})

        confdiag = confdiag.astype('float')
        confdiag[confdiag == 0] = np.nan
        confdiag = go.Heatmap({'coloraxis': 'coloraxis2', 'x': self.labels, 'y': self.labels, 'z': confdiag, 'hoverongaps':False, 'hovertemplate': 'y_true: %{x}<br>y_pred: %{y}<br>Count: %{z}<extra></extra>'})

        fig = go.Figure((confdiag, confmatrix))
        transparent = 'rgba(0, 0, 0, 0)'
        fig.update_layout({'coloraxis1': {'colorscale': [[0, transparent], [0, 'rgba(180, 0, 0, 0.1)'], [1, 'rgba(180, 0, 0, 1)']], 'showscale': False}})
        fig.update_layout({'coloraxis2': {'colorscale': [[0, transparent], [0, 'rgba(0, 180, 0, 0.1)'], [1, 'rgba(0, 180, 0, 1)']], 'showscale': False}})

        xaxis = {'title':{'text':'y_true'}, 'showticklabels':False}
        yaxis = {'title':{'text':'y_pred'}, 'showticklabels':False}

        fig.update_layout(title={'text':'Confusion matrix', 'x':0.5}, paper_bgcolor=transparent, plot_bgcolor=transparent, xaxis=xaxis, yaxis=yaxis)
        
        return {'confusion_matrix': wandb.data_types.Plotly(fig)}


import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

# input image dimensions
img_rows, img_cols = 28, 28
num_classes = 10

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

hyperparameter_defaults = dict(
  dropout_1 = 0.25,
  dropout_2 = 0.5,
  learning_rate = 1.0,
  epochs = 20,
  batch_size = 128,
  conv2d_1_filters = 32,
  conv2d_1_kernel_size= 3,
  conv2d_2_filters = 32,
  conv2d_2_kernel_size = 3,
  dense_1_units = 128,
  pool_1_size = 2,
)

run = wandb.init(entity="mathisfederico", project="wandb_features", group="confusion_matrix", config=hyperparameter_defaults, resume=False, force=True)
if run is None:
    raise ValueError("Wandb didn't initialize properly")
config = wandb.config

model = Sequential()
model.add(Conv2D(config.conv2d_1_filters, kernel_size=(config.conv2d_1_kernel_size, config.conv2d_1_kernel_size),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(config.conv2d_2_filters, (config.conv2d_2_kernel_size, config.conv2d_2_kernel_size), activation='relu'))
model.add(MaxPooling2D(pool_size=(config.pool_1_size, config.pool_1_size)))
model.add(Dropout(config.dropout_1))
model.add(Flatten())
model.add(Dense(config.dense_1_units, activation='relu'))
model.add(Dropout(config.dropout_2))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(config.learning_rate),
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=config.batch_size,
          epochs=config.epochs,
          verbose=1,
          validation_data=(x_test, y_test),
          callbacks=[
              WandbClassificationCallback(log_confusion_matrix=True, validation_data=(x_test, y_test), labels=list(range(10))),
              ]
          )
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])