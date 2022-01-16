import yaml
from pathlib import Path
import numpy as np

import tensorflow as tf
from tensorflow import keras

from chesspos.preprocessing import SampleGenerator
from chesspos.models import CnnResnetAutoencoder

data_params = yaml.safe_load(open(Path(__file__).with_name('params_tensor.yaml')))['train']['data']
model_params = yaml.safe_load(open(Path(__file__).with_name('params_tensor.yaml')))['train']['model']


sample_preprocessor = lambda samples: tuple([samples.reshape((*samples.shape, 1)), samples.reshape((*samples.shape, 1))])
autoencoder_postprocessor = lambda tensor: tensor.reshape(tensor.shape[:-1])

train_generator = SampleGenerator(
	sample_dir = data_params['train_dir'],
	sample_preprocessor = sample_preprocessor,
	batch_size = data_params['train_batch_size'],
	sample_type = bool
)

test_generator = SampleGenerator(
	sample_dir = data_params['test_dir'],
	sample_preprocessor = sample_preprocessor,
	batch_size = data_params['test_batch_size'],
	sample_type = bool
)

autoencoder = CnnResnetAutoencoder(
	input_size = model_params['input_size'],
	embedding_size = model_params['embedding_size'],
	hidden_layers = model_params['hidden_layers'],
	loss = model_params['loss'],
	train_generator = train_generator,
	test_generator = test_generator,
	save_dir = model_params['save_dir'],
	train_steps_per_epoch = model_params['train_steps_per_epoch'],
	test_steps_per_epoch = model_params['test_steps_per_epoch'],
	tf_callbacks = model_params['tf_callbacks'] + [
		keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.03, patience=5, verbose=1, mode='min', restore_best_weights=True),
		keras.callbacks.TensorBoard(log_dir=model_params['save_dir'], histogram_freq=1, write_images=True, embeddings_freq=1)
	]
)

autoencoder.build_model()
autoencoder.compile()
history = autoencoder.train()
autoencoder.save()
