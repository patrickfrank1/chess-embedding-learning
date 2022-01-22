import chess
import yaml
import json
from pathlib import Path
import numpy as np

import tensorflow as tf
from tensorflow import keras

from chesspos.preprocessing import SampleGenerator, tensor_to_board, board_to_tensor
from chesspos.models import ResnetAutoencoder

evaluate_data_params = yaml.safe_load(open(Path(__file__).with_name('params_tensor.yaml')))['evaluate']['data']
evaluate_eval_params = yaml.safe_load(open(Path(__file__).with_name('params_tensor.yaml')))['evaluate']['eval']
model_params = yaml.safe_load(open(Path(__file__).with_name('params_tensor.yaml')))['train']['model']

sample_preprocessor = lambda samples: tuple([samples.reshape((*samples.shape, 1)), samples.reshape((*samples.shape, 1))])

test_generator = SampleGenerator(
	sample_dir = evaluate_data_params['test_dir'],
	sample_preprocessor = sample_preprocessor,
	batch_size = evaluate_data_params['test_batch_size'],
	sample_type = np.float32,
)

autoencoder = ResnetAutoencoder(
    train_generator=test_generator,
	test_generator=test_generator,
	train_steps_per_epoch=model_params['train_steps_per_epoch'],
	test_steps_per_epoch=model_params['test_steps_per_epoch'],
	save_dir=model_params['save_dir'],
	output_to_board = tensor_to_board,
	board_to_input = board_to_tensor,
	loss = model_params['loss'],
	tf_callbacks = model_params['tf_callbacks'] + [
		keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.03, patience=5, verbose=1, mode='min', restore_best_weights=True),
		#keras.callbacks.TensorBoard(log_dir=model_params['save_dir'], histogram_freq=1, write_images=True, embeddings_freq=1)
	]
)

autoencoder.load()
examples = evaluate_eval_params['number_examples']
examples_out = ""

start_board = chess.Board()
prediction = autoencoder.predict_from_board([start_board])
prediction = prediction[0,:,:,:,0]
prediction = autoencoder.binarize_array(prediction)
print(prediction[:,:,0])
print(prediction[:,:,1])
print(tensor_to_board(prediction).__str__())

# best_samples = autoencoder.get_best_samples(examples)
# for i in range(examples):
#     best_sample_input = best_samples[i]['input']
#     examples_out += str(best_samples[i]['loss']) + '\n\n'
#     examples_out += autoencoder.compare_input_to_prediction(best_sample_input)

# worst_samples = autoencoder.get_worst_samples(examples)
# for i in range(examples):
#     worst_sample_input = worst_samples[i]['input']
#     examples_out += str(worst_samples[i]['loss']) + '\n'
#     examples_out += autoencoder.compare_input_to_prediction(worst_sample_input)

#print(examples_out)

# with open(evaluate_eval_params['result_dir']+'/examples.out', 'w') as file:
#     file.write(examples_out)


# with open(evaluate_eval_params['result_dir']+'/scores.json', 'w') as file:
#     json.dump(
# 		{
# 			'train_loss': autoencoder.train_history['loss'][-1],
# 			'test_loss': autoencoder.train_history['val_loss'][-1]
# 		},
# 		file,
# 		indent=2
# 	)

# with open(evaluate_eval_params['result_dir']+'/test_scores.json', 'w') as file:
#     json.dump(
# 		{
# 			'val_loss': [{
# 				'epoch': i,
# 				'train_loss': autoencoder.train_history['loss'][i],
# 				'val_loss':  autoencoder.train_history['val_loss'][i]
# 			} for i in range(len(autoencoder.train_history['val_loss']))]
# 		},
# 		file,
# 		indent=2
# 	)
