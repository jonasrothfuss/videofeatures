from Pipeline import loadFeatures

from keras.models import Model
from keras.layers.core import Dense, Activation
from keras.layers import Input, LSTM, Dense, Concatenate
import numpy as np
from keras.utils import plot_model

RESNET_FEAT_PATH = '/common/homes/students/rothfuss/Documents/video_retrieval_baselines/DataDumps/Features/features_resnet_20bn_val.pickle'

VGG_FEAT_PATH = '/common/homes/students/rothfuss/Documents/video_retrieval_baselines/DataDumps/Features/features_vgg_fc1_20bn_val.pickle'

class CompositeLSTMModel:

  def __init__(self, n_latent_dims=2048, n_feature_dim=2048):

    self.n_feature_dim = n_feature_dim
    self.n_latent_dims = n_latent_dims

    # Define an input sequence and process it.
    encoder_inputs = Input(shape=(None, n_feature_dim))

    ''' ENCODER '''
    #LSTM 1
    self.encoder_LSTM1 = encoder_LSTM1 = LSTM(n_latent_dims, return_state=False, return_sequences=True)
    encoder_LSTM1_out = encoder_LSTM1(encoder_inputs)

    #LSTM 2
    self.encoder_LSTM2 = encoder_LSTM2 = LSTM(n_latent_dims, return_state=True)
    _ , state_h, state_c = encoder_LSTM2(encoder_LSTM1_out)
    encoder_states = [state_h, state_c]

    ''' DECODER FUTURE '''
    decoder_inputs = Input(shape=(None, n_feature_dim))

    decoder_future_LSTM1 = LSTM(n_latent_dims, return_state=False, return_sequences=True)
    decoder_future_LSTM1_out = decoder_future_LSTM1(decoder_inputs, initial_state=encoder_states)

    decoder_future_LSTM2 = LSTM(n_latent_dims, return_sequences=True, return_state=False)
    decoder_future_LSTM2_out = decoder_future_LSTM2(decoder_future_LSTM1_out)

    decoder_futue_dense = Dense(n_feature_dim)
    decoder_futue_outputs = decoder_futue_dense(decoder_future_LSTM2_out)

    ''' DECODER Reconst'''

    decoder_reconst_LSTM1 = LSTM(n_latent_dims, return_state=False, return_sequences=True)
    decoder_reconst_LSTM1_out = decoder_reconst_LSTM1(decoder_inputs, initial_state=encoder_states)

    decoder_reconst_LSTM2 = LSTM(n_latent_dims, return_sequences=True, return_state=False)
    decoder_reconst_LSTM2_out = decoder_reconst_LSTM2(decoder_reconst_LSTM1_out)

    decoder_reconst_dense = Dense(n_feature_dim)
    decoder_reconst_outputs = decoder_reconst_dense(decoder_reconst_LSTM2_out)

    pred = Concatenate(axis=1)([decoder_reconst_outputs, decoder_futue_outputs])


    # Define the model that will turn
    # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
    self.model_train = Model([encoder_inputs, decoder_inputs], pred)

    self.model_train.compile(optimizer='Adam', loss='mean_squared_error')

    self.model_train.summary()


  def fit(self, low_level_features):
    encoder_input_data = low_level_features[:, :10]
    decoder_input_data = np.zeros(encoder_input_data.shape)

    self.model_train.fit([encoder_input_data, decoder_input_data], low_level_features,
              batch_size=100,
              epochs=100,
              validation_split=0.2)


    ''' Validation - Encoder Model'''

    encoder_inputs = Input(shape=(None, self.n_feature_dim))

    ''' ENCODER '''
    # LSTM 1
    encoder_LSTM1_val = LSTM(self.n_latent_dims, return_state=False, return_sequences=True, weights=self.encoder_LSTM1.get_weights())
    encoder_LSTM1_val_out = encoder_LSTM1_val(encoder_inputs)

    # LSTM 2
    encoder_LSTM2_val = LSTM(self.n_latent_dims, return_state=True, weights=self.encoder_LSTM2.get_weights())
    _, state_h, state_c = encoder_LSTM2_val(encoder_LSTM1_val_out)
    latent_repr = Concatenate(axis=-1)([state_h, state_c])

    self.model_val = Model(encoder_inputs, latent_repr)
    self.model_val.compile(optimizer='Adam', loss='mean_squared_error')

    self.fitted = True

  def generate_latent_repr(self, low_level_features, labels=None, dump_path=None):
    assert self.fitted
    latent_reps = self.model_val.predict(low_level_features)

    if dump_path:
      np.save(dump_path + '.npy', latent_reps)
      np.save(dump_path + '_labels.npy', labels)

    return latent_reps




if __name__ == '__main__':

  ''' RESNET Features '''
  
  # """ DATA PREP"""
  # low_level_features, labels = loadFeatures(RESNET_FEAT_PATH)
  #
  # num_encoder_tokens = low_level_features.shape[-1]
  # low_level_features = low_level_features.reshape((-1, low_level_features.shape[1], num_encoder_tokens))
  #
  # print("Data Shape: ", low_level_features.shape)
  #
  # model = CompositeLSTMModel()
  # model.fit(low_level_features)
  #
  # dump_path = '/common/homes/students/rothfuss/Documents/video_retrieval_baselines/DataDumps/FisherVectors/fv_lstm_resnet_20bn_val'
  # feat = model.generate_latent_repr(low_level_features, labels=labels, dump_path=dump_path)
  # print('Dumped LSTM features to:', dump_path)
  #
  # print(feat.shape)
  #
  # del model
  # del feat
  #

  ''' VGG Features'''

  low_level_features, labels = loadFeatures(VGG_FEAT_PATH)

  num_encoder_tokens = low_level_features.shape[-1]
  low_level_features = low_level_features.reshape((-1, low_level_features.shape[1], num_encoder_tokens))

  print("Data Shape: ", low_level_features.shape)

  model = CompositeLSTMModel(n_feature_dim=4096)
  model.fit(low_level_features)

  dump_path = '/common/homes/students/rothfuss/Documents/video_retrieval_baselines/DataDumps/FisherVectors/fv_lstm_vgg_20bn_val'
  feat = model.generate_latent_repr(low_level_features, labels=labels, dump_path=dump_path)
  print('Dumped LSTM features to:', dump_path)

  print(feat.shape)

