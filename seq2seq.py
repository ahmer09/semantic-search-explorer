from matplotlib import pyplot as plt
import tensorflow as tf
from keras import backend as K
from keras.layers import Input
from keras.models import Model
from IPython.display import SVG, display
from keras.utils.vis_utils import model_to_dot

def build_seq2seq_model(word_emb_dim,
                        hidden_state_dim,
                        encoder_seq_len,
                        num_encoder_tokens,
                        num_decoder_tokens):
    """
    Builds architecture for sequence to sequence model.

    Encoder and Decoder layer consist of one GRU layer each. User can specify
    dimensionality of the word embedding and the hidden state

    :param word_emb_dim: int
                dimensionality of word embedding
    :param hidden_state_dim:int
                dimensionality of hidden state in encoder and decoder
    :param encoder_seq_len: int
                the length of input sequence wrt encoder. The input
                sequence is all padded to same length.
    :param num_encoder_tokens: int
                the vocabulary size of the corpus relevant to encoder.
    :param num_decoder_tokens: int
                the vocabulary size of the corpus relavent to decoder.
    :return: Keras.models.Models
    """

    """ Encoder Model """
    encoder_inputs = Input(shape=(encoder_seq_len,), name='Encoder-Input')

    # Word Embedding for encoder (Ex: Issue title, Code)
    x = Embedding(num_encoder_tokens, word_emb_dim, name='Body-Word_Embedding', mask_zero=False)(encoder_inputs)
    x = BatchNormalization(name='Encoder-Batchnorm-1')(x)

    #We donot need Encoder output just the hidden state.
    _, state_h = GRU(hidden_state_dim, , return_state=True, name='Encoder-Last-GRU', dropout=0.5)(x)
    #Encapsulate the encoder as separate entity
    encoder_model = Model(inputs=encoder_inputs, outputs=state_h, name='Encoder-Model')

    seq2seq_encoder_out = encoder_model(encoder_inputs)

    """ Decoder Model """
    decoder_inputs = Input(shape=(None,), name='Decoder-Input')  # for teacher forcing

    # Word Embedding For Decoder (ex: Issue Titles, Docstrings)
    dec_emb = Embedding(num_decoder_tokens, word_emb_dim, name='Decoder-Word-Embedding', mask_zero=False)(decoder_inputs)
    dec_bn = BatchNormalization(name='Decoder-Batchnorm-1')(dec_emb)

    # Set up the decoder, using `decoder_state_input` as initial state.
    decoder_gru = GRU(hidden_state_dim, return_state=True, return_sequences=True, name='Decoder-GRU', dropout=.5)
    decoder_gru_output, _ = decoder_gru(dec_bn, initial_state=seq2seq_encoder_out)
    x = BatchNormalization(name='Decoder-Batchnorm-2')(decoder_gru_output)

    # Dense layer for prediction
    decoder_dense = Dense(num_decoder_tokens, activation='softmax', name='Final-Output-Dense')
    decoder_outputs = decoder_dense(x)

    #### Seq2Seq Model ####
    seq2seq_Model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    return seq2seq_Model

