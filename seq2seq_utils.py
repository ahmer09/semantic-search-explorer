from matplotlib import pyplot as plt
import tensorflow as tf
from keras import backend as K
from keras.layers import Input
from keras.models import Model
from IPython.display import SVG, display
from keras.utils.vis_utils import model_to_dot
import logging
import numpy as np
import dill as dpickle
from annoy import AnnoyIndex
from tqdm import tqdm, tqdm_notebook
from random import random
from nltk.translate.bleu_score import corpus_bleu


def load_text_processor(fname='title_pp.dpkl'):
    """
    Load preprocessors from the disk.

    :param fname: str
                file name for ktext processor object
    :return:
    num_tokens: int
            size of vocabulary loaded into ktext.processor
    pp: ktext.processor
        the processor you are trying to load

    Typical Usage:
        num_decoded_tokens = title_pp = load_text_processor(fname='title_pp.dpkl')
    """
    #Loads files from disk
    with open(fname, 'rb') as f:
        pp = dpickle.load(f)

    num_tokens = max(pp.id2token.keys()) + 1
    print(f'Size of Vocabulary for {fname}:{num_tokens:,}')
    return num_tokens, pp

def load_decoder_inputs(decoder_np_vecs='train_little_vecs.npy'):
    """
    Load decoder inputs.

    :param decoder_np_vecs: str
        filename of serialized.array of decoder input (issue title)
    :return:
    decoder_input_data: numpy.array
        The data fed into the decoder as input during training for teacher forcing.
        This is same as 'decoder_np_vecs' except the last position
    decoder_target_data: numpy.array
        The data that the decoder is trained to generate (issue title)
        Calculated by sliding 'decoder_np_vecs' by one position
    """
    vectorized_title = np.load(decoder_np_vecs)
    # For decoder input, we don't need the last word as it is for prediction
    # when we are training using techer forcing
    decoder_input_data = vectorized_title[:, :-1]

    # Decoder Target Data is ahead by 1 time step from Decoder Input Data (Teacher Forcing)
    decoder_target_data = vectorized_title[:, 1:]

    print(f'Shape of decoder input: {decoder_input_data.shape}')
    print(f'Shape of decoder target: {decoder_target_data.shape}')

    return decoder_input_data, decoder_target_data

def load_encoder_inputs(encoder_np_vecs='train_body_vecs.npy'):
    """
    Load variable and data that are input to decoder.

    :param encoder_np_vecs: str
        filename of serialized numpy.array of encoder input

    :return:
    encoder_input_data: numpy.array
        The issue body
    doc_length: int
        The standard document length of the input for the encoder after padding
        the shape of the array (num_examples, doc_length)
    """
    vectorized_body = np.load(encoder_np_vecs)
    encoder_input_data = vectorized_body
    doc_length = encoder_input_data.shape[1]
    print(f'Shape of encoder input: {encoder_input_data.shape}')
    return encoder_input_data, doc_length

def viz_model_architecture(model):
    """Visualize model architecture"""
    display(SVG(model_to_dot(model).create(prog='dot', format='svg')))

def free_gpu_mem():
    """Attempts to free gpu memory"""
    K.get_session().close()
    cfg = K.tf.ConfigProto()
    cfg.gpu_options.allow_growth = True
    K.set_session(K.tf.Session(config=cfg))

def test_gpu():
    """Runs a toy test computation task to test GPU"""
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)
    hello = tf.constant('Hello Tensorflow!')
    print(session.run(hello))

def plot_model_training_history(history_object):
    """plot model train vs validation loss"""
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.plot(history_object.history['loss'])
    plt.plot(history_object.history['val_loss'])
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

def extract_encoder_model(model):
    """
    Extract encoder from the original Sequence to Sequence model

    Returns a keras model object that has one input (body of issue)
    and one output, which is the last hidden state.
    :param model:
    :return:
    """