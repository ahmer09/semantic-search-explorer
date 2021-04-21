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

    :param model: keras model object
    :return:
    keras model object
    """
    encoder_model = model.get_layer('Encoder-Model')
    return encoder_model

def extract_decoder_model(model):
    """
    Extract the decoder from original model

    :param model: keras model  object
    :return:
    keras model object
    Inputs of Keras Model That Is Returned:
    1: the embedding index for the last predicted word or the <Start> indicator
    2: the last hidden state, or in the case of the first word the hidden state from the encoder

    Outputs of Keras Model That Is Returned:
    1.  Prediction (class probabilities) for the next word
    2.  The hidden state of the decoder, to be fed back into the decoder at the next time step

    """
    latent_dim = model.get_layer('Decoder-Word-Embedding').output_shape[-1]

    # Reconstruct the input into the decoder
    decoder_inputs = model.get_layer('Decoder-Input').input
    dec_emb = model.get_layer('Decoder-Word_Embedding')(decoder_inputs)
    dec_bn = model.get_layer('Decoder-Batchnorm-1')(dec_emb)

    # Instead of setting input state from the encoder, during inference we are not doing teacher forcing,
    # so we have a feedback loop from the predictions back into the GRU.
    gru_inference_state_input = Input(shape=(latent_dim,), name='hidden_state_input')

    gru_out, gru_state_out = model.get_layer('Decoder-GRU')([dec_bn, gru_inference_state_input])

    # Reconstruct dense layer
    dec_bn2 = model.get_layer('Decoder-Batchnorm-2')(gru_out)
    dense_out = model.get_layer('Final-output-Dense')(dec_bn2)
    decoder_model = Model([decoder_inputs, gru_inference_state_input], [dense_out, gru_state_out])
    return decoder_model

class Seq2Seq_Inference(object):
    def __int__(self, encoder_preprocessor, decoder_preprocessor, seq2seq_model):
        self.pp_body = encoder_preprocessor
        self.pp_title = decoder_preprocessor
        self.seq2seq_model = seq2seq_model
        self.encoder_model = extract_encoder_model(seq2seq_model)
        self.decoder_model = extract_decoder_model(seq2seq_model)
        self.default_max_len_title = self.pp_title.padding_maxlen
        self.nn = None
        self.rec_def = None

    def generate_issue_title(self, raw_input_text, max_len_title=None):
        """
        Use the seq2seq model to generate a title given the body of issue

        param: raw_input_text: str
            The body of isssue text as an input string

        param: max_len_title: int (optional)
            The maximum length of the title the model will generate
        """
        if max_len_title is None:
            max_len_title = self.default_max_len_title
        # get encoder features for the decoder
        raw_tokenized = self.pp_body.transform([raw_input_text])
        body_encoding = self.encoder_model.predict(raw_tokenized)
        # we will save the encoder's embedding before its updated by decoder
        original_body_encoding = body_encoding
        state_value = np.array(self.pp_title.token2id['_start_']).reshape(1, 1)

        decoded_sentence = []
        stop_condition = False
        while not stop_condition:
            preds, st = self.decoder_model.predict([state_value, body_encoding])
            pred_idx = np.argmax(preds[:, :, 2:]) + 2

            # retrieve word from index prediction
            pred_word_str = self.pp_title.token2id[pred_idx]

            if pred_word_str == '_end_' or len(decoded_sentence) >= max_len_title:
                stop_condition = True
                break
            decoded_sentence.append(pred_word_str)

            # update decoder for next word
            body_encoding = st
            state_value = np.array(pred_idx).reshape(1, 1)
        return original_body_encoding, ''.join(decoded_sentence)


    def print_example(self, i, body_text, title_text, url, threshold):
        """
        Prints an example for the model's prediction
        """
        if i:
            print('\n\n==============================================')
            print(f'============== Example # {i} =================\n')

        if url:
            print(url)

        print(f"Issue Body:\n {body_text} \n")

        if title_text:
            print(f"Original Title:\n {title_text}")

        emb, gen_title = self.generate_issue_title(body_text)
        print(f"\n****** Machine Generated Title (Prediction) ******:\n {gen_title}")

        if self.nn:
            # return neighbors and distances
            n, d = self.nn.get_nns_by_vector(emb.flatten(), n=4,
                                             include_distances=True)
            neighbors = n[1:]
            dist = d[1:]

            if min(dist) <= threshold:
                cols = ['issue_url', 'issue_title', 'body']
                dfcopy = self.rec_df.iloc[neighbors][cols].copy(deep=True)
                dfcopy['dist'] = dist
                similar_issues_df = dfcopy.query(f'dist <= {threshold}')

                print("\n**** Similar Issues (using encoder embedding) ****:\n")
                display(similar_issues_df)

    def demo_model_predictions(self,
                               n,
                               issue_df,
                               threshold=1):
        """
        Pick n random Issues and display predictions.
        Input:
        ------
        n : int
            Number of issues to display from issue_df
        issue_df : pandas DataFrame
            DataFrame that contains two columns: `body` and `issue_title`.
        threshold : float
            distance threshold for recommendation of similar issues.
        Returns:
        --------
        None
            Prints the original issue body and the model's prediction.
        """
        # Extract body and title from DF
        body_text = issue_df.body.tolist()
        title_text = issue_df.issue_title.tolist()
        url = issue_df.issue_url.tolist()

        demo_list = np.random.randint(low=1, high=len(body_text), size=n)
        for i in demo_list:
            self.print_example(i,
                               body_text=body_text[i],
                               title_text=title_text[i],
                               url=url[i],
                               threshold=threshold)

    def prepare_recommender(self, vectorized_array, original_df):
        """
        Use the annoy library to build recommender
        Parameters
        ----------
        vectorized_array : List[List[int]]
            This is the list of list of integers that represents your corpus
            that is fed into the seq2seq model for training.
        original_df : pandas.DataFrame
            This is the original dataframe that has the columns
            ['issue_url', 'issue_title', 'body']
        Returns
        -------
        annoy.AnnoyIndex  object (see https://github.com/spotify/annoy)
        """
        self.rec_df = original_df
        emb = self.encoder_model.predict(x=vectorized_array,
                                         batch_size=vectorized_array.shape[0] // 200)

        f = emb.shape[1]
        self.nn = AnnoyIndex(f)
        logging.warning('Adding embeddings')
        for i in tqdm(range(len(emb))):
            self.nn.add_item(i, emb[i])
        logging.warning('Building trees for similarity lookup.')
        self.nn.build(50)
        return self.nn

    def set_recsys_data(self, original_df):
        self.rec_df = original_df

    def set_recsys_annoyobj(self, annoyobj):
        self.nn = annoyobj


    def evaluate_model(self, holdout_bodies, holdout_titles):
        """
        Method for calculating BLEU score.

        param: holdout_bodies: List[str]
            These are issue bodies we want to summarize
        param: holdout_titles: List[str]
            These are the things we are trying to predict

        returns: bleu: float
            The BLEU score
        """
        actual, predicted = list(), list()
        assert len(holdout_bodies) == len(holdout_titles)
        num_examples = len(holdout_bodies)

        logging.warning("Generating Predictions")
        # Todo: parralize this
        for i in tqdm_notebook(range(num_examples)):
            _, yhat = self.generate_issue_title(holdout_bodies[i])

            actual.append(self.pp_title.process_text([holdout_titles[i]])[0])
            predicted.append(self.pp_title.process_text([yhat])[0])

        # Calculate BLEU score
        logging.warning("Calculating BLEU")
        bleu = corpus_bleu([[a] for a in actual], predicted)
        return bleu