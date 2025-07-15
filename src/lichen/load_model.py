"""Load trained model"""
import torch
import os

from .parameters import EMB_SIZE, NHEAD, FFN_HID_DIM, NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, MAX_POS_LEN, TOP_P, TEMPERATURE
from .model import Seq2SeqTransformer
from .tokenizer import ABtokenizer
from .inference import Heavy2Light


def load_model(path_to_model, device):
    # get the vocab size based on the tokenizer
    current_dir = os.getcwd()
    VOCAB = os.path.join(current_dir, "src", "lichen", "vocab.json")
    SRC_VOCAB_SIZE = len(ABtokenizer(VOCAB).vocab_to_aa)
    TGT_VOCAB_SIZE = len(ABtokenizer(VOCAB).vocab_to_aa)

    # Initialise the model
    seq2seqtransformer = Seq2SeqTransformer(NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMB_SIZE, NHEAD, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, FFN_HID_DIM, max_pos_len=MAX_POS_LEN)
    h2l = Heavy2Light(seq2seqtransformer, device=device, top_p=TOP_P, temperature=TEMPERATURE, vocab_path=VOCAB)
    seq2seqtransformer = seq2seqtransformer.to(device)
    load_weights(seq2seqtransformer, device, model_path=path_to_model)

    return h2l

def load_weights(model, device, model_path):
    """Load weights of model """
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
