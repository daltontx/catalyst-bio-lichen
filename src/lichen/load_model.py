"""Load trained model"""
import torch
import os
import json
import psutil
import warnings
import importlib.resources as pkg_resources

from .parameters import EMB_SIZE, NHEAD, FFN_HID_DIM, NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, MAX_POS_LEN, TOP_P, TEMPERATURE
from .model import Seq2SeqTransformer
from .tokenizer import ABtokenizer
from .inference import Heavy2Light


def load_model(path_to_model, device):
    # get the vocab size based on the tokenizer
    VOCAB = pkg_resources.files("lichen").joinpath('vocab.json')
    SRC_VOCAB_SIZE = len(ABtokenizer(VOCAB).vocab_to_aa)
    TGT_VOCAB_SIZE = len(ABtokenizer(VOCAB).vocab_to_aa)

    # Ignore UserWarning    
    warnings.filterwarnings("ignore", category=UserWarning)

    # Initialise the model
    seq2seqtransformer = Seq2SeqTransformer(NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMB_SIZE, NHEAD, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, FFN_HID_DIM, max_pos_len=MAX_POS_LEN)
    h2l = Heavy2Light(seq2seqtransformer, device=device, top_p=TOP_P, temperature=TEMPERATURE, vocab_path=VOCAB)
    seq2seqtransformer = seq2seqtransformer.to(device)
    load_weights(seq2seqtransformer, device, model_path=path_to_model)

    return h2l

def load_weights(model, device, model_path):
    """Load weights of model """
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))

def configure_cpus(ncpu: int):
    """Configure CPU allocation based on availability and user input."""
    available_cpus = get_available_cpus()
    if ncpu == -1:
        torch.set_num_threads(available_cpus)
        return available_cpus
    elif ncpu <= available_cpus:
        torch.set_num_threads(ncpu)
        return ncpu
    else:
        print("Number of requested CPUs exceeds available CPUs.")
        torch.set_num_threads(available_cpus)
        return available_cpus
    
def get_available_cpus():
    try:
        process = psutil.Process()
        cpu_affinity = process.cpu_affinity()
        return len(cpu_affinity)
    except Exception:  # noqa: BLE001
        if "SLURM_CPUS_PER_TASK" in os.environ:
            return int(os.environ["SLURM_CPUS_PER_TASK"])
        elif "PBS_NP" in os.environ:
            return int(os.environ["PBS_NP"])
        elif "LSB_DJOB_NUMPROC" in os.environ:
            return int(os.environ["LSB_DJOB_NUMPROC"])
        # Final fallback: detect all cores
        return os.cpu_count()
    
def configure_device(cpu: bool, ncpu: int):
    """
    Configure computation device (CPU or GPU).
    It will default to CPU if no GPU is available or if cpu=True.
    """
    if cpu or not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")

    print(f"Using device {str(device).upper()} with {ncpu} CPUs")
    return device