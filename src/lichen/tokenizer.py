import json
import torch

class ABtokenizer():
    """ Tokenizer for proteins. Both aa to token and token to aa.
    """
    def __init__(self, vocab_path):
       self.set_vocabs(vocab_path)
       self.pad_token = self.vocab_to_token['-']
       self.start_token = self.vocab_to_token['<']
       self.end_token = self.vocab_to_token['>']

    def __call__(self, sequenceList, encode=True, device='cpu'):
        if encode:
            data = [self.encode(seq, device=device) for seq in sequenceList]
            return data
        else: 
           return [self.decode(token) for token in sequenceList]

    def set_vocabs(self, vocab_path):
        with open(vocab_path, encoding="utf-8") as vocab_handle:
            self.vocab_to_token=json.load(vocab_handle)
        
        self.vocab_to_aa = {v: k for k, v in self.vocab_to_token.items()}
 
    def encode(self, sequence, device='cpu'):
       encoded = [self.vocab_to_token["<"]]+[self.vocab_to_token[resn] for resn in sequence]+[self.vocab_to_token[">"]]
       return torch.tensor(encoded, dtype=torch.long, device=device)

    def decode(self, seqtokens):
        if torch.is_tensor(seqtokens): seqtokens = seqtokens.cpu().numpy()
        return ''.join([self.vocab_to_aa[token] for token in seqtokens])