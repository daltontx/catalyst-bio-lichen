"""Inference loop"""

import torch
import torch.nn.functional as F
import random

from Bio.Align import PairwiseAligner
from Bio.Align.substitution_matrices import load
import pandas as pd
from operator import itemgetter

from .tokenizer import ABtokenizer
from .utils import MAP_FR1_FR3, MAP_CDRS_FR1_IMGT, MAP_CDRS_FR1_KABAT, MAP_TYPE_SEED


class Heavy2Light:
    def __init__(self, model, device, top_p=0.9, temperature=1.0, vocab_path="./vocab.json"):
        self.model = model
        self.device = device
        self.tokenizer = ABtokenizer(vocab_path=vocab_path)
        self.top_p = top_p
        self.temperature = temperature

        # FR1 start based on CDR
        # IMGT
        self.map_CDRs_startFR1_imgt = MAP_CDRS_FR1_IMGT
        self.map_CDRs_startFR1_imgt['CDR1CDR2'] = self.map_CDRs_startFR1_imgt['CDR1'] + self.map_CDRs_startFR1_imgt['CDR2']
        # Kabat
        self.map_CDRs_startFR1_kabat = MAP_CDRS_FR1_KABAT
        self.map_CDRs_startFR1_kabat['CDR1CDR2'] = self.map_CDRs_startFR1_kabat['CDR1'] + self.map_CDRs_startFR1_kabat['CDR2']

        # Setup aligner
        self.aligner = PairwiseAligner()
        blosum62 = load("BLOSUM62")
        self.aligner.substitution_matrix = blosum62
        self.aligner.end_open_gap_score = -15 # Strong penalty for opening gaps at ends
        self.aligner.end_extend_gap_score = -5 # Strong penalty for extending gaps at ends

    def _greedy_decode(self, src_, src_mask_, max_len, start_symbol, end_symbol, light_seed, light_cdr, light_cdr_scheme):
        src_ = [src.to(self.device) for src in src_]
        src_mask_ = [src_mask.to(self.device) for src_mask in src_mask_]
        memory_ = [self.model.encode(src_[j], src_mask_[j]) for j in range(len(src_))]

        # Initialise output sequence
        ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(self.device)

        i=0
        cdr1_grafted = False
        conserved_W41_generated = False
        while i < (max_len-1):

            # Seed with the germline information if given
            if light_seed and i < len(light_seed):
                next_word = self.tokenizer.vocab_to_token[light_seed[i]]
                ys = torch.cat([ys, torch.ones(1, 1).type_as(src_[0].data).fill_(next_word)], dim=0)
            
            # Seed with start FR1 if CDR1 and or CDR2 given bit no seed given
            elif i == 0 and light_cdr and (light_cdr[0] or light_cdr[1]) and not light_seed:
                # Get the CDR information provided
                if light_cdr[0] and light_cdr[1]: # CDR1 and CDR2 given
                    cdr_provided_1 = light_cdr[0]
                    cdr_provided_2 = light_cdr[1]
                    max_score_1 = self.aligner.score(cdr_provided_1, cdr_provided_1)
                    max_score_2 = self.aligner.score(cdr_provided_2, cdr_provided_2)
                elif light_cdr[0] and not light_cdr[1]: # only CDR1 given
                    cdr_provided_1 = light_cdr[0]
                    cdr_provided_2 = None
                else: # only CDR2 given
                    cdr_provided_1 = None
                    cdr_provided_2 = light_cdr[1]
                
                # Load the lookup tables
                if light_cdr_scheme=='Kabat':
                    lookup_1 = self.map_CDRs_startFR1_kabat['CDR1'].to_list()
                    lookup_2 = self.map_CDRs_startFR1_kabat['CDR2'].to_list()
                else: # IMGT
                    lookup_1 = self.map_CDRs_startFR1_imgt['CDR1'].to_list() 
                    lookup_2 = self.map_CDRs_startFR1_imgt['CDR2'].to_list()

                # Find the closest FR1 start
                closest_cdrs_scores = {}
                for cdr_lookup_pos in range(len(lookup_1)):
                    if not cdr_provided_1:
                        alignment_score = self.aligner.score(lookup_2[cdr_lookup_pos], cdr_provided_2)
                    elif not cdr_provided_2:
                        alignment_score = self.aligner.score(lookup_1[cdr_lookup_pos], cdr_provided_1)
                    else:
                        # If both given we need to make the sum balanced
                        alignment_score = self.aligner.score(lookup_1[cdr_lookup_pos], cdr_provided_1)/max_score_1 + self.aligner.score(lookup_2[cdr_lookup_pos], cdr_provided_2)/max_score_2

                    closest_cdrs_scores[f'{lookup_1[cdr_lookup_pos] + lookup_2[cdr_lookup_pos]}'] = alignment_score

                # Select FR with highest score and check the type of that sequence (lambda/kappa)
                top_value=1
                closest_cdrs_top = dict(sorted(closest_cdrs_scores.items(), key = itemgetter(1), reverse = True)[:top_value])
                closest_cdr = list(closest_cdrs_top.keys())[0]
                if light_cdr_scheme == 'Kabat':
                    start_FR1 = random.sample(self.map_CDRs_startFR1_kabat[self.map_CDRs_startFR1_kabat['CDR1CDR2'] == closest_cdr]['startFR1'].to_list(),1)[0]
                else:
                    start_FR1 = random.sample(self.map_CDRs_startFR1_imgt[self.map_CDRs_startFR1_imgt['CDR1CDR2'] == closest_cdr]['startFR1'].to_list(),1)[0]

                # Map FR1 to kappa/lambda and then select start from that type
                for l_type in list(MAP_TYPE_SEED.keys()):
                    if start_FR1 in MAP_TYPE_SEED[l_type]:
                        start_FR1 = random.choices(MAP_TYPE_SEED[l_type], k = 1)[0]

                # Seed the restul with this start of FR1
                for k in range(len(start_FR1)):
                    next_word = self.tokenizer.vocab_to_token[start_FR1[k]]
                    ys = torch.cat([ys, torch.ones(1, 1).type_as(src_[0].data).fill_(next_word)], dim=0)
                # update counter i
                i = i + len(start_FR1)

            # Grafting CDRs (if provided)
            # Check position CDR1 reached
            elif light_cdr and self._position_cdr1(light_cdr, ys, light_cdr_scheme):
                cdr1_seq = light_cdr[0]
                if light_cdr_scheme == 'Kabat':
                    # add the conserved W immediately
                    cdr1_seq += 'W'
                # iteratively add the residues to the result
                for j in range(len(cdr1_seq)):
                    next_word = self.tokenizer.vocab_to_token[cdr1_seq[j]]
                    ys = torch.cat([ys, torch.ones(1, 1).type_as(src_[0].data).fill_(next_word)], dim=0)
                # update counter i
                i = i + len(cdr1_seq)
    
                if light_cdr_scheme == 'IMGT':
                    # Keep track of CDR1 grafted for further positions
                    cdr1_grafted = len(ys)-1 # -1 needed because of start token

            # Check position CDR2 reached
            elif light_cdr and self._position_cdr2(light_cdr, ys, light_cdr_scheme):
                cdr2_seq = light_cdr[1]
                # iteratively add the residues to the result
                for j in range(len(cdr2_seq)):
                    next_word = self.tokenizer.vocab_to_token[cdr2_seq[j]]
                    ys = torch.cat([ys,torch.ones(1, 1).type_as(src_[0].data).fill_(next_word)], dim=0)
                # update counter i
                i = i + len(cdr2_seq)

            # Check position CDR3 reached
            elif light_cdr and self._position_cdr3(light_cdr, ys):
                cdr3_seq = light_cdr[2]
                # iteratively add the residues to the result
                for j in range(len(cdr3_seq)):
                    next_word = self.tokenizer.vocab_to_token[cdr3_seq[j]]
                    ys = torch.cat([ys, torch.ones(1, 1).type_as(src_[0].data).fill_(next_word)], dim=0)
                # FR4 always starts with F so add manual to identify end of CDR3
                next_word = self.tokenizer.vocab_to_token['F']
                ys = torch.cat([ys,torch.ones(1, 1).type_as(src_[0].data).fill_(next_word)], dim=0)
                # update counter i
                i = i + len(cdr3_seq) + 1 #last one for the forced F

            # Model should create next token itself
            else:
                probs = []
                for memory in memory_:
                    memory = memory.to(self.device)
                    tgt_mask = (self._generate_square_subsequent_mask(ys.size(0)).type(torch.bool)).to(self.device)

                    # Get probabilities of tokens and sample next word
                    out = self.model.decode(ys, memory, tgt_mask)
                    out = out.transpose(0, 1)
                    probs.append(self.model.generator(out[:, -1]))

                # Average probabilities in case of a bispecific
                if len(probs)>1:
                    prob = (probs[0] + probs[1]) / 2
                else:
                    prob = probs[0]

                next_word = self._top_p_sampling(prob) # for top-p sampling of next word
                next_word = next_word.item()

                # If CDR1 grafted and IMGT than need to carefully consider position of conserved W
                if light_cdr_scheme == 'IMGT':
                    # W is generated, check its position
                    if next_word == self.tokenizer.vocab_to_token['W']:

                        # W too early after grafted CDR1 
                        if cdr1_grafted and i in [cdr1_grafted + 1, cdr1_grafted + 2]:
                            # This should not be a W, so sample again witholding W
                            prob[:, self.tokenizer.vocab_to_token['W']] = float('-inf') # set index of W to infinity
                            next_word = self._top_p_sampling(prob)
                            next_word = next_word.item()
                            ys = torch.cat([ys,torch.ones(1, 1).type_as(src_[0].data).fill_(next_word)], dim=0)

                            # Place the W at the correct position
                            if i == cdr1_grafted + 1:
                                # Need to sample the second residue between CDR1 and conserved W
                                memory = memory.to(self.device)
                                tgt_mask = (self._generate_square_subsequent_mask(ys.size(0)).type(torch.bool)).to(self.device)
                                out = self.model.decode(ys, memory, tgt_mask)
                                out = out.transpose(0, 1)
                                prob = self.model.generator(out[:, -1])
                                # Again this should not be a W
                                prob[:, self.tokenizer.vocab_to_token['W']] = float('-inf') # set index of W to infinity
                                next_word = self._top_p_sampling(prob)
                                next_word = next_word.item()
                                ys = torch.cat([ys,torch.ones(1, 1).type_as(src_[0].data).fill_(next_word)], dim=0)
                                i += 2
                            else:
                                i += 1
                            # Now force the conserved W
                            next_word = self.tokenizer.vocab_to_token['W']
                            cdr1_grafted = False
                            conserved_W41_generated = True

                        # W too later after grafted CDR1 (longer CDR1 than requested)
                        elif cdr1_grafted and i > (cdr1_grafted + 3):
                            # Remove the extra CDR residues
                            ys = torch.cat((ys[:cdr1_grafted+1], ys[len(ys)-2:]), dim=0) 
                            # +1 because start token, -2 because 2 residues between CDR1 end and W
                            cdr1_grafted = False # W correct position now
                            conserved_W41_generated = True

                        elif cdr1_grafted and i == (cdr1_grafted + 3):
                            # Correct position
                            cdr1_grafted = False # W correct position now
                            conserved_W41_generated = True

                    # Force W 3rd position after CDR1 if after 10 still not given
                    if cdr1_grafted and i > cdr1_grafted + 10 and not conserved_W41_generated:
                        # Keep the first two after the CDR and add a W
                        ys = ys[:cdr1_grafted+3] # 1 for start token and 2 for start FR2
                        next_word = self.tokenizer.vocab_to_token['W']
                        conserved_W41_generated = True

                # Add this checked W to the results
                ys = torch.cat([ys,torch.ones(1, 1).type_as(src_[0].data).fill_(next_word)], dim=0)

            # End of sequence reached
            if next_word == end_symbol or i == max_len-1-1:
                break

            # Update i
            i += 1

        return ys
    
    def _position_cdr1(self, light_cdr_list, gen_light, light_cdr_scheme):#
        """Should graft CDR1 if given and if correct position.
        Correct position is 4th after concerved cysteine at position 23"""
        if not light_cdr_list[0]:
            return False
        elif light_cdr_scheme=='IMGT' and len(gen_light) >24 and len(gen_light)<31 and gen_light[-4].item()==self.tokenizer.vocab_to_token['C']:
            return True
        elif light_cdr_scheme=='Kabat' and len(gen_light) >21 and len(gen_light)<28 and gen_light[-1].item()==self.tokenizer.vocab_to_token['C']:
            return True
        else:
            return False

    def _position_cdr2(self, light_cdr_list, gen_light, light_cdr_scheme):
        """Should graft CDR2 if given and if correct position.
        Correct position is 15th after concerved Tryptophan (W) at position 41.
        If CDR1 grafted as well base position on CDR1.
        We now don't allow length flexibility in FR2 region"""
        if not light_cdr_list[1]:
            return False
        elif light_cdr_list[0]:
            # Position based on the CDR1
            if light_cdr_scheme=='IMGT' and len(gen_light)>41 and len(gen_light)<60 and gen_light[-15].item()==self.tokenizer.vocab_to_token['W'] and ''.join(self.tokenizer.decode(gen_light.flatten())).replace('<', '').replace('>', '')[-19:-17]==light_cdr_list[0][-2:]:
                return True
            elif light_cdr_scheme=='Kabat' and len(gen_light)>41 and len(gen_light)<60 and gen_light[-15].item()==self.tokenizer.vocab_to_token['W'] and ''.join(self.tokenizer.decode(gen_light.flatten())).replace('<', '').replace('>', '')[-17:-15]==light_cdr_list[0][-2:]:
                return True
            else:
                return False
        elif len(gen_light)>41 and len(gen_light)<60 and gen_light[-15].item()==self.tokenizer.vocab_to_token['W']:
            # If no CDR1 grafted same rule applies for IMGT and Kabat numbering
            return True
        else:
            return False
        
    def _position_cdr3(self, light_cdr_list, gen_light):
        """Should graft CDR3 if given and if correct position.
        Correct position is 1st after concerved cysteine at position 104"""
        if not light_cdr_list[2]:
            return False
        elif len(gen_light)>82 and len(gen_light)<108 and gen_light[-1].item()==self.tokenizer.vocab_to_token['C']:
            return True
        else:
            return False

    def _top_p_sampling(self, logits):
        """Sample from top p based on cumulative probabilities of classes"""
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(torch.nn.functional.softmax(sorted_logits / self.temperature, dim=-1), dim=-1)
        
        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > self.top_p
        sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
        sorted_indices_to_remove[:, 0] = 0
        
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        # Sort indices_to_remove in ascending order
        sorted_indices_to_remove = torch.sort(indices_to_remove).values
        # Apply the mask to sorted_logits
        sorted_logits[:, sorted_indices_to_remove] = float('-inf')
        sampled_token = torch.multinomial(torch.nn.functional.softmax(sorted_logits / self.temperature, dim=-1), 1)
        return sampled_token
    
    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones((sz, sz), device=self.device)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
    
    def generate_light(self, heavy_seq, light_seed=None, light_cdr=None, light_cdr_scheme='IMGT'):
        # Make sure model in evaluation modus
        self.model.eval()

        # encode the heavy sequence
        src = []
        src_mask = []
        for j in range(len(heavy_seq)):
            src.append(self.tokenizer.encode(heavy_seq[j]).view(-1, 1))
            num_tokens = src[j].shape[0]
            src_mask.append((torch.zeros(num_tokens, num_tokens)).type(torch.bool))

        # Set maximum length of result
        if num_tokens < 5:
            max_length_light = 200
        else:
            max_length_light = num_tokens + 5

        tgt_tokens = self._greedy_decode(
            src,
            src_mask,
            max_len=max_length_light,
            start_symbol=self.tokenizer.start_token,
            end_symbol=self.tokenizer.end_token,
            light_seed=light_seed,
            light_cdr=light_cdr,
            light_cdr_scheme=light_cdr_scheme
            )
        
        # format result
        tgt_tokens = tgt_tokens.flatten()

        return ''.join(self.tokenizer.decode(tgt_tokens)).replace("<", "").replace(">", "")
    

    def likelihood_light(self, heavy_seq, light_seq):
        # Make sure model in evaluation modus
        self.model.eval()

        # encode the heavy sequence
        src = self.tokenizer.encode(heavy_seq).view(-1, 1)
        num_tokens = src.shape[0]
        src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)

        # encode the light sequence
        tgt = self.tokenizer.encode(light_seq).view(-1, 1)

        # Get probability
        log_probability = self._decode_likelihood(src, 
                                                  src_mask, 
                                                  tgt,
                                                  start_symbol=self.tokenizer.start_token,
                                                  end_symbol=self.tokenizer.end_token,)

        # Handle the output
        log_probability = log_probability.cpu().item()

        return log_probability

    def _decode_likelihood(self, src, src_mask, tgt, start_symbol, end_symbol):
        src = src.to(self.device)
        src_mask = src_mask.to(self.device)
        tgt = tgt.to(self.device)

        memory = self.model.encode(src, src_mask)

        # initialise output
        ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(self.device)

        total_log_prob = 0
        i=0
        while i < tgt.shape[0]+1:
            memory = memory.to(self.device)
            tgt_mask = (self._generate_square_subsequent_mask(ys.size(0)).type(torch.bool)).to(self.device)
                
            # Get probabilities next token
            out = self.model.decode(ys, memory, tgt_mask)
            out = out.transpose(0, 1)
            prob = self.model.generator(out[:, -1])

            token_index = tgt[i+1] # Need +1 as considering next token
            if token_index != end_symbol: # don't include start and end symbol
                probabilities = F.softmax(prob, dim=-1) # convert logits to probabilities
                token_prob = probabilities[:,token_index] # get probability of specific next token
                token_log_prob = torch.log(token_prob.detach() + 1e-20) # add small value to prevent log of zero
                total_log_prob += token_log_prob
            next_word = token_index.item()
            ys = torch.cat([ys,torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)

            if next_word == end_symbol:
                break

            # update i in while loop
            i += 1

        # Check given and generated same
        assert torch.equal(tgt, ys)

        return total_log_prob
        

        




       




        











        


    

    

                




            


                


            

