""" Usage:
    <file-name> --nc=NOUN_COMPOUND --n=NUM_OF_SLOTS --bw=BEAM_WIDTH --out=OUT_FILE [--debug]
"""
# External imports
import logging
import pdb
from pprint import pprint
from pprint import pformat
from docopt import docopt
from collections import defaultdict
from operator import itemgetter
from tqdm import tqdm
import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
from torch.nn import functional
from typing import Tuple, List
# Local imports

#=-----
class BertMaskedLM:
    """
    Container for BERT.
    """
    def __init__(self, beam_width):
        """
        Init structures needed for BERT.
        """
        self.beam_width = beam_width
        self.counter = 0
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertForMaskedLM.from_pretrained('bert-base-uncased')
        self.model.eval()

    def convert_sent_to_tokens(self, sent_ids):
        """
        Convert ids to string.
        """
        return self.tokenizer.convert_ids_to_tokens([t.item() for t in sent_ids])

    def predict(self, masked_input: str) -> Tuple[str, float]:
        """
        Return prediction for an input with masked input word(s).
        Recursive and assumes properly tokenized input.
        """
          tokenized_text = self.tokenizer.tokenize(masked_input)

        # Convert token to vocabulary indices
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)

        # Convert inputs to PyTorch tensors
        tokens_tensor = torch.tensor([indexed_tokens])
        segments_tensors = torch.tensor([[0] * len(indexed_tokens)])

        # Get indexes of masked words
        masked_indexes = [word_ind for word_ind, word in enumerate(tokenized_text)
                          if word == MASK_TOKEN]

        probs, preds = zip(*self._predict(masked_indexes, tokens_tensor, segments_tensors))
        preds_str = [" ".join(self.tokenizer.convert_ids_to_tokens([t.item() for t in indices[0]]))
                     for indices in preds]
        probs_and_preds = zip(probs, preds_str)
        return list(probs_and_preds)

    def _predict(self, masked_indexes: List[int], tokens_tensor, segments_tensor) -> Tuple[str, float]:
        """
        Return prediction for an input with masked input word(s).
        Recursive and assumes properly tokenized input.
        """
        if not masked_indexes:
            # Nothing to predict
            return [(1, tokens_tensor)]

        # Predict first masked token
        masked_index = masked_indexes[0]

        with torch.no_grad():
            self.counter += 1
            if (self.counter % 10) == 0:
                logging.info(self.counter)
            predictions = functional.normalize(self.model(tokens_tensor, segments_tensor),
                                               dim = 2, p = 1)
            cur_row = predictions[0, masked_index]

            # Get K most probable instances
            probs, indices = cur_row.topk(self.beam_width)
            probs = [t.item() for t in probs]
            predicted_tokens = [t.item() for t in indices]

        predicted_ls = []
        for word_prob, pred_word in zip(probs, predicted_tokens):
            # Replace with the current option
            cur_opt = tokens_tensor.clone()
            cur_opt[0, masked_index] = pred_word

            # Predict the rest
            pred_rest = self._predict(masked_indexes[1:], cur_opt, segments_tensor)
            for (rest_prob, rest_str) in pred_rest:
                predicted_ls.append((word_prob * rest_prob, rest_str))

        top_predictions = sorted(predicted_ls, key = itemgetter(0), reverse = True)
        top_k_predictions = top_predictions[: self.beam_width]

        return top_k_predictions

MASK_TOKEN = "[MASK]"

if __name__ == "__main__":
    # Parse command line arguments
    args = docopt(__doc__)
    debug = args["--debug"]
    if debug:
        logging.basicConfig(level = logging.DEBUG)
    else:
        logging.basicConfig(level = logging.INFO)

    logging.info(args)
    nc = args["--nc"].split()
    num_of_slots = int(args["--n"])
    beam_width = int(args["--bw"])
    out_fn = args["--out"]


    # Parse input
    assert len(nc) == 2
    n1, n2 = nc
    mask = " ".join([MASK_TOKEN] * num_of_slots)
w
    # Tokenized input
    text = f"[CLS] {n1} {n2} is a {n2} that {mask} {n1} . [SEP]"

    bert = BertMaskedLM(beam_width = beam_width)
    ls = bert.predict(text)
    with open(out_fn, "w", encoding = "utf8") as fout:
        fout.write(pformat(ls))

    logging.info(pformat(ls[: 10]))


    logging.info("DONE")
