""" Usage:
    <file-name> --in=INPUT_FILE --out=OUT_FILE [--debug]
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
from pytorch_pretrained_bert import BertTokenizer, BertModel
from torch.nn import functional
from typing import Tuple, List
# Local imports

#=-----
class BertLM:
    """
    Container for BERT pretrained model.
    """
    def __init__(self):
        """
        Init structures needed for BERT.
        """
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased')
        self.model.eval()

    def embed(self, input_sent: str):
        """
        Return prediction for an input with masked input word(s).
        Recursive and assumes properly tokenized input.
        """
        tokenized_text = self.tokenizer.tokenize(input_sent)

        # Convert token to vocabulary indices
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)

        # Convert inputs to PyTorch tensors
        tokens_tensor = torch.tensor([indexed_tokens])

        # embed
        with torch.no_grad():
            encoded_layers, _ = self.model(tokens_tensor, output_all_encoded_layers = False)

        return tokenized_text, encoded_layers

if __name__ == "__main__":
    # Parse command line arguments
    args = docopt(__doc__)
    debug = args["--debug"]
    if debug:
        logging.basicConfig(level = logging.DEBUG)
    else:
        logging.basicConfig(level = logging.INFO)

    logging.info(args)
    inp_fn = args["--in"]
    out_fn = args["--out"]

    bert_encoder = BertLM()
    sent = "Jim Henson was a puppeteer"
    tokenized_text, emb = bert_encoder.embed(sent)
    logging.info(tokenized_text)

    logging.info("DONE")
