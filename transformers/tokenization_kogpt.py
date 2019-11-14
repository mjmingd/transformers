# coding=utf-8
# Copyright 2018 KOGPT Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Tokenization class for model KOGPT."""

from __future__ import absolute_import, division, print_function, unicode_literals

import collections
import logging
import os
import unicodedata
from io import open
import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import sentencepiece as spm
import tqdm

from .common import UNK, END_OF_TEXT, END_OF_LINE



from .tokenization_utils import PreTrainedTokenizer

logger = logging.getLogger(__name__)

####################################################
# In this template, replace all the KOGPT (various casings) with your model name
####################################################

####################################################
# Mapping from the keyword arguments names of Tokenizer `__init__`
# to file names for serializing Tokenizer instances
####################################################
VOCAB_FILES_NAMES = {'vocab_file': 'kogpt_vocab.vocab',
                     'sp_model_file' : 'kogpt_sp_model.model' }

####################################################
# Mapping from the keyword arguments names of Tokenizer `__init__`
# to pretrained vocabulary URL for all the model shortcut names.
####################################################
PRETRAINED_VOCAB_FILES_MAP = {
    'vocab_file':
    {
        'kogpt-base-uncased': "https://drive.google.com/open?id=16bTFcqfzEWKMfkqCIF1hBtv0sUF-7NR8" #sp-model.vocab
    }
    'sp_model_file' :
    {
        'kogpt-base-uncased' : "https://drive.google.com/open?id=1Y98QfX6JdTyQN57I-bhh4MtfKIashUKF" #sp-model.model
    }
}

####################################################
# Mapping from model shortcut names to max length of inputs
####################################################
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    'kogpt-base-uncased': 512
}

####################################################
# Mapping from model shortcut names to a dictionary of additional
# keyword arguments for Tokenizer `__init__`.
# To be used for checkpoint specific configurations.
####################################################
PRETRAINED_INIT_CONFIGURATION = {
    'kogpt-base-uncased': {'do_lower_case': True}
}


def load_vocab(vocab_file):
    """Loads a vocabulary file into a dictionary."""
    vocab = collections.OrderedDict()
    with open(vocab_file, "r", encoding="utf-8") as reader:
        tokens = reader.readlines()
        for token in tokens:
            token, index = token.rstrip('\n').split('\t')
            vocab[token] = index
    return vocab


class KOGPTTokenizer(PreTrainedTokenizer):
    r"""
    Constructs a KOGPTTokenizer.
    :class:`~transformers.KOGPTTokenizer` runs end-to-end tokenization: sentencepiece

    Args:
        vocab_file: Path to a one-wordpiece-per-line vocabulary file
        do_lower_case: Whether to lower case the input. Only has an effect when do_wordpiece_only=False
    """

    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    pretrained_init_configuration = PRETRAINED_INIT_CONFIGURATION
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES

    def __init__(self, vocab_file,
                 unk_token="<unk>", bos_token="<|endoftext|>", eos_token="<|endoftext|>", **kwargs):

        """
        KOGPT Tokenizer : sentencepiece
        """
        super(KOGPTTokenizer, self).__init__(unk_token=unk_token, bos_token=bos_token, eos_token=eos_token, **kwargs)
        self.max_len_single_sentence = self.max_len
        self.max_len_sentences_pair = self.max_len

        self.sp_model = spm.SentencePieceProcessor()
        assert self.sp_model.load(self.sp_model_file)

        self.errors = errors  # how to handle errors in decoding
        self.cache = {}


    @property
    def vocab_size(self):
        return self.sp_model.GetPieceSize()


    def _tokenize(self, text:str) -> List[str]:
        """ Take as input a string and return a list of tokens for sub-words
            Args:
                - add_prefix_space (boolean, default False):
                    Begin the sentence with at least one space toto get invariance to word order in GPT-2 (and RoBERTa) tokenizers.
        """

        return self.sp_model.EncodeAsPieces(text)

    def _convert_token_to_id(self, token:str) -> int:
        """ Converts a token (str/unicode) in an id using the vocab. """
        return self.sp_model.PieceToId(token)

    def _convert_id_to_token(self, index:int) -> str:
        """Converts an index (integer) in a token (string/unicode) using the vocab."""
        return self.sp_model.IdToPiece(int(index))

    def convert_tokens_to_string(self, tokens):
        """ Converts a sequence of tokens (string) in a single string. """
        return sp.decode_pieces(tokens)