# coding=utf-8
# Copyright 2010, KOGPT authors
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
""" KOGPT model configuration """

from __future__ import absolute_import, division, print_function, unicode_literals

import json
import logging
import sys
import six
from io import open

from .configuration_utils import PretrainedConfig
from .modeling_kogpt import HParams

logger = logging.getLogger(__name__)

KOGPT_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    'kogpt-base-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/kogpt-base-uncased-config.json",
    'kogpt-large-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/kogpt-large-uncased-config.json",
}


class KOGPTConfig(PretrainedConfig):
    r"""
        :class:`~transformers.KOGPTConfig` is the configuration class to store the configuration of a
        `KOGPTModel`.


        Arguments:
            vocab_size_or_config_json_file: Vocabulary size of `inputs_ids` in `KOGPTModel`.
            hidden_size: Size of the encoder layers and the pooler layer.
            num_hidden_layers: Number of hidden layers in the Transformer encoder.
            num_attention_heads: Number of attention heads for each attention layer in
                the Transformer encoder.
            intermediate_size: The size of the "intermediate" (i.e., feed-forward)
                layer in the Transformer encoder.
            hidden_act: The non-linear activation function (function or string) in the
                encoder and pooler. If string, "gelu", "relu", "swish" and "gelu_new" are supported.
            hidden_dropout_prob: The dropout probabilitiy for all fully connected
                layers in the embeddings, encoder, and pooler.
            attention_probs_dropout_prob: The dropout ratio for the attention
                probabilities.
            max_position_embeddings: The maximum sequence length that this model might
                ever be used with. Typically set this to something large just in case
                (e.g., 512 or 1024 or 2048).
            type_vocab_size: The vocabulary size of the `token_type_ids` passed into
                `KOGPTModel`.
            initializer_range: The sttdev of the truncated_normal_initializer for
                initializing all weight matrices.
            layer_norm_eps: The epsilon used by LayerNorm.
    """
    pretrained_config_archive_map = KOGPT_PRETRAINED_CONFIG_ARCHIVE_MAP

    def __init__(self,
                 vocab_size_or_config_json_file=32000,
                 n_positions=1024,
                 n_ctx=1024,
                 n_embd=128,
                 n_layer=12,
                 n_head=12,
                 resid_pdrop=0.1,
                 embd_pdrop=0.1,
                 attn_pdrop=0.1,
                 layer_norm_epsilon=1e-5,
                 initializer_range=0.02,

                 num_labels=1,
                 summary_type='cls_index',
                 summary_use_proj=True,
                 summary_activation=None,
                 summary_proj_to_labels=True,
                 summary_first_dropout=0.1,
                 **kwargs):
        super(KOGPTConfig, self).__init__(**kwargs)
        self.vocab_size = vocab_size_or_config_json_file if isinstance(vocab_size_or_config_json_file, six.string_types) else -1
        self.n_ctx = n_ctx
        self.n_positions = n_positions
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head
        self.resid_pdrop = resid_pdrop
        self.embd_pdrop = embd_pdrop
        self.attn_pdrop = attn_pdrop
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_range = initializer_range

        self.num_labels = num_labels
        self.summary_type = summary_type
        self.summary_use_proj = summary_use_proj
        self.summary_activation = summary_activation
        self.summary_first_dropout = summary_first_dropout
        self.summary_proj_to_labels = summary_proj_to_labels

        self.hparams = HParams(
        n_vocab=self.vocab_size,
        n_ctx=self.n_ctx,
        n_embed=self.n_embed,
        n_hidden=self.n_hidden or self.n_embed,
        n_head=self.n_head,
        n_layer=self.n_layer,
        gradient_checkpointing=gradient_checkpointing,
        )

        if isinstance(vocab_size_or_config_json_file, six.string_types):
            with open(vocab_size_or_config_json_file, "r", encoding="utf-8") as reader:
                json_config = json.loads(reader.read())
            for key, value in json_config.items():
                self.__dict__[key] = value
        elif not isinstance(vocab_size_or_config_json_file, int):
            raise ValueError(
                "First argument must be either a vocabulary size (int)"
                "or the path to a pretrained model config file (str)"
            )

    @property
    def max_position_embeddings(self):
        return self.n_positions

    @property
    def hidden_size(self):
        return self.n_embd

    @property
    def num_attention_heads(self):
        return self.n_head

    @property
    def num_hidden_layers(self):
        return self.n_layer
