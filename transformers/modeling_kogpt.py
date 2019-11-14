# coding=utf-8
# Copyright 2018 KOGPT Authors
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
""" PyTorch KOGPT model. """

####################################################
# In this template, replace all the KOGPT (various casings) with your model name
####################################################

from __future__ import absolute_import, division, print_function, unicode_literals

import collections
import json
import logging
import math
import os
import sys
from io import open

import attr
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn import CrossEntropyLoss
import torch.utils.checkpoint
from torch.nn.parameter import Parameter

from .modeling_utils import PreTrainedModel, prune_conv1d_layer, SequenceSummary
from .modeling_utils import KO_Conv1D as Conv1D
from .configuration_kogpt import KOGPTConfig
from .file_utils import add_start_docstrings





logger = logging.getLogger(__name__)

####################################################
# This dict contrains shortcut names and associated url
# for the pretrained weights provided with the models
####################################################
KOGPT_PRETRAINED_MODEL_ARCHIVE_MAP = {
    'kogpt-base-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/kogpt-base-uncased-pytorch_model.bin"
}

####################################################
# This is a conversion method from TF 1.0 to PyTorch
# More details: https://medium.com/huggingface/from-tensorflow-to-pytorch-265f40ef2a28
####################################################
def load_tf_weights_in_kogpt(model, config, tf_checkpoint_path):
    """ Load tf checkpoints in a pytorch model.
    """
    try:
        import re
        import numpy as np
        import tensorflow as tf
    except ImportError:
        logger.error("Loading a TensorFlow model in PyTorch, requires TensorFlow to be installed. Please see "
            "https://www.tensorflow.org/install/ for installation instructions.")
        raise
    tf_path = os.path.abspath(tf_checkpoint_path)
    logger.info("Converting TensorFlow checkpoint from {}".format(tf_path))
    # Load weights from TF model
    init_vars = tf.train.list_variables(tf_path)
    names = []
    arrays = []
    for name, shape in init_vars:
        logger.info("Loading TF weight {} with shape {}".format(name, shape))
        array = tf.train.load_variable(tf_path, name)
        names.append(name)
        arrays.append(array)

    for name, array in zip(names, arrays):
        name = name.split('/')
        # adam_v and adam_m are variables used in AdamWeightDecayOptimizer to calculated m and v
        # which are not required for using pretrained model
        if any(n in ["adam_v", "adam_m", "global_step"] for n in name):
            logger.info("Skipping {}".format("/".join(name)))
            continue
        pointer = model
        for m_name in name:
            if re.fullmatch(r'[A-Za-z]+_\d+', m_name):
                l = re.split(r'_(\d+)', m_name)
            else:
                l = [m_name]
            if l[0] == 'kernel' or l[0] == 'gamma':
                pointer = getattr(pointer, 'weight')
            elif l[0] == 'output_bias' or l[0] == 'beta':
                pointer = getattr(pointer, 'bias')
            elif l[0] == 'output_weights':
                pointer = getattr(pointer, 'weight')
            elif l[0] == 'squad':
                pointer = getattr(pointer, 'classifier')
            else:
                try:
                    pointer = getattr(pointer, l[0])
                except AttributeError:
                    logger.info("Skipping {}".format("/".join(name)))
                    continue
            if len(l) >= 2:
                num = int(l[1])
                pointer = pointer[num]
        if m_name[-11:] == '_embeddings':
            pointer = getattr(pointer, 'weight')
        elif m_name == 'kernel':
            array = np.transpose(array)
        try:
            assert pointer.shape == array.shape
        except AssertionError as e:
            e.args += (pointer.shape, array.shape)
            raise
        logger.info("Initialize PyTorch weight {}".format(name))
        pointer.data = torch.from_numpy(array)
    return model


####################################################
# PyTorch Models are constructed by sub-classing
# - torch.nn.Module for the layers and
# - PreTrainedModel for the models (itself a sub-class of torch.nn.Module)
####################################################

####################################################
# Here is an example of typical layer in a PyTorch model of the library
# The classes are usually identical to the TF 2.0 ones without the 'TF' prefix.
#
# See the conversion methods in modeling_tf_pytorch_utils.py for more details
####################################################




@attr.s(auto_attribs=True, frozen=True)
class HParams:
    n_vocab: int
    n_ctx: int
    n_embed: int
    n_hidden: int
    n_head: int
    n_layer: int
    gradient_checkpointing: bool

class Block(nn.Module):
    def __init__(self, hparams: HParams):
        super().__init__()
        self.ln_1 = Norm(hparams.n_hidden)
        self.ln_2 = Norm(hparams.n_hidden)
        self.mlp = MLP(hparams.n_hidden, hparams.n_hidden * 4)
        self.attn = Attention(hparams)

    def forward(self, x, past):
        a, present = self.attn(self.ln_1(x), past=past)
        x = x + a
        m = self.mlp(self.ln_2(x))
        x = x + m
        return x, present


class Norm(nn.Module):
    """ Normalize to mean = 0, std = 1, then do a diagonal affine transform.
    """
    def __init__(self, n_features, *, dim=-1, epsilon=1e-5):
        super().__init__()
        self.n_features = n_features
        self.dim = dim
        self.epsilon = epsilon
        self.g = nn.Parameter(torch.ones(n_features))
        self.b = nn.Parameter(torch.zeros(n_features))

    def forward(self, x):
        assert x.shape[-1] == self.n_features
        u = torch.mean(x, dim=self.dim, keepdim=True)
        xmu = x - u
        s = torch.mean(xmu * xmu, dim=self.dim, keepdim=True)
        return xmu * torch.rsqrt(s + self.epsilon) * self.g + self.b



class MLP(nn.Module):
    def __init__(self, n_features, n_hidden):
        super().__init__()
        self.c_fc = Conv1D(n_features, n_hidden)
        self.c_proj = Conv1D(n_hidden, n_features)

    def forward(self, x):
        x = gelu(self.c_fc(x))
        x = self.c_proj(x)
        return x


class Attention(nn.Module):
    def __init__(self, hparams: HParams):
        super().__init__()
        assert hparams.n_hidden % hparams.n_head == 0
        self.hparams = hparams
        self.c_attn = Conv1D(hparams.n_hidden, hparams.n_hidden * 3)
        self.c_proj = Conv1D(hparams.n_hidden, hparams.n_hidden)

    def forward(self, x, past):
        assert len(x.shape) == 3  # [batch, sequence, features]
        assert x.shape[-1] == self.hparams.n_hidden
        if past is not None:
            # Should be [batch, 2, heads, sequence, features], where 2 is [k, v]
            assert len(past.shape) == 5
            assert past.shape[-1] == self.hparams.n_hidden
        c = self.c_attn(x)
        q, k, v = map(self.split_heads, torch.split(c, x.shape[-1], dim=2))
        present = torch.stack([k, v], dim=1)
        if past is not None:
            pk, pv = past[:, 0], past[:, 1]
            k = torch.cat([pk, k], dim=-2)
            v = torch.cat([pv, v], dim=-2)
        a = self.multihead_attn(q, k, v)
        a = self.merge_heads(a)
        a = self.c_proj(a)
        return a, present

    def split_heads(self, x):
        """ From [batch, sequence, features] to
        [batch, heads, sequence, features].
        """
        return self.split_states(x, self.hparams.n_head).permute(0, 2, 1, 3)

    @staticmethod
    def split_states(x, n):
        """ Reshape the last dimension of x into [n, x.shape[-1]/n].
        """
        *start, m = x.shape
        return x.reshape(start + [n, m // n])

    def merge_heads(self, x):
        """ Reverse of split_heads.
        """
        return self.merge_states(x.permute(0, 2, 1, 3))

    @staticmethod
    def merge_states(x):
        """ Smash the last two dimensions of x into a single dimension.
        """
        *start, a, b = x.shape
        return x.reshape(start + [a * b])

    def mask_attn_weights(self, w):
        # w has shape [batch, heads, dst_sequence, src_sequence],
        # where information flows from src to dst.
        _, _, nd, ns = w.shape
        b = self.attention_mask(nd, ns, dtype=w.dtype, device=w.device)
        b = b.reshape((1, 1, nd, ns))
        w = w * b - 1e10 * (1 - b)
        return w

    @staticmethod
    def attention_mask(nd, ns, *, dtype, device=None):
        """ 1's in the lower triangle, counting from the lower right corner.
        Same as tf.matrix_band_part(tf.ones([nd, ns]), -1, ns-nd),
        but doesn't produce garbage on TPUs.
        """
        i = torch.arange(0, nd).unsqueeze(1)
        j = torch.arange(ns)
        return (i >= j - ns + nd).to(dtype=dtype, device=device)

    def multihead_attn(self, q, k, v):
        # q, k, v have shape [batch, heads, sequence, features]
        w = torch.matmul(q, k.permute(0, 1, 3, 2))
        w = w / math.sqrt(v.shape[-1])
        w = self.mask_attn_weights(w)
        w = F.softmax(w, dim=-1)
        a = torch.matmul(w, v)
        return a


def gelu(x, c=math.sqrt(2 / math.pi)):
    return 0.5 * x * (1 + torch.tanh(c * (x + 0.044715 * torch.pow(x, 3))))


def position_for(batch_size, n_steps, past_length, device=None):
    return (torch.arange(past_length, n_steps + past_length, device=device)
            .unsqueeze(0).repeat(batch_size, 1))



####################################################
# PreTrainedModel is a sub-class of torch.nn.Module
# which take care of loading and saving pretrained weights
# and various common utilities.
#
# Here you just need to specify a few (self-explanatory)
# pointers for your model and the weights initialization
# method if its not fully covered by PreTrainedModel's default method
####################################################
class KOGPTPreTrainedModel(PreTrainedModel):
    """ An abstract class to handle weights initialization and
        a simple interface for dowloading and loading pretrained models.
    """
    config_class = KOGPTConfig
    pretrained_model_archive_map = KOGPT_PRETRAINED_MODEL_ARCHIVE_MAP
    load_tf_weights = load_tf_weights_in_kogpt
    base_model_prefix = "transformer"

    def __init__(self, *inputs, **kwargs):
        super(KOGPTPreTrainedModel, self).__init__(*inputs, **kwargs)

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding, Conv1D)):
            # Slightly different from the original huggingfaces' gpt2
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, Norm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


KOGPT_START_DOCSTRING =  r"""    OpenAI GPT-2 model was proposed in
    `Language Models are Unsupervised Multitask Learners`_
    by Alec Radford*, Jeffrey Wu*, Rewon Child, David Luan, Dario Amodei** and Ilya Sutskever**.
    It's a causal (unidirectional) transformer pre-trained using  language modeling on a very large
    corpus of ~40 GB of text data.

    This model is a PyTorch `torch.nn.Module`_ sub-class. Use it as a regular PyTorch Module and
    refer to the PyTorch documentation for all matter related to general usage and behavior.

    .. _`Language Models are Unsupervised Multitask Learners`:
        https://openai.com/blog/better-language-models/

    .. _`torch.nn.Module`:
        https://pytorch.org/docs/stable/nn.html#module

    Parameters:
        config (:class:`~transformers.KOGPTConfig`): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the configuration.
            Check out the :meth:`~transformers.PreTrainedModel.from_pretrained` method to load the model weights.
"""

KOGPT_INPUTS_DOCSTRING = r"""    Inputs:
        **input_ids**: ``torch.LongTensor`` of shape ``(batch_size, sequence_length)``:
            Indices of input sequence tokens in the vocabulary.
            GPT-2 is a model with absolute position embeddings so it's usually advised to pad the inputs on
            the right rather than the left.
            Indices can be obtained using :class:`transformers.KOGPTTokenizer`.
            See :func:`transformers.PreTrainedTokenizer.encode` and
            :func:`transformers.PreTrainedTokenizer.convert_tokens_to_ids` for details.
        **past**:
            list of ``torch.FloatTensor`` (one for each layer):
            that contains pre-computed hidden-states (key and values in the attention blocks) as computed by the model
            (see `past` output below). Can be used to speed up sequential decoding.
        **attention_mask**: (`optional`) ``torch.FloatTensor`` of shape ``(batch_size, sequence_length)``:
            Mask to avoid performing attention on padding token indices.
            Mask values selected in ``[0, 1]``:
            ``1`` for tokens that are NOT MASKED, ``0`` for MASKED tokens.
        **token_type_ids**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size, sequence_length)``:
            A parallel sequence of tokens (can be used to indicate various portions of the inputs).
            The embeddings from these tokens will be summed with the respective token embeddings.
            Indices are selected in the vocabulary (unlike BERT which has a specific vocabulary for segment indices).
        **position_ids**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size, sequence_length)``:
            Indices of positions of each input sequence tokens in the position embeddings.
            Selected in the range ``[0, config.max_position_embeddings - 1]``.
        **head_mask**: (`optional`) ``torch.FloatTensor`` of shape ``(num_heads,)`` or ``(num_layers, num_heads)``:
            Mask to nullify selected heads of the self-attention modules.
            Mask values selected in ``[0, 1]``:
            ``1`` indicates the head is **not masked**, ``0`` indicates the head is **masked**.
        **inputs_embeds**: (`optional`) ``torch.FloatTensor`` of shape ``(batch_size, sequence_length, embedding_dim)``:
            Optionally, instead of passing ``input_ids`` you can choose to directly pass an embedded representation.
            This is useful if you want more control over how to convert `input_ids` indices into associated vectors
            than the model's internal embedding lookup matrix.
"""

@add_start_docstrings("The bare KOGPT Model transformer outputting raw hidden-states without any specific head on top.",
                      KOGPT_START_DOCSTRING, KOGPT_INPUTS_DOCSTRING)
class KOGPTModel(KOGPTPreTrainedModel):
    r"""
    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **last_hidden_state**: ``torch.FloatTensor`` of shape ``(batch_size, sequence_length, hidden_size)``
            Sequence of hidden-states at the last layer of the model.
        **past**:
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            that contains pre-computed hidden-states (key and values in the attention blocks).
            Can be used (see `past` input) to speed up sequential decoding.
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

    Examples::

        tokenizer = KOGPTTokenizer.from_pretrained('kogpt')
        model = KOGPTModel.from_pretrained('kogpt')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute")).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids)
        last_hidden_states = outputs[0]  # The last hidden-state is the first element of the output tuple

    """
    def __init__(self, hparams:KOGPTconfig.hparams):

        super().__init__()
        self.hparams = hparams
        self.wpe = nn.Embedding(hparams.n_ctx, hparams.n_embed)
        nn.init.normal_(self.wpe.weight, std=0.01)
        self.wte = nn.Embedding(hparams.n_vocab, hparams.n_embed)
        nn.init.normal_(self.wte.weight, std=0.02)
        self.blocks = nn.ModuleList(
            [Block(hparams) for _ in range(hparams.n_layer)])
        self.ln_f = Norm(self.hparams.n_hidden)
        if hparams.n_hidden != hparams.n_embed:
            self.in_proj = Conv1D(hparams.n_embed, hparams.n_hidden)
            self.out_proj = Conv1D(hparams.n_hidden, hparams.n_embed)
        else:
            self.in_proj = self.out_proj = None

    def forward(self, x, past=None):
        # Embedding
        past_length = 0 if past is None else past.shape[-2]
        batch_size, n_ctx = x.shape
        position = position_for(batch_size, n_ctx, past_length, x.device)
        h = self.wte(x) + self.wpe(position)
        assert h.shape == (batch_size, n_ctx, self.hparams.n_embed)
        if self.in_proj:
            h = self.in_proj(h)
        # Transformer
        presents = []
        for i, block in enumerate(self.blocks):
            if self.hparams.gradient_checkpointing:
                h, present = torch.utils.checkpoint.checkpoint(block, h, past[:, i] if past is not None else None)
            else:
                h, present = block(h, past=past[:, i] if past is not None else None)
            presents.append(present)
        h = self.ln_f(h)
        if self.out_proj:
            h = self.out_proj(h)
        # Output logits
        h_flat = h.reshape([batch_size * n_ctx, self.hparams.n_embed])
        logits = torch.matmul(h_flat, self.wte.weight.t())
        logits = logits.reshape([batch_size, n_ctx, self.hparams.n_vocab])
        return {
            'presents': torch.stack(tuple(presents), dim=1),
            'logits': logits,
            }



@add_start_docstrings("""The KOGPT Model transformer with a language modeling head on top
(linear layer with weights tied to the input embeddings). """, KOGPT_START_DOCSTRING, KOGPT_INPUTS_DOCSTRING)
class KOGPTLMHeadModel(KOGPTPreTrainedModel):
    r"""
        **labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size, sequence_length)``:
            Labels for language modeling.
            Note that the labels **are shifted** inside the model, i.e. you can set ``lm_labels = input_ids``
            Indices are selected in ``[-1, 0, ..., config.vocab_size]``
            All labels set to ``-1`` are ignored (masked), the loss is only
            computed for labels in ``[0, ..., config.vocab_size]``

    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **loss**: (`optional`, returned when ``labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Language modeling loss.
        **prediction_scores**: ``torch.FloatTensor`` of shape ``(batch_size, sequence_length, config.vocab_size)``
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        **past**:
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            that contains pre-computed hidden-states (key and values in the attention blocks).
            Can be used (see `past` input) to speed up sequential decoding.
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

    Examples::

        import torch
        from transformers import KOGPTTokenizer, KOGPTLMHeadModel

        tokenizer = KOGPTTokenizer.from_pretrained('kogpt')
        model = KOGPTLMHeadModel.from_pretrained('kogpt')

        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute")).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, labels=input_ids)
        loss, logits = outputs[:2]

    """
    def __init__(self, config):
        super(KOGPTLMHeadModel, self).__init__(config)
        self.transformer = KOGPTModel(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.init_weights()

    def get_output_embeddings(self):
        return self.lm_head

    def forward(self, input_ids=None, past=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, inputs_embeds=None,
                labels=None):
        transformer_outputs = self.transformer(input_ids,
                                               past=past,
                                               attention_mask=attention_mask,
                                               token_type_ids=token_type_ids,
                                               position_ids=position_ids,
                                               head_mask=head_mask,
                                               inputs_embeds=inputs_embeds)
        hidden_states = transformer_outputs[0]

        lm_logits = self.lm_head(hidden_states)

        outputs = (lm_logits,) + transformer_outputs[1:]
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)),
                            shift_labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), lm_logits, presents, (all hidden_states), (attentions)


@add_start_docstrings("""The KOGPT Model transformer with a language modeling and a multiple-choice classification
head on top e.g. for RocStories/SWAG tasks. The two heads are two linear layers.
The language modeling head has its weights tied to the input embeddings,
the classification head takes as input the input of a specified classification token index in the input sequence).
""", KOGPT_START_DOCSTRING, KOGPT_INPUTS_DOCSTRING)
class KOGPTDoubleHeadsModel(KOGPTPreTrainedModel):
    r"""
        **mc_token_ids**: (`optional`, default to index of the last token of the input) ``torch.LongTensor`` of shape ``(batch_size, num_choices)``:
            Index of the classification token in each input sequence.
            Selected in the range ``[0, input_ids.size(-1) - 1[``.
        **lm_labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size, sequence_length)``:
            Labels for language modeling.
            Note that the labels **are shifted** inside the model, i.e. you can set ``lm_labels = input_ids``
            Indices are selected in ``[-1, 0, ..., config.vocab_size]``
            All labels set to ``-1`` are ignored (masked), the loss is only
            computed for labels in ``[0, ..., config.vocab_size]``
        **mc_labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size)``:
            Labels for computing the multiple choice classification loss.
            Indices should be in ``[0, ..., num_choices]`` where `num_choices` is the size of the second dimension
            of the input tensors. (see `input_ids` above)

    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **lm_loss**: (`optional`, returned when ``lm_labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Language modeling loss.
        **mc_loss**: (`optional`, returned when ``multiple_choice_labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Multiple choice classification loss.
        **lm_prediction_scores**: ``torch.FloatTensor`` of shape ``(batch_size, num_choices, sequence_length, config.vocab_size)``
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        **mc_prediction_scores**: ``torch.FloatTensor`` of shape ``(batch_size, num_choices)``
            Prediction scores of the multiplechoice classification head (scores for each choice before SoftMax).
        **past**:
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            that contains pre-computed hidden-states (key and values in the attention blocks).
            Can be used (see `past` input) to speed up sequential decoding.
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

    Examples::

        import torch
        from transformers import KOGPTTokenizer, KOGPTDoubleHeadsModel

        tokenizer = KOGPTTokenizer.from_pretrained('kogpt')
        model = KOGPTDoubleHeadsModel.from_pretrained('kogpt')

        # Add a [CLS] to the vocabulary (we should train it also!)
        tokenizer.add_special_tokens({'cls_token': '[CLS]'})
        model.resize_token_embeddings(len(tokenizer))  # Update the model embeddings with the new vocabulary size
        print(tokenizer.cls_token_id, len(tokenizer))  # The newly token the last token of the vocabulary

        choices = ["Hello, my dog is cute [CLS]", "Hello, my cat is cute [CLS]"]
        encoded_choices = [tokenizer.encode(s) for s in choices]
        cls_token_location = [tokens.index(tokenizer.cls_token_id) for tokens in encoded_choices]

        input_ids = torch.tensor(encoded_choices).unsqueeze(0)  # Batch size: 1, number of choices: 2
        mc_token_ids = torch.tensor([cls_token_location])  # Batch size: 1

        outputs = model(input_ids, mc_token_ids=mc_token_ids)
        lm_prediction_scores, mc_prediction_scores = outputs[:2]

    """

    def __init__(self, config):
        super(KOGPTDoubleHeadsModel, self).__init__(config)
        self.transformer = KOGPTModel(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.multiple_choice_head = SequenceSummary(config)

        self.init_weights()

    def get_output_embeddings(self):
        return self.lm_head

    def forward(self, input_ids=None, past=None, attention_mask=None, token_type_ids=None, position_ids=None,
                head_mask=None, inputs_embeds=None,
                mc_token_ids=None, lm_labels=None, mc_labels=None):
        transformer_outputs = self.transformer(input_ids,
                                               past=past,
                                               attention_mask=attention_mask,
                                               token_type_ids=token_type_ids,
                                               position_ids=position_ids,
                                               head_mask=head_mask,
                                               inputs_embeds=inputs_embeds)

        hidden_states = transformer_outputs[0]

        lm_logits = self.lm_head(hidden_states)
        mc_logits = self.multiple_choice_head(hidden_states, mc_token_ids).squeeze(-1)

        outputs = (lm_logits, mc_logits) + transformer_outputs[1:]
        if mc_labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(mc_logits.view(-1, mc_logits.size(-1)),
                            mc_labels.view(-1))
            outputs = (loss,) + outputs
        if lm_labels is not None:
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = lm_labels[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)),
                            shift_labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (lm loss), (mc loss), lm logits, mc logits, presents, (all hidden_states), (attentions)