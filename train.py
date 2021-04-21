from models import Embedding, EncoderRNN, Generator, \
    DecoderRNN, Seq2Seq, Discriminator
from losses import translation_loss, classification_loss
from utils import noise, log_probs2indices

import torch
import torch.nn as nn
from torch import optim


class Trainer:
    def __init__(self, frozen_src2tgt: Seq2Seq, frozen_tgt2src: Seq2Seq,
                 src_embedding: Embedding, tgt_embedding: Embedding,
                 encoder_rnn, decoder_rnn, src_hat: Generator, tgt_hat: Generator,
                 discriminator: Discriminator, hidden_size, max_length,
                 src_sos_index, tgt_sos_index, src_eos_index, tgt_eos_index, src_pad_index, tgt_pad_index,
                 device, lr_core=0.0003, lr_disc=0.0005, use_attention=True,
                 use_cuda=True):

        self.word_word_translate = True
        self.frozen_src2tgt = frozen_src2tgt
        self.frozen_tgt2src = frozen_tgt2src

        self.src_embedding = src_embedding
        self.tgt_embedding = tgt_embedding
        self.encoder_rnn = encoder_rnn
        self.decoder_rnn = decoder_rnn
        self.src_hat = src_hat
        self.tgt_hat = tgt_hat

        self.discriminator = discriminator
        self.hidden_size = hidden_size
        self.max_length = max_length

        self.src_sos_index = src_sos_index
        self.tgt_sos_index = tgt_sos_index
        self.src_eos_index = src_eos_index
        self.tgt_eos_index = tgt_eos_index
        self.src_pad_index = src_pad_index
        self.tgt_pad_index = tgt_pad_index
        self.device = device
        self.lr_core = lr_core
        self.lr_disc = lr_disc

        self.use_cuda = use_cuda
        self.use_attention = use_attention

    def re_init_params(self):

        self.core_model = nn.ModuleList([
            self.src_embedding,	self.tgt_embedding, self.encoder_rnn,
            self.decoder_rnn, self.src_hat, self.tgt_hat
        ])
        self.core_model.to(self.device)
        self.discriminator.to(self.device)

        self.src2src = Seq2Seq(self.src_embedding, self.src_embedding, self.encoder_rnn, self.decoder_rnn,
                               self.hidden_size, self.src_hat, self.max_length, use_cuda=self.use_cuda,
                               use_attention=self.use_attention).to(self.device)

        self.src2tgt = Seq2Seq(self.src_embedding, self.tgt_embedding, self.encoder_rnn, self.decoder_rnn,
                               self.hidden_size, self.tgt_hat, self.max_length, use_cuda=self.use_cuda,
                               use_attention=self.use_attention).to(self.device)

        self.tgt2tgt = Seq2Seq(self.tgt_embedding, self.tgt_embedding, self.encoder_rnn, self.decoder_rnn,
                               self.hidden_size, self.tgt_hat, self.max_length, use_cuda=self.use_cuda,
                               use_attention=self.use_attention).to(self.device)

        self.tgt2src = Seq2Seq(self.tgt_embedding, self.src_embedding, self.encoder_rnn, self.decoder_rnn,
                               self.hidden_size, self.src_hat, self.max_length, use_cuda=self.use_cuda,
                               use_attention=self.use_attention).to(self.device)

        self.core_optimizer = optim.Adam(self.core_model.parameters(), lr=self.lr_core, betas=(0.5, 0.999))
        self.discriminator_optimizer = optim.RMSprop(self.discriminator.parameters(), lr=self.lr_disc)

    def train_step(self, batch, weights=(1, 1, 1), drop_probability=0.1, permutation_constraint=3):
        batch = {l: t.to(self.device) for l, t in batch.items()}

        source_length, current_batch = noise(batch['src'], self.src_pad_index, drop_probability, permutation_constraint)
        src2src_dec, src2src_enc = self.src2src(
            current_batch.to(self.device), source_length.to(self.device),
            self.src_sos_index, batch['src']
        )

        target_length, current_batch = noise(batch['tgt'], self.tgt_pad_index, drop_probability, permutation_constraint)
        tgt2tgt_dec, tgt2tgt_enc = self.tgt2tgt(
            current_batch.to(self.device), target_length.to(self.device),
            self.tgt_sos_index, batch['tgt']
        )

        if self.word_word_translate is True:

            length, current_batch = \
                noise(self.frozen_src2tgt(batch['src']), self.tgt_pad_index, drop_probability, permutation_constraint)
            tgt2src_dec, tgt2src_enc = self.tgt2src(
                current_batch.to(self.device), length.to(self.device),
                self.src_sos_index, batch['src']
            )

            length, current_batch = \
                noise(self.frozen_tgt2src(batch['tgt']), self.src_pad_index, drop_probability, permutation_constraint)
            src2tgt_dec, src2tgt_enc = self.src2tgt(
                current_batch.to(self.device), length.to(self.device),
                self.tgt_sos_index, batch['tgt']
            )

        else:
            src2tgt_translation = \
                log_probs2indices(self.frozen_src2tgt.evaluate(batch['src'], source_length, self.tgt_sos_index,
                                                               self.tgt_eos_index, n_iters=self.max_length))
            src2tgt_translation.requires_grad = False

            tgt2src_translation = \
                log_probs2indices(self.frozen_tgt2src.evaluate(batch['tgt'], target_length, self.src_sos_index,
                                                               self.src_eos_index,  n_iters=self.max_length))
            tgt2src_translation.requires_grad = False

            length, current_batch = \
                noise(src2tgt_translation, self.tgt_pad_index, drop_probability, permutation_constraint)
            tgt2src_dec, tgt2src_enc = self.tgt2src(
                current_batch.to(self.device),
                length.to(self.device), self.src_sos_index, batch['src']
            )

            length, current_batch = \
                noise(tgt2src_translation, self.src_pad_index, drop_probability, permutation_constraint)
            src2tgt_dec, src2tgt_enc = self.src2tgt(
                current_batch.to(self.device),
                length.to(self.device), self.tgt_sos_index, batch['tgt']
            )

        # core_loss is the loss for updating the model except for discriminator
        # DEA loss
        core_loss = weights[0] * (
                translation_loss(src2src_dec, batch['src']) +
                translation_loss(tgt2tgt_dec, batch['tgt'])
        )

        # cross domain loss
        core_loss += weights[1] * (
            translation_loss(tgt2src_dec, batch['src']) +
            translation_loss(src2tgt_dec, batch['tgt'])
        )

        # adversarial loss
        core_loss += weights[2] * (
            classification_loss(self.discriminator(src2src_enc), 'tgt') +
            classification_loss(self.discriminator(tgt2tgt_enc), 'src') +
            classification_loss(self.discriminator(tgt2src_enc), 'src') +
            classification_loss(self.discriminator(src2tgt_enc), 'tgt')
        )

        # update core model's parameters
        self.core_optimizer.zero_grad()
        core_loss.backward(retain_graph=True)
        self.core_optimizer.step()

        # training discriminator
        discriminator_loss = \
            classification_loss(self.discriminator(src2src_enc.detach()), 'src') + \
            classification_loss(self.discriminator(tgt2tgt_enc.detach()), 'tgt') + \
            classification_loss(self.discriminator(tgt2src_enc.detach()), 'tgt') + \
            classification_loss(self.discriminator(src2tgt_enc.detach()), 'src')

        # update discriminator parameters
        self.discriminator_optimizer.zero_grad()
        discriminator_loss.backward()
        self.discriminator_optimizer.step()

        return core_loss.item(), discriminator_loss.item()

    def load(self, directory):
        for layer, name in [(self.__getattribute__(name), name)
                            for name in ['src_embedding', 'tgt_embedding', 'encoder_rnn', 'decoder_rnn',
                                         'src_hat', 'tgt_hat', 'discriminator']]:
            layer.load_state_dict(torch.load(directory + name))

    def save(self, directory):
        for layer, name in [(self.__getattribute__(name), name)
                            for name in ['src_embedding', 'tgt_embedding', 'encoder_rnn', 'decoder_rnn',
                                         'src_hat', 'tgt_hat', 'discriminator']]:
            torch.save(layer.state_dict(), directory + name)

    def predict(self, batch, length, l1='src', l2='tgt', n_iters=None):
        model = {('src', 'src'): self.src2src, ('src', 'tgt'): self.src2tgt,
                 ('tgt', 'src'): self.tgt2src, ('tgt', 'tgt'): self.tgt2tgt}[(l1, l2)]
        sos_index, eos_index = (self.src_sos_index, self.src_eos_index) if l2 == 'src' \
            else (self.tgt_sos_index, self.tgt_eos_index)
        return log_probs2indices(model.evaluate(batch.to(self.device),
                                                length.to(self.device), sos_index, eos_index, n_iters=n_iters))

    def predict_on_test(self, batch, visualize, l1='src', l2='tgt', n_iters=None):
        length, batch = batch
        predict = visualize(self.predict(batch, length, l1=l1, l2=l2, n_iters=n_iters), l2)
        return predict