import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack
import numpy as np

"""
Code is based on https://github.com/IlyaGusev/UNMT/blob/master/src/models.py
"""

class Embedding(nn.Module):
    def __init__(self, emb_matrix, requires_grad=True):
        super(Embedding, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(torch.from_numpy(emb_matrix), freeze=not requires_grad)
        self.vocab_size = self.embedding.num_embeddings
        self.embedding_dim = self.embedding.embedding_dim

    def forward(self, inputs):
        return self.embedding(inputs)


class EncoderRNN(nn.Module):
    def __init__(self, embedding: Embedding, rnn):
        super(EncoderRNN, self).__init__()

        self.embedding = embedding
        self.rnn = rnn

    def forward(self, input_seqs, input_lengths, hidden=None):
        self.rnn.flatten_parameters()
        # print(input_seqs.size())  # length * batch_size
        embedded = self.embedding(input_seqs)
        # print(embedded.size())
        # print(input_lengths)
        packed = pack(embedded, input_lengths, enforce_sorted=False, batch_first=False)
        outputs, hidden = self.rnn(packed, hidden)
        outputs, output_lengths = unpack(outputs)

        # for bidirectional
        n = hidden[0].size(0)
        hidden = (torch.cat([hidden[0][0:n:2], hidden[0][1:n:2]], 2),
                  torch.cat([hidden[1][0:n:2], hidden[1][1:n:2]], 2))
        return outputs, hidden


# for generating new proposal based on current hidden state
class Generator(nn.Module):
    def __init__(self, hidden_size, vocab_size):
        super(Generator, self).__init__()

        self.hidden_size = hidden_size
        self.vocab_size = vocab_size

        self.out = nn.Linear(hidden_size, vocab_size)
        self.sm = nn.LogSoftmax(dim=1)

    def forward(self, inputs):
        assert inputs.size(1) == self.hidden_size
        return self.sm(self.out(inputs))


class DecoderRNN(nn.Module):
    def __init__(self, embedding: Embedding, hidden_size, max_length,
                 rnn, generator, use_cuda=False, use_attention=True):
        super(DecoderRNN, self).__init__()

        self.embedding = embedding
        self.embedding_dim = self.embedding.embedding_dim
        self.hidden_size = hidden_size
        self.vocab_size = self.embedding.vocab_size
        self.use_cuda = use_cuda
        self.max_length = max_length
        self.use_attention = use_attention

        if self.use_attention:
            self.attn = nn.Linear(hidden_size + self.embedding_dim, self.max_length, bias=False)
            self.attn_sm = nn.Softmax(dim=1)
            self.attn_out = nn.Linear(hidden_size + self.embedding_dim, self.embedding_dim, bias=False)
            self.attn_out_relu = nn.ReLU()

        self.rnn = rnn
        self.generator = generator

    def step(self, batch_input, hidden, encoder_output):
        # batch_input: B
        # hidden: (n_layers x B x N, n_layers x B x N)
        # encoder_output: L x B x N
        # output: 1 x B x N
        # embedded:  B x E
        # attn_weights: B x 1 x L
        # context: B x 1 x N
        # rnn_input: B x N

        embedded = self.embedding(batch_input)
        self.rnn.flatten_parameters()
        if self.use_attention:
            attn_weights = self.attn_sm(self.attn(torch.cat((embedded, hidden[0][-1]), 1))).unsqueeze(1)
            max_length = encoder_output.size(0)
            context = torch.bmm(attn_weights[:, :, :max_length], encoder_output.transpose(0, 1))
            rnn_input = torch.cat((embedded, context.squeeze(1)), 1)
            rnn_input = self.attn_out_relu(self.attn_out(rnn_input))
        else:
            rnn_input = embedded
        output, hidden = self.rnn(rnn_input.unsqueeze(0), hidden)
        return output, hidden

    def init_state(self, batch_size, sos_index):
        initial_input = Variable(torch.zeros((batch_size,)).type(torch.LongTensor), requires_grad=False)
        initial_input = torch.add(initial_input, sos_index)
        initial_input = initial_input.cuda() if self.use_cuda else initial_input
        return initial_input

    def forward(self, current_input, hidden, length, encoder_output, gtruth=None):
        outputs = Variable(torch.zeros(length, current_input.size(0), self.vocab_size), requires_grad=False)
        outputs = outputs.cuda() if self.use_cuda else outputs
        self.rnn.flatten_parameters()
        for t in range(length):
            output, hidden = self.step(current_input, hidden, encoder_output)
            scores = self.generator.forward(output.squeeze(0))
            outputs[t] = scores
            if gtruth is None:
                top_indices = scores.topk(1, dim=1)[1].view(-1)
                current_input = top_indices
            else:
                current_input = gtruth[t]
        # print(outputs.size())
        return outputs, hidden

    def evaluate(self, current_input, hidden, encoder_outputs, eos_index, n_iters=None):
        """
        hidden: n_layers x batch_size x (1 or 2)*hidden_size
        encoder_outputs: length x batch_size x (1 or 2)*hidden_size
        input: batch_size
        """
        outputs = []
        ended = np.zeros(hidden[0].size()[1])
        # np.all: test whether all array elements along a given axis evaluate to True
        # ~np.all(ended) = True since np.all(ended) = False
        while ~np.all(ended) and n_iters != 0:
            output, hidden = self.step(current_input, hidden, encoder_outputs)
            generator_output = self.generator(output.squeeze(0))
            # print(generator_output.size())  # size = [batch_size * vocab_size]
            outputs.append(generator_output.unsqueeze(0))
            input = torch.topk(generator_output, k=1)[1].squeeze(-1)
            # print(input.size())  # size = [batch_size]
            current_input = input
            ended += (input == eos_index).cpu().numpy() if self.use_cuda else (input == eos_index).numpy()
            if n_iters is not None:
                n_iters -= 1

        return torch.cat(outputs)


class Discriminator(nn.Module):
    def __init__(self, max_length, encoder_hidden_size, hidden_size, n_layers):
        super(Discriminator, self).__init__()

        self.encoder_hidden_size = encoder_hidden_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.max_length = max_length

        layers = list()
        layers.append(nn.Linear(encoder_hidden_size * max_length, hidden_size))
        layers.append(nn.LeakyReLU())
        for i in range(n_layers - 2):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.LeakyReLU())
        self.layers = nn.ModuleList(layers)
        self.out = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, encoder_output):
        max_length = encoder_output.size(0)
        batch_size = encoder_output.size(1)
        output = encoder_output.transpose(0, 1).contiguous().view(batch_size, max_length * self.encoder_hidden_size)
        output = F.pad(output, (0, (self.max_length - max_length) * self.encoder_hidden_size), "constant", 0)
        # S = batch_size, max_length * encoder_hidden_size
        for i in range(len(self.layers)):
            output = self.layers[i](output)
        return self.sigmoid(self.out(output))


class Seq2Seq(nn.Module):
    # rnn_size is the size of the hidden state
    def __init__(self, encoder_embedding: Embedding, decoder_embedding: Embedding,
                 encoder_rnn, decoder_rnn, hidden_size, generator: Generator,
                 max_length, use_cuda, use_attention=True):
        super(Seq2Seq, self).__init__()

        self.hidden_size = hidden_size
        self.max_length = max_length
        self.use_cuda = use_cuda
        self.use_attention = use_attention

        self.encoder = EncoderRNN(encoder_embedding, rnn=encoder_rnn)
        self.decoder = DecoderRNN(decoder_embedding, hidden_size, rnn=decoder_rnn,
                                  generator=generator, max_length=max_length,
                                  use_cuda=use_cuda, use_attention=use_attention)

    def forward(self, variable, lengths, sos_index, gtruth=None):
        encoder_output, encoder_hidden = self.encoder.forward(variable, lengths)
        current_input = self.decoder.init_state(variable.size(1), sos_index)
        max_length = self.max_length
        if gtruth is not None:
            max_length = min(self.max_length, gtruth.size(0))
        decoder_output, _ = self.decoder.forward(current_input, encoder_hidden, max_length,
                                                 encoder_output, gtruth)

        return decoder_output, encoder_output

    def evaluate(self, variable, lengths, sos_index, eos_index, n_iters=None):
        encoder_outputs, hidden = self.encoder(variable, lengths)
        current_input = self.decoder.init_state(variable.size(1), sos_index)

        decoder_output = \
            self.decoder.evaluate(current_input, hidden, encoder_outputs, eos_index, n_iters=n_iters)

        return decoder_output

