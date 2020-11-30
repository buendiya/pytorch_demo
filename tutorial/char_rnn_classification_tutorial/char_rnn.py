# -*-coding:utf-8-*-

from tutorial.char_rnn_classification_tutorial.preprocess import  *
from tutorial.char_rnn_classification_tutorial.model import *


n_hidden = 128
rnn = RNN(n_letters, n_hidden, n_categories)

