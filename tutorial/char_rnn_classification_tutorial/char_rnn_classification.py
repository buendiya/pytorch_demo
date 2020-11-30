import random

from tutorial.char_rnn_classification_tutorial.preprocess import *
from tutorial.char_rnn_classification_tutorial.model import *

print('n_letters: %s' % n_letters)
print('n_categories: %s' % n_categories)
n_hidden = 128
rnn = RNN(n_letters, n_hidden, n_categories)


def test():
    input = letterToTensor('A')
    hidden =torch.zeros(1, n_hidden)

    print('input.size: %s' % str(input.size()))
    print('hidden.size: %s' % str(hidden.size()))
    output, next_hidden = rnn(input, hidden)

    print(output)

    input = lineToTensor('Albert')
    hidden = torch.zeros(1, n_hidden)

    output, next_hidden = rnn(input[0], hidden)
    print(output)

    print(categoryFromOutput(output))


def randomChoice(l):
    return l[random.randint(0, len(l) - 1)]


def randomTrainingExample():
    category = randomChoice(all_categories)
    line = randomChoice(category_lines[category])
    category_tensor = torch.tensor([all_categories.index(category)], dtype=torch.long)
    line_tensor = lineToTensor(line)
    return category, line, category_tensor, line_tensor


def test2():
    for i in range(10):
        category, line, category_tensor, line_tensor = randomTrainingExample()
        print('category =', category, '/ line =', line, '/ category_tensor = ', category_tensor)


def main():
    test()
    test2()


if __name__ == '__main__':
    main()
