from tutorial.char_rnn_classification_tutorial.evalute import *
from torch.utils.tensorboard import SummaryWriter

# default `log_dir` is "runs" - we'll be more specific here


def predict(input_line, n_predictions=3):
    print('\n> %s' % input_line)
    with torch.no_grad():
        output = evaluate(lineToTensor(input_line))

        # Get top N categories
        topv, topi = output.topk(n_predictions, 1, True)
        predictions = []

        for i in range(n_predictions):
            value = topv[0][i].item()
            category_index = topi[0][i].item()
            print('(%.2f) %s' % (value, all_categories[category_index]))
            predictions.append([value, all_categories[category_index]])


def write_tensorboard():
    writer = SummaryWriter('runs/char_rnn_classification')
    input = lineToTensor("wang")
    hidden = rnn.initHidden()
    writer.add_graph(rnn, [input[0], hidden])
    writer.close()


def test():
    predict('Dovesky')
    predict('Jackson')
    predict('Satoshi')
    write_tensorboard()


def main():
    test()


if __name__ == '__main__':
    main()
