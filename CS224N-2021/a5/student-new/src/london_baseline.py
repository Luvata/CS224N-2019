# Calculate the accuracy of a baseline that simply predicts "London" for every
#   example in the dev set.
# Hint: Make use of existing code.
# Your solution here should only be a few lines.
import utils


eval_corpus_path = "./birth_dev.tsv"
with open(eval_corpus_path) as fin:
    predictions = ["London" for _ in fin]
    total, correct = utils.evaluate_places(eval_corpus_path, predictions)
    print('Correct: {} out of {}: {}%'.format(correct, total, correct/total*100))
