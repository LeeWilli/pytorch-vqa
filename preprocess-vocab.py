import json
from collections import Counter
import itertools

import config
import data
import utils


def extract_vocab(iterable, top_k=None, start=0):
    """ Turns an iterable of list of tokens into a vocabulary.
        These tokens could be single answers or word tokens in questions.
    """
    all_tokens = itertools.chain.from_iterable(iterable)
    counter = Counter(all_tokens)
    if top_k:
        most_common = counter.most_common(top_k)
        most_common = (t for t, c in most_common)
    else:
        most_common = counter.keys()
    # descending in count, then lexicographical order
    tokens = sorted(most_common, key=lambda x: (counter[x], x), reverse=True)
    vocab = {t: i for i, t in enumerate(tokens, start=start)}
    vocabi = {int(i): t for i, t in enumerate(tokens, start=start)}
    return vocab, vocabi

def extract_vocabi(iterable, top_k=None, start=0):
    """ Turns an iterable of list of tokens into a vocabulary.
        These tokens could be single answers or word tokens in questions.
    """
    all_tokens = itertools.chain.from_iterable(iterable)
    counter = Counter(all_tokens)
    if top_k:
        most_common = counter.most_common(top_k)
        most_common = (t for t, c in most_common)
    else:
        most_common = counter.keys()
    # descending in count, then lexicographical order
    tokens = sorted(most_common, key=lambda x: (counter[x], x), reverse=True)
    ivocab = {i: t for i, t in enumerate(tokens, start=start)}
    return ivocab


def main():
    questions = utils.path_for(train=True, question=True)
    answers = utils.path_for(train=True, answer=True)

    with open(questions, 'r') as fd:
        questions = json.load(fd)
    with open(answers, 'r') as fd:
        answers = json.load(fd)

    questions = data.prepare_questions(questions)
    answers = data.prepare_answers(answers)

    question_vocab, question_vocabi = extract_vocab(questions, start=1)
    answer_vocab, answer_vocabi = extract_vocab(answers, top_k=config.max_answers)

    vocabs = {
        'question': question_vocab,
        'answer': answer_vocab,
    }
    with open(config.vocabulary_path, 'w') as fd:
        json.dump(vocabs, fd)

    print(answer_vocabi)
    vocabsi = {
        'answeri': answer_vocabi,
    }
    with open(config.vocabularyi_path, 'w') as fd:
        json.dump(vocabsi, fd)


if __name__ == '__main__':
    main()
