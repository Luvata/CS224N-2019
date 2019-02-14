## Assignment 3: Dependency parsing


### Prob 1. Machine learning and Neural Networks 
#### A. Adam optimizer

Reference :
  1. [Optimizing gradient descent - Sebastian Ruder](http://ruder.io/optimizing-gradient-descent/index.html)
  2. [CS231N - Neural networks 3](http://cs231n.github.io/neural-networks-3/)

- i. Momentum :
    - The momentum term **increases** for dimensions whose **gradients point in the same directions** and reduces updates for dimensions whose gradients change directions
    - Gain faster convergence and reduced oscillation
- ii. Adaptive learning rate:
    - Used to **normalize** the parameter update step, element wise
    - Weights that receive high gradients will have their effective learning rate reduced
    - Weights that receive small / infrequent updates will have effective learning rate increased
#### B. Dropout
- $\gamma$ = 1/p for scaling output on training (Not sure)
- Dropout in training but not in testing :
    - At test time all neurons see all their inputs, so we want the outputs of neurons at test time to be identical to their expected outputs at training time

### Prob 2. Neural Transition-Based dependency parsing
#### A. Parsing a sentence

| Stack                                 | Buffer                                     | New dependency    | Transition |step|
|---------------------------------------|--------------------------------------------|-------------------|------------|----|
| [ROOT]                                |     [I, parsed, this, sentence, correctly] |                   | Init       |0   |
| [ROOT, I]                             |     [parsed, this, sentence, correctly]    |                   | Shift      |1   |
| [ROOT, I, parsed]                     |     [this, sentence, correctly]            |                   | Shift      |2   |
| [ROOT, parsed]                        |     [this, sentence, correctly]            | I <- parsed       | Left-Arc   |3   |
| [ROOT, parsed, this]                  |     [ sentence, correctly]                 |                   | Shift      |4   |
| [ROOT, parsed, this, sentence]        |               [correctly]                  |                   | Shift      |5   |
| [ROOT, parsed, sentence]              |               [correctly]                  |this <- sentence   | Left-Arc   |6   |
| [ROOT, parsed]                        |               [correctly]                  |parsed -> sentence | Right-Arc  |7   |
| [ROOT, parsed, correctly]             |               []                           |                   | Shift      |8   |
| [ROOT, parsed]                        |               []                           |parsed -> correctly| Right-Arc  |9   |
| [ROOT]                                |               []                           |Root -> parsed     | Right-Arc  |10  |

#### B. Number of steps

A sentence contain N words will be parsed in 2N steps:
 - Need total N "SHIFT" operations to read all words in a sentence
 - Each time using an "Arc" operation, one word in stack is removed
 - When finish parsing, Stack only contains ROOT
 - So total removed words in Stack is N, which mean we need N "Arc" operations
 - Total steps: N (SHIFT) + N (Arc) = 2N

#### F. Four types of parsing error examples (Need contribution):

- Prepositional Phrase Attachment Error
- Verb Phrase Attachment Error
- Modifier Attachment Error
- Coordination Attachment Error

#### EVALUATION:
After train, model has 88.46 UAS on DEV set and 88.85 UAS on Test set
#### TAKE AWAY:
- Feature extraction for Neural Dependency Parsing: 
  - Indices of top *n* words in stacks and buffers, and its children
  - Embedding of words, POS tags and dependency relations can be trained together 
- Adaptive learning rate affects the generalization of model
- Pytorch
```python
# Save and load weight
torch.save(model.state_dict(), path)
model.load_state_dict(torch.load(path))

## Drop out layer
model.train() # put this before train to enable train mode : apply drop out to model
model.eval() # don't apply drop out when evaluate

## nn.Embedding take input as vector of index, return embedding vectors
```