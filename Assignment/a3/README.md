## Assignment 3: Dependency parsing


### Prob 1. Machine learning and Neural Networks [TODO]
#### A. Adam optimizer
- i. Momentum
- ii. Adaptive learning rate
#### B. Dropout
- $\gamma$ equal ?
- Dropout in training but not in testing ?

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

#### F. Four types of parsing error examples [TODO]:

- Prepositional Phrase Attachment Error: "Moscow sent troops into Afghanistan"
- Verb Phrase Attachment Error: "Leaving the store unattended, I went
outside to watch the parade, the phrase leaving the store unattended"
- Modifier Attachment Error: "I am extremely short"
- Coordination Attachment Error: "Would you like brown rice or garlic naan?"