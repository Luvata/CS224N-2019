# Assignment 5: Self-attention, Transformers and Pretraining

## 1. Attention exploration

[TODO]

## 2. Pretrained Transformer models and knowledge access
- [colab notebook](https://colab.research.google.com/drive/1eUwaEZFWhOMjUtDTKKPcaUjOGJHrzixf?usp=sharing)

- [x] c. Implement finetuning (without pretraining)
- [x] d. Make predictions (without pretraining).
    - Model's accuracy on dev set: `Correct: 8.0 out of 500.0: 1.6%`
    - `london_baseline`: `Correct: 25.0 out of 500.0: 5.0%`

- [x] e. Define a span corruption function for pretraining.
    - In this question, you’ll implement a simplification that only masks out a 
    single sequence of characters.

- [x] f. Pretrain, finetune, and make predictions
    - Accuracy on dev set: `Correct: 129.0 out of 500.0: 25.8%` 

- [ ] g. Research! Write and try out the synthesizer variant
    - Accuracy on dev set: `Correct: 50.0 out of 500.0: 10.0%`
    - Why might the synthesizer self-attention not be able to do, in a single layer, what the key-query-value self-attention can do ?
        - [TODO]

## 3. Considerations in pretrained knowledge

[TODO]
