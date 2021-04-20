# NMT Assignment
Note: Heavily inspired by the https://github.com/pcyin/pytorch_nmt repository

#### WRITING QUESTIONS
- Problem 1
  - 1G : ```generate_sent_masks()```
    - Mask has shape (b, max_slen), and contains 1s in positions corresponding to 'pad' tokens
    - Because 'pad' tokens don't contain any information relevant to the context of sentence, it's just a placeholder to 
    normalize different sentences' length, mask was applied to 'pad' tokens' to map its score into ```-inf``` before 
    calculating attention vector, in order to zero out probability of 'pad' tokens in attention vector
  
  - 1I : After training, model has Corpus BLEU: 22.51906462908753

  - 1J : Advantage and disadvantage of attention variants
    - $e_{t, i}$ is the similarity score of decoder time-step **t** w.r.t encoder time-step **i**
    - Dot and Multiplicative attention was introduced by (Luong et al., 2015). Dot product attention directly measures
    attention with dot (s_t and h_i must have same dimension), whereas in Multiplicative attention, *W* was introduced as
    a weighted similarity for a more general notion 
    - Reference : http://ruder.io/deep-learning-nlp-best-practices/index.html#attention:
    - > Additive and multiplicative attention are similar in complexity, although multiplicative attention is faster 
    and more space-efficient in practice as it can be implemented more efficiently using matrix multiplication
    - > Both variants perform similar for small dimensionality $d_h$ of the decoder states, 
    but additive attention performs better for larger dimensions.
    

|  Attention mechanism   |                              Formula           |
|------------------------|------------------------------------------------|
|Dot product attention   |$$e_{t,i} = s_t^T h_i$$                         |
|Multiplicative attention|$$e_{t,i} = s_t^T W h_i$$                       |
|Additive attention      |$$e_{t,i} = e_{t, i} = v^T (W_1 h_i + W_2 s_t)$$|

    
- Problem 2 
 

#### TAKE AWAY

- **LOVE** Python 3.5 type hints, reminds me of scala :D
- Always check **bias** in Linear layer, default is True

```python
# (2, 4, 1, 4, 2).squeeze() -> (2, 4, 4, 2). specific dimension will squeeze this dim
# (2, 4, 4, 2).unsqueeze(-1) -> (2, 4, 4, 2, 1)
# torch.cat(( T(2, 4), T(2, 3) ), dim=1) -> T(2, 7)
# torch.split(T(5, 2, 1), 1, dim=0) # split dimension 0, 1 each : [T(2, 1)] * 5
# torch.bmm( T(a, b, c), T(a, c, d)) -> T(a, b, d)
```