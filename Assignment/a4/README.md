# NMT Assignment
Note: Heavily inspired by the https://github.com/pcyin/pytorch_nmt repository

#### WRITING QUESTIONS
- Problem 1
  - 1G : ```generate_sent_masks()```
  - 1J : Advantage and disadvantage of attention variants
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