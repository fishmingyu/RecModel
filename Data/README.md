## Rec sys data generator

We use the methods represented in paper _Scalable Realistic Recommendation Datasets through Fractal Expansions_. [link](https://arxiv.org/pdf/1901.08910.pdf)
In this paper, it gives a way to expand the original dataset based on svd method.

### How to run

```shell
python gen.py [file path] [down sampling factor]
```

### Dataflow

- sparse svd of input sparse dataset-->u, s, v
- image resize of u, v matrices
- apply the frobenius norm to u, v-->u*, v*
- matmul u*, s, v*
- use kronecker product to generate final output

**Refinement**
In the original paper, it apply $A\otimes_F B = F(a, B, w)$, where w is pseudo-random number. Here we only use the original definition of kronecker product.

---

**Requirement**

```
skimage
scipy >= 1.7
numpy
```
