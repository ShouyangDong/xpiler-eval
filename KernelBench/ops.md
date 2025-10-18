| Operator Category | Description                     | Operator Names                              |
|-------------------|---------------------------------|---------------------------------------------|
| Element-wise      | Per-element computation         | add, sign, sub, sin                         |
| Reduction         | Reduces along a dimension       | sum, mean, max, min                         |
| Matrix Ops        | Matrix-matrix or matrix-vector  | gemm, gemv, bmm, dense                      |
| Convolution       | Spatial feature extraction      | conv1d, conv2dNHWC, con2dNCHW, DepthwiseConv|
| Normalization     | Normalize inputs/features       | batchnorm, layernorm, RMSnorm, instancenorm          |
| Activation        | Non-linear transformations      | relu, gelu, sigmoid, softmax                |
| Pooling           | Downsample feature maps         | maxpool2d, avgpool2d, minpool2d, sumpool2d  |
| Layout Transform  | Reorganize data layout          | gather, transpose, scatter, concat                                      |
| LLM               | Ops used in LLM                 | self-atten, DAT, GateMLP, GQA                         |
