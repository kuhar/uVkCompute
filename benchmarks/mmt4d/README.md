# mmt4d benchmark

This directory contains microbenchmarks for evaluating different strategy to
implement matrix matrix transposed multiplication of two 4D inputs (mmt4d).

This implementation follows the naming conventions used by MLIR's `linalg.mmt4d`
op:
> The right hand side is transposed, whence the ’t' in ‘mmt’.
> The input and output tensors have a 4D shape instead of a 2D shape. They are
> interpreted as 2D matrices with one level of 2D tile subdivision, whence the
> 2+2=4 dimensions. The inner tile dimensions are identified with ‘0’ suffixes
> below, for instance the LHS matrix shape (M, K, M0, K0) reads as: MxK tiles,
> each of shape M0xK0.

The reference C++ implementation is available at:
https://github.com/iree-org/iree/blob/main/runtime/src/iree/builtins/ukernel/tools/mmt4d_test.cc.
