#!/usr/bin/env python
"""Random projection layer in MXNet as a custom python op.
INCOMPLETE work in progress.
"""
import os
# MXNET_CPU_WORKER_NTHREADS must be greater than 1 for custom op to work on CPU
os.environ["MXNET_CPU_WORKER_NTHREADS"] = "2"
#os.environ["MXNET_ENGINE_TYPE"] = "NaiveEngine"  # See https://github.com/dmlc/mxnet/issues/3813

import numpy as np
import mxnet as mx


# ref: http://mxnet.io/how_to/new_op.html


class RandomProjection(mx.operator.CustomOp):
    """Random projection embedding layer.
    Takes an n-hot (sparse) input layer, effectively on coordinate (COO) format,
    where the row number is implicit, because it's the minibatch record.
    """

    def __init__(self, vocab_size, projected_dimensions):
        # need_top_grad=True means this is not a loss layer
        super(RandomProjection, self).__init__()
        self._vocab = vocab_size
        self._proj_dim = projected_dimensions
        self.W = np.random.normal(size=(self._vocab,self._proj_dim))

    def forward(self, is_train, req, in_data, out_data, aux):
        print("running forwards code in randomprojection")
        #Note: see this run in notebooks/howto-numpy-random-proj.ipynb
        # Notation for shapes: b = batch_size, mnz = max_nonzero, d = proj_dim
        idx = in_data[0].asnumpy().astype('int32') # shape=(b,mnz)

        wd = self.W[idx]  # shape= (b,mnz,d)
        mask = idx >= 0  # bool False for -1 values that should be removed. shape=(b,mnz)
        mask = np.expand_dims(mask,2) # shape = (b,mnz,1)
        mask = np.repeat(mask, self._proj_dim, axis=2) # shape = (b,mnz,d)
        wd = np.multiply(wd,mask)  # shape=(b,mnz,d), but zero'd out non-masked
        y = np.sum(wd,axis=1)  # shape=(b,d)
        print("Computed Y w/ shape=%s" % str(y.shape))
        print(y)
        print("Converting to NDArray")
        mxy = mx.nd.array(y)  # this hangs!?
        print("Assigning")

        self.assign(out_data[0], req[0], mxy)
        print("Done")


@mx.operator.register("randomprojection")
class RandomProjectionProp(mx.operator.CustomOpProp):
    def __init__(self, vocab_size, projected_dimensions):
        # need_top_grad=True means this is not a loss layer
        super(RandomProjectionProp, self).__init__(need_top_grad=True)
        self._kwargs = {
            'vocab_size': int(vocab_size),
            'projected_dimensions': int(projected_dimensions),
        }
        print("rproj-prop constructed w/ args %s" % str(self._kwargs))

    def list_arguments(self):
        return ['indexes', 'values']  #NOTE: Values currently ignored.

    def list_outputs(self):
        return ['output']

    def create_operator(self, ctx, shapes, dtypes, **kwargs):
        return RandomProjection(**self._kwargs)

    def infer_shape(self, in_shape):
        print("infer_shape for randomproj got %s" % str(in_shape))
        # check that indexes and values are the same shape.
        if in_shape[0] != in_shape[1]:
            raise ValueError("Input shapes differ. indexes:%s. values:%s. must be same"
                    % (str(in_shape[0]),str(in_shape[1])))
        batch_size = in_shape[0][0]
        output_shape = (batch_size, self._kwargs['projected_dimensions'])
        return in_shape, [output_shape], []


if __name__ == "__main__":
    print("Simple test of proj layer")
    data = mx.symbol.Variable('data')
    vals = mx.symbol.Variable('vals')
    net = mx.symbol.Custom(indexes=data, values=vals, name='rproj', 
            op_type='randomprojection', 
            vocab_size=999, projected_dimensions=29)
    d = mx.nd.zeros(shape=(3,100))
    v = mx.nd.ones(shape=(3,100))
    e = net.bind(ctx=mx.cpu(), args={'data':d, 'vals':v})
    e.forward()
    print(e.outputs[0].asnumpy())
    print("Done with simple test")

