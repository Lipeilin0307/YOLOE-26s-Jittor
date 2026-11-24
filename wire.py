"""NumPy 1.x/2.x neutral frames for the trusted local CPU worker."""
import pickle
import numpy as np


def pack(value):
    if isinstance(value,np.ndarray):
        return ("__ndarray__",value.dtype.str,value.shape,value.tobytes(order="C"))
    if isinstance(value,tuple): return tuple(pack(v) for v in value)
    if isinstance(value,dict): return {k:pack(v) for k,v in value.items()}
    return value


def unpack(value):
    if isinstance(value,tuple):
        if len(value)==4 and isinstance(value[0],str) and value[0]=="__ndarray__":
            return np.frombuffer(value[3],dtype=value[1]).reshape(value[2])
        return tuple(unpack(v) for v in value)
    if isinstance(value,dict): return {k:unpack(v) for k,v in value.items()}
    return value


def write(stream,value):
    pickle.dump(pack(value),stream,protocol=5); stream.flush()


def read(stream):
    return unpack(pickle.load(stream))
