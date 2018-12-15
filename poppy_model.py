# import ctypes

# _sum = ctypes.CDLL('./libsum.so')
# _sum.our_function.argtypes = (ctypes.c_int, ctypes.POINTER(ctypes.c_int))

# def our_function(numbers):
#     global _sum
#     num_numbers = len(numbers)
#     array_type = ctypes.c_int * num_numbers
#     result = _sum.our_function(ctypes.c_int(num_numbers), array_type(*numbers))
#     print result
#     return int(result)

# a=[1,2,3]
# print a
# print our_function([1,2,3])



import numpy as np
import ctypes
from numpy.ctypeslib import ndpointer

lib = ctypes.cdll.LoadLibrary('/home/chronos/.mujoco/mjpro150/Model/model1.so')

class Model(object):
    def __init__(self,set_chain, set_precision, set_grap_mode):
        #lib.Model_new.argtypes = None #[ctypes.c_void_p]
        lib.Model_new.argtypes = [ctypes.c_byte, ctypes.c_double, ctypes.c_int]
        lib.Model_new.restype = ctypes.c_void_p
        lib.Model_reset.argtypes = [ctypes.c_void_p]
        lib.Model_reset.restype = ctypes.c_void_p
        lib.Model_get_state.argtypes = [ctypes.c_void_p]
        lib.Model_get_state.restype = ndpointer(dtype=ctypes.c_double, shape=(29,))
        lib.Model_get_distance.restype = ndpointer(dtype=ctypes.c_double, shape=(2,))
        lib.Model_get_action_bound.restype = ndpointer(dtype=ctypes.c_double, shape=(2,))
        #lib.Model_step.argtypes = [ctypes.c_void_p]
        #lib.Model_step.restype = ctypes.c_double
        lib.Model_step.restype = ndpointer(dtype=ctypes.c_double, shape=(2,))
        #lib.Foo_foobar.argtypes = [ctypes.c_void_p, ctypes.c_int]
        #lib.Foo_foobar.restype = ctypes.c_int
        self.obj = lib.Model_new(set_chain, set_precision, set_grap_mode)
	#self.obj.state_dim = 27;
	#self.obj.action_dim = 13;
	#self.obj.action_bound = [-0.1,0.1];

    def reset(self):
        lib.Model_reset(self.obj)
    def step(self,action):
        array = ctypes.c_double*15
        #action_array = array(0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1)
        action_array = array(*action)
        reward = lib.Model_step(self.obj,action_array)
        return np.hstack(lib.Model_get_state(self.obj)),np.hstack(reward),lib.Model_get_stable(self.obj), np.hstack(lib.Model_get_distance(self.obj))
    def close(self):
        lib.Model_close(self.obj)
    def state(self):
        return lib.Model_get_state(self.obj);
    def stable(self):
        return lib.Model_get_stable(self.obj);
    def state_dim(self):
        return lib.Model_get_state_dim(self.obj);
    def action_dim(self):
        return lib.Model_get_action_dim(self.obj);
    def action_bound(self):
        return lib.Model_get_action_bound(self.obj);
