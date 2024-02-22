from numba import jit, cuda,vectorize, float32, int32, int64, float64, guvectorize, njit
import numpy as np
# to measure exec time
from timeit import default_timer as timer   

from numba import cuda
print(cuda.gpus)
  
# normal function to run on cpu
@njit
def func(k,a):  
    for i in range(k):
        temp = i*i
        temp_1 = temp + 8
        a[i] = i + temp_1  
    return a 

  
# function optimized to run on gpu
# @jit(target='cuda')
@guvectorize(['void(int64,float32[:], float32[:])'],'(),(n)->(n)', \
    target='cuda', nopython=True)
# @jit()
# @cuda.jit
def func2(k,a,res):
    for i in range(k):
        temp = i*i
        temp_1 = temp + 8
        res[i] = i + temp_1  


# @vectorize([int32(int32, int32),
#             int64(int64, int64),
#             float32(float32, float32),
#             float64(float64, float64)])
# def f(x, y):
#     return x + y

if __name__=="__main__":
    k = 100000000                            
    a = np.ones(k, dtype = np.float32)
    b = np.ones(k, dtype = np.float32)
      
    start = timer()
    func(k,a)
    print("without GPU:", timer()-start)    
      
    start = timer()
    func2(k,a,b)
    print("with GPU with compliation:", timer()-start)

    start = timer()
    func2(k,a,b)
    print("with GPU after compliation:", timer()-start)
    # a = np.arange(6)
    # b= f(a, a)
    # print(b)


    # a = np.array([0,1])
    # b = np.array([2,3])
    # c = a - b
    # d = c**2
    # print(c)
    # print(d)