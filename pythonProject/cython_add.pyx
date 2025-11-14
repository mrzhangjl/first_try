cdef int c_add(int a,int b):
    return a+b
def py_add(int a,int b):
    return c_add(a, b)
