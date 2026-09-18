"""Small optional CUDA Driver graph wrapper; no dependency without CUDA use."""
import ctypes as ct
import os


class Graph:
    def __init__(self,stream,record):
        self.lib=ct.WinDLL('nvcuda.dll') if os.name=='nt' else ct.CDLL('libcuda.so.1')
        self.graph=ct.c_void_p();self.execution=ct.c_void_p()
        specs={
            'cuStreamBeginCapture':[ct.c_void_p,ct.c_int],
            'cuStreamEndCapture':[ct.c_void_p,ct.POINTER(ct.c_void_p)],
            'cuGraphInstantiateWithFlags':[ct.POINTER(ct.c_void_p),ct.c_void_p,ct.c_ulonglong],
            'cuGraphLaunch':[ct.c_void_p,ct.c_void_p],
            'cuGraphDestroy':[ct.c_void_p],
            'cuGraphExecDestroy':[ct.c_void_p],
        }
        for name,types in specs.items():
            fn=getattr(self.lib,name);fn.argtypes=types;fn.restype=ct.c_int
        self.check(self.lib.cuStreamBeginCapture(stream.handle,0))
        try:
            record()
        except BaseException:
            self.lib.cuStreamEndCapture(stream.handle,ct.byref(self.graph))
            self.close()
            raise
        self.check(self.lib.cuStreamEndCapture(stream.handle,ct.byref(self.graph)))
        self.check(self.lib.cuGraphInstantiateWithFlags(ct.byref(self.execution),self.graph,0))

    @staticmethod
    def check(code):
        if code:raise RuntimeError(f'CUDA graph driver error {code}')

    def launch(self,stream):
        self.check(self.lib.cuGraphLaunch(self.execution,stream.handle))

    def close(self):
        if getattr(self,'execution',None) and self.execution.value:
            self.lib.cuGraphExecDestroy(self.execution);self.execution=ct.c_void_p()
        if getattr(self,'graph',None) and self.graph.value:
            self.lib.cuGraphDestroy(self.graph);self.graph=ct.c_void_p()

    def __del__(self):
        try:self.close()
        except Exception:pass
