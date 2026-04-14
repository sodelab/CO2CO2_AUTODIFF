import os
import sys
import ctypes
import numpy as np

# Find the shared library in the same directory as this file
_pkg_dir = os.path.dirname(__file__)

# --- Platform-aware library loading ---
if sys.platform == "darwin":
    _lib_name = "libCO2CO2.dylib"
# elif sys.platform == "win32":
#     _lib_name = "CO2CO2.dll"  # Assuming this is the name for Windows
else:  # Linux and other Unix-like OS
    _lib_name = "libCO2CO2.so"

_path = os.path.join(_pkg_dir, _lib_name)

try:
    lib = ctypes.CDLL(_path)
except OSError:
    raise ImportError(
        f"Could not load shared library: {_path}. "
        f"File may be missing or incompatible with the architecture. "
        f"Platform: {sys.platform}, arch: {os.uname().machine}"
    )

# ---- Dimension and version getters ----

# lib.get_p1b_dim.restype = ctypes.c_int
# lib.get_p2b_dim.restype = ctypes.c_int
# lib.get_p2b_4_dim.restype = ctypes.c_int
# lib.get_p2b_5_dim.restype = ctypes.c_int
# lib.get_sapt_dim.restype = ctypes.c_int
#lib.get_version.restype = ctypes.c_char_p

# def get_p1b_dim():
#     return lib.get_p1b_dim()

# def get_p2b_dim():
#     return lib.get_p2b_dim()

# def get_p2b_4_dim():
#     return lib.get_p2b_4_dim()

# def get_p2b_5_dim():
#     return lib.get_p2b_5_dim()

# def get_sapt_dim():
#     return lib.get_sapt_dim()

def get_version():
    return "0.5.3"  # Hardcoded for now; can switch to dynamic retrieval if needed
    #return lib.get_version().decode("utf-8")

# # ---- 1B: Monomer energy, gradient, hessian ----

# lib.p1b.argtypes = [ctypes.POINTER(ctypes.c_double)]
# lib.p1b.restype = ctypes.c_double

# lib.p1b_gradient.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double)]
# lib.p1b_gradient.restype = None

# lib.p1b_hessian_rev.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double)]
# lib.p1b_hessian_rev.restype = None

# lib.p1b_hessian_fwd.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double)]
# lib.p1b_hessian_fwd.restype = None

# def p1b(xyz):
#     """Monomer energy (expects xyz as length-9 np.array)"""
#     arr = np.ascontiguousarray(xyz, dtype=np.double)
#     return lib.p1b(arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))

# def p1b_gradient(xyz):
#     """Monomer gradient (returns np.array shape (9,))"""
#     arr = np.ascontiguousarray(xyz, dtype=np.double)
#     grad = np.zeros(9, dtype=np.double)
#     lib.p1b_gradient(
#         arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
#         grad.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
#     )
#     return grad

# def p1b_hessian_rev(xyz):
#     arr = np.ascontiguousarray(xyz, dtype=np.double)
#     hess = np.zeros(9*9, dtype=np.double)
#     lib.p1b_hessian_rev(
#         arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
#         hess.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
#     )
#     return hess.reshape(9, 9)

# def p1b_hessian_fwd(xyz):
#     arr = np.ascontiguousarray(xyz, dtype=np.double)
#     hess = np.zeros(9*9, dtype=np.double)
#     lib.p1b_hessian_fwd(
#         arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
#         hess.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
#     )
#     return hess.reshape(9, 9)

# # ---- 2B: Dimer energy, gradient, hessian (4th order) ----

# lib.p2b_4.argtypes = [ctypes.POINTER(ctypes.c_double)]
# lib.p2b_4.restype = ctypes.c_double

# lib.p2b_gradient_4.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double)]
# lib.p2b_gradient_4.restype = None

# lib.p2b_hessian_4_rev.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double)]
# lib.p2b_hessian_4_rev.restype = None

# lib.p2b_hessian_4_fwd.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double)]
# lib.p2b_hessian_4_fwd.restype = None

# def p2b_4(xyz):
#     arr = np.ascontiguousarray(xyz, dtype=np.double)
#     return lib.p2b_4(arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))

# def p2b_gradient_4(xyz):
#     arr = np.ascontiguousarray(xyz, dtype=np.double)
#     grad = np.zeros(18, dtype=np.double)
#     lib.p2b_gradient_4(
#         arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
#         grad.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
#     )
#     return grad

# def p2b_hessian_4_rev(xyz):
#     arr = np.ascontiguousarray(xyz, dtype=np.double)
#     hess = np.zeros(18*18, dtype=np.double)
#     lib.p2b_hessian_4_rev(
#         arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
#         hess.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
#     )
#     return hess.reshape(18, 18)

# def p2b_hessian_4_fwd(xyz):
#     arr = np.ascontiguousarray(xyz, dtype=np.double)
#     hess = np.zeros(18*18, dtype=np.double)
#     lib.p2b_hessian_4_fwd(
#         arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
#         hess.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
#     )
#     return hess.reshape(18, 18)

# # ---- 2B: Dimer energy, gradient, hessian (5th order) ----

# lib.p2b_5.argtypes = [ctypes.POINTER(ctypes.c_double)]
# lib.p2b_5.restype = ctypes.c_double

# lib.p2b_gradient_5.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double)]
# lib.p2b_gradient_5.restype = None

# lib.p2b_hessian_5_rev.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double)]
# lib.p2b_hessian_5_rev.restype = None

# lib.p2b_hessian_5_fwd.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double)]
# lib.p2b_hessian_5_fwd.restype = None

# def p2b_5(xyz):
#     arr = np.ascontiguousarray(xyz, dtype=np.double)
#     return lib.p2b_5(arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))

# def p2b_gradient_5(xyz):
#     arr = np.ascontiguousarray(xyz, dtype=np.double)
#     grad = np.zeros(18, dtype=np.double)
#     lib.p2b_gradient_5(
#         arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
#         grad.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
#     )
#     return grad

# def p2b_hessian_5_rev(xyz):
#     arr = np.ascontiguousarray(xyz, dtype=np.double)
#     hess = np.zeros(18*18, dtype=np.double)
#     lib.p2b_hessian_5_rev(
#         arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
#         hess.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
#     )
#     return hess.reshape(18, 18)

# def p2b_hessian_5_fwd(xyz):
#     arr = np.ascontiguousarray(xyz, dtype=np.double)
#     hess = np.zeros(18*18, dtype=np.double)
#     lib.p2b_hessian_5_fwd(
#         arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
#         hess.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
#     )
#     return hess.reshape(18, 18)


# # ---- SAPT-S: energy, gradient, hessian ----

# lib.sapt.argtypes = [ctypes.POINTER(ctypes.c_double)]
# lib.sapt.restype = ctypes.c_double

# lib.sapt_gradient.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double)]
# lib.sapt_gradient.restype = None

# lib.sapt_hessian_rev.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double)]
# lib.sapt_hessian_rev.restype = None

# lib.sapt_hessian_fwd.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double)]
# lib.sapt_hessian_fwd.restype = None

# def sapt(xyz):
#     arr = np.ascontiguousarray(xyz, dtype=np.double)
#     return lib.sapt(arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))

# def sapt_gradient(xyz):
#     arr = np.ascontiguousarray(xyz, dtype=np.double)
#     grad = np.zeros(18, dtype=np.double)
#     lib.sapt_gradient(
#         arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
#         grad.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
#     )
#     return grad

# def sapt_hessian_rev(xyz):
#     arr = np.ascontiguousarray(xyz, dtype=np.double)
#     hess = np.zeros(18*18, dtype=np.double)
#     lib.sapt_hessian_rev(
#         arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
#         hess.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
#     )
#     return hess.reshape(18, 18)

# def sapt_hessian_fwd(xyz):
#     arr = np.ascontiguousarray(xyz, dtype=np.double)
#     hess = np.zeros(18*18, dtype=np.double)
#     lib.sapt_hessian_fwd(
#         arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
#         hess.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
#     )
#     return hess.reshape(18, 18)

# # Default p2b functions (5th order)
# p2b = p2b_5
# p2b_gradient = p2b_gradient_5
# p2b_hessian_rev = p2b_hessian_5_rev
# p2b_hessian_fwd = p2b_hessian_5_fwd

# # ---- SAPT-SR: short-range (exponential only) energy, gradient, hessian ----

# lib.sapt_sr.argtypes = [ctypes.POINTER(ctypes.c_double)]
# lib.sapt_sr.restype = ctypes.c_double

# lib.sapt_sr_gradient.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double)]
# lib.sapt_sr_gradient.restype = None

# lib.sapt_sr_hessian_rev.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double)]
# lib.sapt_sr_hessian_rev.restype = None

# lib.sapt_sr_hessian_fwd.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double)]
# lib.sapt_sr_hessian_fwd.restype = None

# def sapt_sr(xyz):
#     arr = np.ascontiguousarray(xyz, dtype=np.double)
#     return lib.sapt_sr(arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))

# def sapt_sr_gradient(xyz):
#     arr = np.ascontiguousarray(xyz, dtype=np.double)
#     grad = np.zeros(18, dtype=np.double)
#     lib.sapt_sr_gradient(
#         arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
#         grad.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
#     )
#     return grad

# def sapt_sr_hessian_rev(xyz):
#     arr = np.ascontiguousarray(xyz, dtype=np.double)
#     hess = np.zeros(18*18, dtype=np.double)
#     lib.sapt_sr_hessian_rev(
#         arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
#         hess.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
#     )
#     return hess.reshape(18, 18)

# def sapt_sr_hessian_fwd(xyz):
#     arr = np.ascontiguousarray(xyz, dtype=np.double)
#     hess = np.zeros(18*18, dtype=np.double)
#     lib.sapt_sr_hessian_fwd(
#         arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
#         hess.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
#     )
#     return hess.reshape(18, 18)

# # ---- SAPT-LR: long-range (QQ + C6 + C8) energy, gradient, hessian ----

# lib.sapt_lr.argtypes = [ctypes.POINTER(ctypes.c_double)]
# lib.sapt_lr.restype = ctypes.c_double

# lib.sapt_lr_gradient.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double)]
# lib.sapt_lr_gradient.restype = None

# lib.sapt_lr_hessian_rev.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double)]
# lib.sapt_lr_hessian_rev.restype = None

# lib.sapt_lr_hessian_fwd.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double)]
# lib.sapt_lr_hessian_fwd.restype = None

# def sapt_lr(xyz):
#     arr = np.ascontiguousarray(xyz, dtype=np.double)
#     return lib.sapt_lr(arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))

# def sapt_lr_gradient(xyz):
#     arr = np.ascontiguousarray(xyz, dtype=np.double)
#     grad = np.zeros(18, dtype=np.double)
#     lib.sapt_lr_gradient(
#         arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
#         grad.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
#     )
#     return grad

# def sapt_lr_hessian_rev(xyz):
#     arr = np.ascontiguousarray(xyz, dtype=np.double)
#     hess = np.zeros(18*18, dtype=np.double)
#     lib.sapt_lr_hessian_rev(
#         arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
#         hess.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
#     )
#     return hess.reshape(18, 18)

# def sapt_lr_hessian_fwd(xyz):
#     arr = np.ascontiguousarray(xyz, dtype=np.double)
#     hess = np.zeros(18*18, dtype=np.double)
#     lib.sapt_lr_hessian_fwd(
#         arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
#         hess.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
#     )
#     return hess.reshape(18, 18)

# # Example usage
# if __name__ == "__main__":
#     # Example: dimer geometry
#     xyz = np.array([
#         0.0,  0.0,  0.000,
#         0.0,  0.0, -1.162,
#         0.0,  0.0,  1.162,
#         3.75, 0.0,  0.000,
#         3.75, 0.0, -1.162,
#         3.75, 0.0,  1.162
#     ], dtype=np.double)
#     print("p2b_5 energy:", p2b_5(xyz))
#     print("SAPT-S energy:", sapt(xyz))
#     print("SAPT-SR energy:  ", sapt_sr(xyz))
#     print("SAPT-LR energy:  ", sapt_lr(xyz))
#     print("SR + LR == SAPT: ", abs(sapt_sr(xyz) + sapt_lr(xyz) - sapt(xyz)) < 1e-10)
#     print("p2b_5 gradient:", p2b_gradient_5(xyz))
#     print("p2b_5 Hessian:", p2b_hessian_5_fwd(xyz))
