#include "poly-1b.h"

// AutoDiff scalar types
#include <autodiff/forward/real.hpp>     // real, real2nd, …
#include <autodiff/forward/dual.hpp>     // dual, dual2nd, …
#include <autodiff/reverse/var.hpp>      // var

namespace x1b
{

//-----------------------------------------------------------------------------
// 1‐body polynomial basis evaluation (non‐AD double version)
//-----------------------------------------------------------------------------
void eval(const double x[3], double *p)
{
    // Dispatch to templated implementation
    eval_t<double>(x, p);
}

//-----------------------------------------------------------------------------
// Templated polynomial basis evaluation (works for real, var, dual2nd, etc.)
//-----------------------------------------------------------------------------
template<typename T>
void eval_t(const T x[3], T *p)
{
    // A 3‐coordinate input generates 49 polynomial basis functions:
    p[0]  = x[2];
    p[1]  = x[1] + x[0];
    p[2]  = x[0]*x[0] + x[1]*x[1];
    p[3]  = x[0]*x[2] + x[1]*x[2];
    p[4]  = x[0]*x[1];
    p[5]  = x[2]*x[2];
    p[6]  = x[0]*x[0]*x[2] + x[1]*x[1]*x[2];
    p[7]  = x[0]*x[0]*x[0] + x[1]*x[1]*x[1];
    p[8]  = x[0]*x[1]*x[1] + x[0]*x[0]*x[1];
    p[9]  = x[0]*x[1]*x[2];
    p[10] = x[1]*x[2]*x[2] + x[0]*x[2]*x[2];
    p[11] = x[2]*x[2]*x[2];
    p[12] = x[0]*x[0]*x[0]*x[1] + x[0]*x[1]*x[1]*x[1];
    p[13] = x[0]*x[0]*x[1]*x[1];
    p[14] = x[0]*x[0]*x[2]*x[2] + x[1]*x[1]*x[2]*x[2];
    p[15] = x[2]*x[2]*x[2]*x[2];
    p[16] = x[1]*x[1]*x[1]*x[2] + x[0]*x[0]*x[0]*x[2];
    p[17] = x[1]*x[1]*x[1]*x[1] + x[0]*x[0]*x[0]*x[0];
    p[18] = x[0]*x[1]*x[1]*x[2] + x[0]*x[0]*x[1]*x[2];
    p[19] = x[0]*x[2]*x[2]*x[2] + x[1]*x[2]*x[2]*x[2];
    p[20] = x[0]*x[1]*x[2]*x[2];
    p[21] = x[0]*x[2]*x[2]*x[2]*x[2] + x[1]*x[2]*x[2]*x[2]*x[2];
    p[22] = x[0]*x[0]*x[2]*x[2]*x[2] + x[1]*x[1]*x[2]*x[2]*x[2];
    p[23] = x[1]*x[1]*x[1]*x[1]*x[2] + x[0]*x[0]*x[0]*x[0]*x[2];
    p[24] = x[0]*x[0]*x[0]*x[0]*x[1] + x[0]*x[1]*x[1]*x[1]*x[1];
    p[25] = x[0]*x[0]*x[0]*x[0]*x[0] + x[1]*x[1]*x[1]*x[1]*x[1];
    p[26] = x[0]*x[0]*x[1]*x[2]*x[2] + x[0]*x[1]*x[1]*x[2]*x[2];
    p[27] = x[0]*x[1]*x[1]*x[1]*x[2] + x[0]*x[0]*x[0]*x[1]*x[2];
    p[28] = x[1]*x[1]*x[1]*x[2]*x[2] + x[0]*x[0]*x[0]*x[2]*x[2];
    p[29] = x[2]*x[2]*x[2]*x[2]*x[2];
    p[30] = x[0]*x[0]*x[0]*x[1]*x[1] + x[0]*x[0]*x[1]*x[1]*x[1];
    p[31] = x[0]*x[1]*x[2]*x[2]*x[2];
    p[32] = x[0]*x[0]*x[1]*x[1]*x[2];
    p[33] = x[1]*x[2]*x[2]*x[2]*x[2]*x[2] + x[0]*x[2]*x[2]*x[2]*x[2]*x[2];
    p[34] = x[0]*x[1]*x[2]*x[2]*x[2]*x[2];
    p[35] = x[1]*x[1]*x[1]*x[1]*x[2]*x[2] + x[0]*x[0]*x[0]*x[0]*x[2]*x[2];
    p[36] = x[0]*x[0]*x[0]*x[1]*x[2]*x[2] + x[0]*x[1]*x[1]*x[1]*x[2]*x[2];
    p[37] = x[0]*x[0]*x[1]*x[1]*x[1]*x[2] + x[0]*x[0]*x[0]*x[1]*x[1]*x[2];
    p[38] = x[0]*x[0]*x[0]*x[0]*x[0]*x[0] + x[1]*x[1]*x[1]*x[1]*x[1]*x[1];
    p[39] = x[0]*x[0]*x[0]*x[1]*x[1]*x[1];
    p[40] = x[0]*x[0]*x[0]*x[0]*x[0]*x[1] + x[0]*x[1]*x[1]*x[1]*x[1]*x[1];
    p[41] = x[0]*x[1]*x[1]*x[1]*x[1]*x[2] + x[0]*x[0]*x[0]*x[0]*x[1]*x[2];
    p[42] = x[0]*x[0]*x[1]*x[1]*x[2]*x[2];
    p[43] = x[0]*x[0]*x[1]*x[1]*x[1]*x[1] + x[0]*x[0]*x[0]*x[0]*x[1]*x[1];
    p[44] = x[2]*x[2]*x[2]*x[2]*x[2]*x[2];
    p[45] = x[0]*x[0]*x[0]*x[0]*x[0]*x[2] + x[1]*x[1]*x[1]*x[1]*x[1]*x[2];
    p[46] = x[1]*x[1]*x[1]*x[2]*x[2]*x[2] + x[0]*x[0]*x[0]*x[2]*x[2]*x[2];
    p[47] = x[1]*x[1]*x[2]*x[2]*x[2]*x[2] + x[0]*x[0]*x[2]*x[2]*x[2]*x[2];
    p[48] = x[0]*x[1]*x[1]*x[2]*x[2]*x[2] + x[0]*x[0]*x[1]*x[2]*x[2]*x[2];
}

} // namespace x1b

//-----------------------------------------------------------------------------
// Explicit template instantiations
//-----------------------------------------------------------------------------
template void x1b::eval_t<autodiff::real>    (const autodiff::real   [3], autodiff::real   *);
template void x1b::eval_t<autodiff::real2nd>(const autodiff::real2nd[3], autodiff::real2nd*);
template void x1b::eval_t<autodiff::var>     (const autodiff::var    [3], autodiff::var    *);
template void x1b::eval_t<autodiff::dual2nd> (const autodiff::dual2nd[3], autodiff::dual2nd*);
