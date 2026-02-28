#ifndef POLY_1B_H
#define POLY_1B_H

// AutoDiff core types
#include <autodiff/forward/real.hpp>  // real, real2nd, …
#include <autodiff/forward/dual.hpp>  // dual, dual2nd, …
#include <autodiff/reverse/var.hpp>   // var

//
// reduced (no bonds breaking) permutational symmetry
// including only 1B terms
//
//    x[0] = @VAR@-|x-intra-CO|(Ca, Oa1);
//    x[1] = @VAR@-|x-intra-CO|(Ca, Oa2);
//    x[2] = @VAR@-|x-intra-OO|(Oa1, Oa2);
//
//

namespace x1b
{

//-----------------------------------------------------------------------------
// Evaluate 1-body polynomial for double inputs
//-----------------------------------------------------------------------------
void eval(const double x[3], double *mono);

//-----------------------------------------------------------------------------
// Templated version for AD types (real, real2nd, dual2nd, var, …)
//-----------------------------------------------------------------------------
template<typename T>
void eval_t(const T x[3], T *mono);

//-----------------------------------------------------------------------------
// Explicit template instantiations (to reduce compile times)
//-----------------------------------------------------------------------------
extern template void eval_t<autodiff::real>    (const autodiff::real    x[3], autodiff::real    *mono);
extern template void eval_t<autodiff::real2nd>(const autodiff::real2nd x[3], autodiff::real2nd *mono);
extern template void eval_t<autodiff::dual2nd> (const autodiff::dual2nd x[3], autodiff::dual2nd *mono);
extern template void eval_t<autodiff::var>     (const autodiff::var     x[3], autodiff::var     *mono);

} // namespace x1b

#endif // POLY_1B_H
