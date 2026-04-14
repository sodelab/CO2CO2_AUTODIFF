#ifndef POLY_2B_H
#define POLY_2B_H

#include <autodiff/forward/real.hpp>

//
// reduced (no bonds breaking) permutational symmetry
// including only 2B terms
//
//    x[0] = @VAR@-|x-intra-CO|(Ca, Oa1);
//    x[1] = @VAR@-|x-intra-CO|(Ca, Oa2);
//    x[2] = @VAR@-|x-intra-OO|(Oa1, Oa2);
//    x[3] = @VAR@-|x-intra-CO|(Cb, Ob1);
//    x[4] = @VAR@-|x-intra-CO|(Cb, Ob2);
//    x[5] = @VAR@-|x-intra-OO|(Ob1, Ob2);
//    x[6] = @VAR@-|x-CC|(Ca, Cb);
//    x[7] = @VAR@-|x-CO|(Ca, Ob1);
//    x[8] = @VAR@-|x-CO|(Ca, Ob2);
//    x[9] = @VAR@-|x-CO|(Cb, Oa1);
//    x[10] = @VAR@-|x-CO|(Cb, Oa2);
//    x[11] = @VAR@-|x-OO|(Oa1, Ob1);
//    x[12] = @VAR@-|x-OO|(Oa1, Ob2);
//    x[13] = @VAR@-|x-OO|(Oa2, Ob1);
//    x[14] = @VAR@-|x-OO|(Oa2, Ob2);
//
namespace x2b {

// Original function for backward compatibility
void eval_fourth(double x[15], double p[596]);

// Templated version for autodiff
template<typename T>
void eval_fourth_t(T x[15], T p[596]);

// Original function for backward compatibility
void eval_fifth(double x[15], double p[2269]);

// Templated version for autodiff
template<typename T>
void eval_fifth_t(T x[15], T p[2269]);

// Explicitly tell the compiler about the autodiff version
extern template void eval_fourth_t<autodiff::real>(autodiff::real x[15], autodiff::real p[596]);
extern template void eval_fifth_t<autodiff::real>(autodiff::real x[15], autodiff::real p[2269]);
}

#endif // POLY_2B_H
