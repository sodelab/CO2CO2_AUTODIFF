#ifndef MBCO2CO2_H
#define MBCO2CO2_H

#include <autodiff/forward/real.hpp>

#ifdef __cplusplus
extern "C" {
#endif

// -- Original energy functions --------------------------------------------
double   p1b        (double *xyz);
double   p2b_4      (double *xyz);
double   p2b_5      (double *xyz);
double   sapt       (double *xyz);

// -- Forward‐mode gradient (C‐API) ----------------------------------------
void     p1b_gradient       (double *X, double *grad);
void     p2b_gradient_4     (double *X, double *grad);
void     p2b_gradient_5     (double *X, double *grad);
void     sapt_gradient      (double *X, double *grad);

// -- Reverse‐mode and forward‐mode Hessians (C‐API) -----------------------
void     p1b_hessian_rev    (double *X, double *hess);
void     p1b_hessian_fwd    (double *X, double *hess);

void     p2b_hessian_4_rev  (double *X, double *hess);
void     p2b_hessian_4_fwd  (double *X, double *hess);

void     p2b_hessian_5_rev  (double *X, double *hess);
void     p2b_hessian_5_fwd  (double *X, double *hess);

void     sapt_hessian_rev   (double *X, double *hess);
void     sapt_hessian_fwd   (double *X, double *hess);

// -- Dimension getters ----------------------------------------------------
int      get_p1b_dim    ();
int      get_p2b_dim    ();
int      get_p2b_4_dim  ();
int      get_p2b_5_dim  ();
int      get_sapt_dim   ();
const char* get_version ();

#ifdef __cplusplus
}
#endif


// -- Templated interfaces for any scalar type ----------------------------
template<typename T>
T p1b_t      (T *xyz);

template<typename T>
T p2b_4_t    (T *xyz);

template<typename T>
T p2b_5_t    (T *xyz);

template<typename T>
T sapt_t     (T *xyz);

template<typename T>
void p1b_gradient_t   (const T *xyz, T *grad);

template<typename T>
void p2b_gradient_4_t (const T *xyz, T *grad);

template<typename T>
void p2b_gradient_5_t (const T *xyz, T *grad);

template<typename T>
void sapt_gradient_t  (const T *xyz, T *grad);

template<typename T>
void p1b_hessian_t    (const T *xyz, T *hess);

template<typename T>
void p2b_hessian_4_t  (const T *xyz, T *hess);

template<typename T>
void p2b_hessian_5_t  (const T *xyz, T *hess);

template<typename T>
void sapt_hessian_t   (const T *xyz, T *hess);


// -- Explicit instantiation declarations for autodiff::real --------------
extern template autodiff::real p1b_t<autodiff::real>     (autodiff::real *xyz);
extern template autodiff::real p2b_4_t<autodiff::real>   (autodiff::real *xyz);
extern template autodiff::real p2b_5_t<autodiff::real>   (autodiff::real *xyz);
extern template autodiff::real sapt_t<autodiff::real>    (autodiff::real *xyz);

extern template void p1b_gradient_t<autodiff::real>      (const autodiff::real *xyz, autodiff::real *grad);
extern template void p2b_gradient_4_t<autodiff::real>    (const autodiff::real *xyz, autodiff::real *grad);
extern template void p2b_gradient_5_t<autodiff::real>    (const autodiff::real *xyz, autodiff::real *grad);
extern template void sapt_gradient_t<autodiff::real>     (const autodiff::real *xyz, autodiff::real *grad);

extern template void p1b_hessian_t<autodiff::real>       (const autodiff::real *xyz, autodiff::real *hess);
extern template void p2b_hessian_4_t<autodiff::real>     (const autodiff::real *xyz, autodiff::real *hess);
extern template void p2b_hessian_5_t<autodiff::real>     (const autodiff::real *xyz, autodiff::real *hess);
extern template void sapt_hessian_t<autodiff::real>      (const autodiff::real *xyz, autodiff::real *hess);

#endif // MBCO2CO2_H