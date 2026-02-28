#include <cmath>
#include <cassert>
#include <cstdlib>
#include <iomanip>
#include <sstream>

#include <autodiff/forward/real.hpp>
#include <autodiff/forward/real/eigen.hpp>
#include <autodiff/forward/dual.hpp>
#include <autodiff/reverse/var.hpp>
#include <autodiff/reverse/var/eigen.hpp>

#include "kit.h"
#include "poly-2b.h"
#include "x2b.h"


namespace x2b {


//-----------------------------------------------------------------------------
// 1) Switching function (fade‐in/fade‐out) based on the shortest distance
//-----------------------------------------------------------------------------
template<typename T>
T f_switch_t(const T* xyz)
{
    //find smallest intermolecular distance
    T r = kit::distance_short_t<T>(xyz);

    if (r > T(m_r2f_long)) {
        return T(0.0);
    } else if (r > T(m_r2i_long)) {
        const T x = (r - T(m_r2i_long))/(T(m_r2f_long) - T(m_r2i_long));
        return (T(1.0) + cos(T(M_PI)*x))/T(2);
    } else if (r > T(m_r2f_small)) {
        return T(1.0);
    } else if (r > T(m_r2i_small)) {
        const T x = (r - T(m_r2i_small))/(T(m_r2f_small) - T(m_r2i_small));
        //return pow(x, T(1)/T(5));
        return pow(x, 1/5);
    } else {
        return T(0.0);
    }
}

double f_switch(const double* xyz)
{
    return f_switch_t<double>(xyz);
}


//-----------------------------------------------------------------------------
// 2) Build polynomial variables from Cartesian coords (4th‐order)
//-----------------------------------------------------------------------------
template<typename T>
void cart_to_vars_fourth_t(const T* xyz, T* v, T& s)
{
    const T* Ca  = xyz;
    const T* Oa1 = xyz + 3;
    const T* Oa2 = xyz + 6;

    const T* Cb  = xyz + 9;
    const T* Ob1 = xyz + 12;
    const T* Ob2 = xyz + 15;

    using kit::distance;

    v[0]  = var_intra_t<T>(T(d0_intra_CO), T(m_k_CO_intra_fourth), distance(  Ca, Oa1));
    v[1]  = var_intra_t<T>(T(d0_intra_CO), T(m_k_CO_intra_fourth), distance(  Ca, Oa2));
    v[2]  = var_intra_t<T>(T(d0_intra_OO), T(m_k_OO_intra_fourth), distance( Oa1, Oa2));

    v[3]  = var_intra_t<T>(T(d0_intra_CO), T(m_k_CO_intra_fourth), distance(  Cb, Ob1));
    v[4]  = var_intra_t<T>(T(d0_intra_CO), T(m_k_CO_intra_fourth), distance(  Cb, Ob2));
    v[5]  = var_intra_t<T>(T(d0_intra_OO), T(m_k_OO_intra_fourth), distance( Ob1, Ob2));

    v[6]  = var_inter_t<T>(T(d0_inter_CC), T(m_k_CC_inter_fourth), distance( Ca,  Cb));

    v[7]  = var_inter_t<T>(T(d0_inter_CO), T(m_k_CO_inter_fourth), distance( Ca, Ob1));
    v[8]  = var_inter_t<T>(T(d0_inter_CO), T(m_k_CO_inter_fourth), distance( Ca, Ob2));
    v[9]  = var_inter_t<T>(T(d0_inter_CO), T(m_k_CO_inter_fourth), distance( Cb, Oa1));
    v[10] = var_inter_t<T>(T(d0_inter_CO), T(m_k_CO_inter_fourth), distance( Cb, Oa2));

    v[11] = var_inter_t<T>(T(d0_inter_OO), T(m_k_OO_inter_fourth), distance( Oa1, Ob1));
    v[12] = var_inter_t<T>(T(d0_inter_OO), T(m_k_OO_inter_fourth), distance( Oa1, Ob2));
    v[13] = var_inter_t<T>(T(d0_inter_OO), T(m_k_OO_inter_fourth), distance( Oa2, Ob1));
    v[14] = var_inter_t<T>(T(d0_inter_OO), T(m_k_OO_inter_fourth), distance( Oa2, Ob2));

    s = f_switch_t<T>(xyz);
} 

void cart_to_vars_fourth(const double* xyz, double* v, double& s)
{
    cart_to_vars_fourth_t<double>(xyz, v, s);
}


//-----------------------------------------------------------------------------
// 3) Build polynomial variables from Cartesian coords (5th‐order)
//-----------------------------------------------------------------------------
template<typename T>
void cart_to_vars_fifth_t(const T* xyz, T* v, T& s)
{
    const T* Ca  = xyz;
    const T* Oa1 = xyz + 3;
    const T* Oa2 = xyz + 6;

    const T* Cb  = xyz + 9;
    const T* Ob1 = xyz + 12;
    const T* Ob2 = xyz + 15;

    using kit::distance;

    v[0]  = var_intra_t<T>(T(d0_intra_CO), T(m_k_CO_intra_fifth), distance(  Ca, Oa1));
    v[1]  = var_intra_t<T>(T(d0_intra_CO), T(m_k_CO_intra_fifth), distance(  Ca, Oa2));
    v[2]  = var_intra_t<T>(T(d0_intra_OO), T(m_k_OO_intra_fifth), distance( Oa1, Oa2));

    v[3]  = var_intra_t<T>(T(d0_intra_CO), T(m_k_CO_intra_fifth), distance(  Cb, Ob1));
    v[4]  = var_intra_t<T>(T(d0_intra_CO), T(m_k_CO_intra_fifth), distance(  Cb, Ob2));
    v[5]  = var_intra_t<T>(T(d0_intra_OO), T(m_k_OO_intra_fifth), distance( Ob1, Ob2));

    v[6]  = var_inter_t<T>(T(d0_inter_CC), T(m_k_CC_inter_fifth), distance( Ca,  Cb));

    v[7]  = var_inter_t<T>(T(d0_inter_CO), T(m_k_CO_inter_fifth), distance( Ca, Ob1));
    v[8]  = var_inter_t<T>(T(d0_inter_CO), T(m_k_CO_inter_fifth), distance( Ca, Ob2));
    v[9]  = var_inter_t<T>(T(d0_inter_CO), T(m_k_CO_inter_fifth), distance( Cb, Oa1));
    v[10] = var_inter_t<T>(T(d0_inter_CO), T(m_k_CO_inter_fifth), distance( Cb, Oa2));

    v[11] = var_inter_t<T>(T(d0_inter_OO), T(m_k_OO_inter_fifth), distance( Oa1, Ob1));
    v[12] = var_inter_t<T>(T(d0_inter_OO), T(m_k_OO_inter_fifth), distance( Oa1, Ob2));
    v[13] = var_inter_t<T>(T(d0_inter_OO), T(m_k_OO_inter_fifth), distance( Oa2, Ob1));
    v[14] = var_inter_t<T>(T(d0_inter_OO), T(m_k_OO_inter_fifth), distance( Oa2, Ob2));

    s = f_switch_t<T>(xyz);
} 

void cart_to_vars_fifth(const double* xyz, double* v, double& s)
{
    cart_to_vars_fifth_t<double>(xyz, v, s);
}


//-----------------------------------------------------------------------------
// 4) Evaluate 2-body potential (4th & 5th order)
//-----------------------------------------------------------------------------
template<typename T>
T value_fourth_t(const T* xyz)
{
    T v[num_vars], s;
    cart_to_vars_fourth_t<T>(xyz, v, s);

    T E_poly(0);
    {   
        T mono[num_linear_params_fourth];
        eval_fourth_t<T>(v, mono);
        for (size_t n = 0; n < num_linear_params_fourth; ++n) 
            E_poly += T(poly_fourth[n])*mono[n];
    }

    return s*E_poly + sapt_s_t<T>(xyz);
}

double value_fourth(const double* xyz)
{
    return value_fourth_t<double>(xyz);
}

template<typename T>
T value_fifth_t(const T* xyz)
{
    T v[num_vars], s;
    cart_to_vars_fifth_t<T>(xyz, v, s);

    T E_poly(0);
    {   
        T mono[num_linear_params_fifth];
        eval_fifth_t<T>(v, mono);
        for (size_t n = 0; n < num_linear_params_fifth; ++n) 
            E_poly += T(poly_fifth[n])*mono[n];
    
    }

    return s*E_poly + sapt_s_t<T>(xyz);
}

double value_fifth(const double* xyz)
{
    return value_fifth_t<double>(xyz);
}

// switching function
template autodiff::real f_switch_t<autodiff::real>(const autodiff::real* xyz);
template autodiff::real2nd f_switch_t<autodiff::real2nd>(const autodiff::real2nd*);
template autodiff::dual2nd f_switch_t<autodiff::dual2nd>(const autodiff::dual2nd*);
template autodiff::var f_switch_t<autodiff::var>(const autodiff::var* xyz);

// cart_to_vars (fourth/fifth)
template void x2b::cart_to_vars_fourth_t<autodiff::real>(const autodiff::real*, autodiff::real x[15], autodiff::real&);
template void x2b::cart_to_vars_fourth_t<autodiff::dual2nd>(const autodiff::dual2nd*, autodiff::dual2nd x[15], autodiff::dual2nd&);
template void x2b::cart_to_vars_fourth_t<autodiff::var>(const autodiff::var*, autodiff::var x[15], autodiff::var&);

template void x2b::cart_to_vars_fifth_t<autodiff::real>(const autodiff::real*, autodiff::real x[15], autodiff::real&);
template void x2b::cart_to_vars_fifth_t<autodiff::dual2nd>(const autodiff::dual2nd*, autodiff::dual2nd x[15], autodiff::dual2nd&);
template void x2b::cart_to_vars_fifth_t<autodiff::var>(const autodiff::var*, autodiff::var x[15], autodiff::var&);

// value_t (fourth/fifth)
template autodiff::real value_fourth_t<autodiff::real>(const autodiff::real*);
template autodiff::real2nd value_fourth_t<autodiff::real2nd>(const autodiff::real2nd*);
template autodiff::dual2nd value_fourth_t<autodiff::dual2nd>(const autodiff::dual2nd*);
template autodiff::var value_fourth_t<autodiff::var>(const autodiff::var*);

template autodiff::real value_fifth_t<autodiff::real>(const autodiff::real*);
template autodiff::real2nd value_fifth_t<autodiff::real2nd>(const autodiff::real2nd*);
template autodiff::dual2nd value_fifth_t<autodiff::dual2nd>(const autodiff::dual2nd*);
template autodiff::var value_fifth_t<autodiff::var>(const autodiff::var*);

} // namespace x2b
////////////////////////////////////////////////////////////////////////////////
