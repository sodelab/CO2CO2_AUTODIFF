#include <cmath>
#include <cassert>
#include <cstdlib>
#include <iomanip>
#include <autodiff/forward/real.hpp>
#include <autodiff/forward/real/eigen.hpp>
#include <autodiff/reverse/var.hpp>
#include <autodiff/reverse/var/eigen.hpp>
//#include <autodiff/forward/real2nd.hpp>
//#include <autodiff/forward/real2nd/eigen.hpp>
#include <autodiff/forward/dual.hpp>
//#include <autodiff/forward/dual2nd/eigen.hpp>

#include "kit.h" 
#include "poly-1b.h"
#include "x1b.h"

////////////////////////////////////////////////////////////////////////////////
namespace x1b {

// Original implementation that calls the template version
void cart_to_vars(const double* xyz, double* v) 
{
    cart_to_vars_t<double>(xyz, v);
}

// Template implementation
template<typename T>
void cart_to_vars_t(const T* xyz, T* v) 
{
    const T* C  = xyz;
    const T* O1 = xyz + 3;
    const T* O2 = xyz + 6;

    using kit::distance;

    v[0] = var_intra_t<T>(T(d0_intra_CO), T(m_k_CO_intra), distance(C, O1));
    v[1] = var_intra_t<T>(T(d0_intra_CO), T(m_k_CO_intra), distance(C, O2));
    v[2] = var_intra_t<T>(T(d0_intra_OO), T(m_k_OO_intra), distance(O1, O2));
    

}

// Original value function that calls template version
double value(const double xyz[9])
{
    return value_t<double>(xyz);
}

// Template implementation
template<typename T>
T value_t(const T* xyz)
{
    T v[num_vars];
    cart_to_vars_t<T>(xyz, v);

    T E_poly(0);

    // Use the templated version of eval instead
    T mono[num_linear_params];
    eval_t<T>(v, mono);

    for (size_t n = 0; n < num_linear_params; ++n)
        E_poly += T(poly[n])*mono[n];

    return E_poly;
}

// Compute gradient using autodiff
void gradient(const double xyz[9], double grad[9])
{

    // Create autodiff variables
    autodiff::ArrayXreal x(9);
    for (int i = 0; i < 9; i++) 
        x[i] = xyz[i];
    
    // Define function that takes ArrayXreal and returns real
    auto func = [](const autodiff::ArrayXreal& x) -> autodiff::real {
        autodiff::real xyz_ad[9];
        for (int i = 0; i < 9; i++) xyz_ad[i] = x[i];
        return value_t<autodiff::real>(xyz_ad);
    };
    
    // Compute gradient
    autodiff::ArrayXreal g = autodiff::gradient(func, wrt(x), at(x));
    
    // Copy results to output array
    for (int i = 0; i < 9; i++)
        grad[i] = g[i].val();
}

// Compute Hessian using autodiff
// void hessian(const double xyz[9], double hess[9][9])
// {

//     // Create autodiff variables
//     autodiff::ArrayXreal2nd x(9);
//     for (int i = 0; i < 9; i++)
//         x[i] = xyz[i];
    
//     // Define function that takes ArrayXreal2nd and returns real2nd
//     auto func = [](const autodiff::ArrayXreal2nd& x) -> autodiff::real2nd {
//         autodiff::real2nd xyz_ad[9];
//         for (int i = 0; i < 9; i++) xyz_ad[i] = x[i];
//         return value_t<autodiff::real2nd>(xyz_ad);
//     };
    
//     // Compute Hessian
//     autodiff::MatrixXreal H = autodiff::hessian(func, wrt(x), at(x));
    
//     // Copy results to output array
//     for (int i = 0; i < 9; i++)
//         for (int j = 0; j < 9; j++)
//             hess[i][j] = H(i, j).val();
// }

// Explicit template instantiations
template void cart_to_vars_t<autodiff::real>(const autodiff::real*, autodiff::real*);
template autodiff::real value_t<autodiff::real>(const autodiff::real*);
template autodiff::real2nd value_t<autodiff::real2nd>(const autodiff::real2nd*);

// Explicit instantiations for autodiff::var (reverse mode)
template void cart_to_vars_t<autodiff::var>(const autodiff::var*, autodiff::var*);
template autodiff::var value_t<autodiff::var>(const autodiff::var*);
//template void eval_t<autodiff::var>(const autodiff::var*, autodiff::var*);

// Explicit instantiations for autodiff::var (reverse mode)
template void cart_to_vars_t<autodiff::dual2nd>(const autodiff::dual2nd*, autodiff::dual2nd*);
template autodiff::dual2nd value_t<autodiff::dual2nd>(const autodiff::dual2nd*);


} // namespace x1b

////////////////////////////////////////////////////////////////////////////////