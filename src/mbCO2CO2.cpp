#include <cmath>
#include <algorithm>
#include <iostream>
#include <stdexcept>

// autodiff core + Eigen support
#include <autodiff/forward/real.hpp>
#include <autodiff/forward/real/eigen.hpp>
#include <autodiff/forward/dual.hpp>
#include <autodiff/forward/dual/eigen.hpp>
#include <autodiff/reverse/var.hpp>
#include <autodiff/reverse/var/eigen.hpp>

#include "x1b.h"
#include "x2b.h"

using autodiff::real;
using autodiff::dual;
using autodiff::var;
using autodiff::ArrayXreal;
using autodiff::VectorXdual;
using autodiff::MatrixXdual;
using autodiff::VectorXvar;

// Physical constants (CODATA 2010)
static constexpr double Eh_J      = 4.35974434e-18;
static constexpr double Na        = 6.02214129e+23;
static constexpr double kcal_J    = 4184.0;
static constexpr double Eh_kcalmol = Eh_J * Na / kcal_J;

// -----------------------------------------------------------------------------
// Required C‐API: getN(), calcSurface()
// -----------------------------------------------------------------------------

#ifdef _WIN32
# define DLLEXPORT __declspec(dllexport)
#else
# define DLLEXPORT
#endif

extern "C" {

// Return number of atoms (6 CO₂ atoms)
DLLEXPORT int getN() { return 6; }

// Evaluate full dimer PES (1b + 2b + SAPT‐S) in cm⁻¹
DLLEXPORT double calcSurface(double *X, int * /*PERM*/) {
    // monomers: 3×3 coords each
    double mon1[9], mon2[9];
    std::copy(X,       X + 9,  mon1);
    std::copy(X + 9,   X + 18, mon2);

    // energy from 1-body terms
    double val = x1b::value(mon1) + x1b::value(mon2);

    // add 2-body 5th-order potential
    val += x2b::value_fifth(X);

    // convert to cm⁻¹
    return val * 349.75;
}

// 1‐body dimension (3 atoms × 3 coords)
DLLEXPORT int get_p1b_dim()   { return  9; }

// 2‐body total dimension
DLLEXPORT int get_p2b_dim()   { return 18; }

// 2‐body 4th‐order (same as total)
DLLEXPORT int get_p2b_4_dim() { return 18; }

// 2‐body 5th‐order (same as total)
DLLEXPORT int get_p2b_5_dim() { return 18; }

// SAPT‐S term dimension
DLLEXPORT int get_sapt_dim()  { return 18; }

// Library version
DLLEXPORT const char* get_version() {
    return "CO2CO2_AUTODIFF 0.4.4 (2026-02-26)";
}

} // extern "C"

// -----------------------------------------------------------------------------
// Simple energy‐only calls
// -----------------------------------------------------------------------------

extern "C" {

// 2-body, 4th‐order potential in kcal/mol
DLLEXPORT double p2b_4(double *X) {
    return x2b::value_fourth(X);
}

// 2-body, 5th‐order potential in kcal/mol
DLLEXPORT double p2b_5(double *X) {
    return x2b::value_fifth(X);
}

// 1-body potential in kcal/mol
DLLEXPORT double p1b(double *X) {
    return x1b::value(X);
}

// SAPT-S term in kcal/mol
DLLEXPORT double sapt(double *X) {
    return x2b::sapt_s(X);
}

} // extern "C"

// -----------------------------------------------------------------------------
// Forward‐mode gradients (dual) C‐API
// -----------------------------------------------------------------------------

extern "C" {

// 1b gradient (9 → 9)
DLLEXPORT void p1b_gradient(double *X, double *grad) {
    auto f = [](const ArrayXreal& x) -> real {
        real c[9]; for(int i=0;i<9;++i) c[i]=x[i];
        return x1b::value_t(c);
    };
    ArrayXreal x(9); for(int i=0;i<9;++i) x[i]=X[i];
    ArrayXreal g = autodiff::gradient(f, wrt(x), autodiff::at(x));
    for(int i=0;i<9;++i) grad[i] = g[i].val();
}

// 2b 4th‐order gradient (18 → 18)
DLLEXPORT void p2b_gradient_4(double *X, double *grad) {
    auto f = [](const ArrayXreal& x) -> real {
        real c[18]; for(int i=0;i<18;++i) c[i]=x[i];
        return x2b::value_fourth_t(c);
    };
    ArrayXreal x(18); for(int i=0;i<18;++i) x[i]=X[i];
    ArrayXreal g = autodiff::gradient(f, wrt(x), autodiff::at(x));
    for(int i=0;i<18;++i) grad[i] = g[i].val();
}

// 2b 5th‐order gradient (18 → 18)
DLLEXPORT void p2b_gradient_5(double *X, double *grad) {
    auto f = [](const ArrayXreal& x) -> real {
        real c[18]; for(int i=0;i<18;++i) c[i]=x[i];
        return x2b::value_fifth_t(c);
    };
    ArrayXreal x(18); for(int i=0;i<18;++i) x[i]=X[i];
    ArrayXreal g = autodiff::gradient(f, wrt(x), autodiff::at(x));
    for(int i=0;i<18;++i) grad[i] = g[i].val();
}

// SAPT gradient (18 → 18)
DLLEXPORT void sapt_gradient(double *X, double *grad) {
    auto f = [](const ArrayXreal& x) -> real {
        real c[18]; for(int i=0;i<18;++i) c[i]=x[i];
        return x2b::sapt_s_t(c);
    };
    ArrayXreal x(18); for(int i=0;i<18;++i) x[i]=X[i];
    ArrayXreal g = autodiff::gradient(f, wrt(x), autodiff::at(x));
    for(int i=0;i<18;++i) grad[i] = g[i].val();
}

} // extern "C"

// -----------------------------------------------------------------------------
// Reverse‐mode Hessians (var) C‐API
// -----------------------------------------------------------------------------

extern "C" {

// p1b Hessian via reverse‐mode tape
DLLEXPORT void p1b_hessian_rev(double *X, double *hess) {
    VectorXvar x(9); for(int i=0;i<9;++i) x[i]=X[i];
    var u = [&](const VectorXvar& v)->var {
        var c[9]; for(int i=0;i<9;++i) c[i]=v[i];
        return x1b::value_t(c);
    }(x);
    Eigen::VectorXd g; Eigen::MatrixXd H = autodiff::hessian(u, x, g);
    for(int i=0;i<9;++i) for(int j=0;j<9;++j) hess[i*9+j]=H(i,j);
}

// p2b 4th‐order Hessian via reverse‐mode
DLLEXPORT void p2b_hessian_4_rev(double *X, double *hess) {
    VectorXvar x(18); for(int i=0;i<18;++i) x[i]=X[i];
    var u = [&](auto const& v)->var {
        var c[18]; for(int i=0;i<18;++i) c[i]=v[i];
        return x2b::value_fourth_t(c);
    }(x);
    Eigen::VectorXd g; Eigen::MatrixXd H = autodiff::hessian(u, x, g);
    for(int i=0;i<18;++i) for(int j=0;j<18;++j) hess[i*18+j]=H(i,j);
}

// p2b 5th‐order Hessian via reverse‐mode
DLLEXPORT void p2b_hessian_5_rev(double *X, double *hess) {
    VectorXvar x(18); for(int i=0;i<18;++i) x[i]=X[i];
    var u = [&](auto const& v)->var {
        var c[18]; for(int i=0;i<18;++i) c[i]=v[i];
        return x2b::value_fifth_t(c);
    }(x);
    Eigen::VectorXd g; Eigen::MatrixXd H = autodiff::hessian(u, x, g);
    for(int i=0;i<18;++i) for(int j=0;j<18;++j) {
        hess[i*18+j] = H(i,j);
        hess[j*18+i] = H(i,j);
    }
}

// SAPT Hessian via reverse‐mode
DLLEXPORT void sapt_hessian_rev(double *X, double *hess) {
    VectorXvar x(18); for(int i=0;i<18;++i) x[i]=X[i];
    var u = [&](auto const& v)->var {
        var c[18]; for(int i=0;i<18;++i) c[i]=v[i];
        return x2b::sapt_s_t(c);
    }(x);
    Eigen::VectorXd g; Eigen::MatrixXd H = autodiff::hessian(u, x, g);
    for(int i=0;i<18;++i) for(int j=0;j<18;++j) hess[i*18+j]=H(i,j);
}

} // extern "C"

// -----------------------------------------------------------------------------
// Forward‐mode Hessians (dual2nd) C‐API
// -----------------------------------------------------------------------------

extern "C" {

// p1b Hessian via forward‐mode dual2nd
DLLEXPORT void p1b_hessian_fwd(double *X, double *hess) {
    Eigen::VectorXdual2nd x(9); for(int i=0;i<9;++i) x[i]=X[i];
    auto f = [&](auto const& v)->Eigen::dual2nd {
        Eigen::dual2nd c[9]; for(int i=0;i<9;++i) c[i]=v[i];
        return x1b::value_t(c);
    };
    Eigen::dual2nd u; VectorXdual g; 
    Eigen::MatrixXd H = autodiff::hessian(f, wrt(x), autodiff::at(x), u, g);
    for(int i=0;i<9;++i) for(int j=0;j<9;++j) hess[i*9+j]=H(i,j);
}

// p2b 4th‐order Hessian via forward‐mode dual2nd
DLLEXPORT void p2b_hessian_4_fwd(double *X, double *hess) {
    Eigen::VectorXdual2nd x(18); for(int i=0;i<18;++i) x[i]=X[i];
    auto f = [&](auto const& v)->Eigen::dual2nd {
        Eigen::dual2nd c[18]; for(int i=0;i<18;++i) c[i]=v[i];
        return x2b::value_fourth_t(c);
    };
    Eigen::dual2nd u; VectorXdual g; 
    Eigen::MatrixXd H = autodiff::hessian(f, wrt(x), autodiff::at(x), u, g);
    for(int i=0;i<18;++i) for(int j=0;j<18;++j) hess[i*18+j]=H(i,j);
}

// p2b 5th‐order Hessian via forward‐mode dual2nd
DLLEXPORT void p2b_hessian_5_fwd(double *X, double *hess) {
    Eigen::VectorXdual2nd x(18); for(int i=0;i<18;++i) x[i]=X[i];
    auto f = [&](auto const& v)->Eigen::dual2nd {
        Eigen::dual2nd c[18]; for(int i=0;i<18;++i) c[i]=v[i];
        return x2b::value_fifth_t(c);
    };
    Eigen::dual2nd u; VectorXdual g; 
    Eigen::MatrixXd H = autodiff::hessian(f, wrt(x), autodiff::at(x), u, g);
    for(int i=0;i<18;++i) for(int j=0;j<18;++j) hess[i*18+j]=H(i,j);
}

// SAPT Hessian via forward‐mode dual2nd
DLLEXPORT void sapt_hessian_fwd(double *X, double *hess) {
    Eigen::VectorXdual2nd x(18); for(int i=0;i<18;++i) x[i]=X[i];
    auto f = [&](auto const& v)->Eigen::dual2nd {
        Eigen::dual2nd c[18]; for(int i=0;i<18;++i) c[i]=v[i];
        return x2b::sapt_s_t(c);
    };
    Eigen::dual2nd u; VectorXdual g; 
    Eigen::MatrixXd H = autodiff::hessian(f, wrt(x), autodiff::at(x), u, g);
    for(int i=0;i<18;++i) for(int j=0;j<18;++j) hess[i*18+j]=H(i,j);
}

} // extern "C"
