#include <cmath>
#include <cassert>
#include <cstdlib>
#include <iostream>
#include <vector>
#include <memory>
#include <autodiff/forward/real.hpp>
#include <autodiff/forward/dual.hpp>

// #include <autodiff/forward/real.hpp>
#include <autodiff/forward/real/eigen.hpp>
#include <autodiff/reverse/var.hpp>
#include <autodiff/reverse/var/eigen.hpp>

#include "x2b.h"
#include "kit.h"

////////////////////////////////////////////////////////////////////////////////

namespace x2b {

//-----------------------------------------------------------------------------
// 1) Site‐point construction
//    Ba = Ca + (Oa–Ca)/|Oa–Ca| * msite
//-----------------------------------------------------------------------------
template<typename T>
inline void site_t(const T* Ca, const T* Oa, T *Ba)
{
    // vector from C to O
    T dx = Oa[0] - Ca[0];
    T dy = Oa[1] - Ca[1];
    T dz = Oa[2] - Ca[2];
    T rsq = dx*dx + dy*dy + dz*dz;
    T r   = sqrt(rsq);

    // unit vector
    T ux = dx / r;
    T uy = dy / r;
    T uz = dz / r;

    // shift length
    const T msite = T(0.8456);

    Ba[0] = Ca[0] + msite * ux;
    Ba[1] = Ca[1] + msite * uy;
    Ba[2] = Ca[2] + msite * uz;
}

inline void site(const double* Ca, const double* Oa, double *Ba)
{
    site_t<double>(Ca, Oa, Ba);
}


//-----------------------------------------------------------------------------
// 2) Exponential‐damping term: exp(α – β·r)
//-----------------------------------------------------------------------------
template<typename T>
inline T exponential_t(const T* A, const T* B, const T &alpha, const T &beta)
{
    T dx  = A[0] - B[0];
    T dy  = A[1] - B[1];
    T dz  = A[2] - B[2];
    T rsq = dx*dx + dy*dy + dz*dz;
    T r   = sqrt(rsq);
    return exp(alpha - beta * r);
}

inline double exponential(const double* A, const double* B,
                          const double &alpha, const double &beta)
{
    return exponential_t<double>(A, B, alpha, beta);
}


//-----------------------------------------------------------------------------
// 3) Charge–charge term: qA·qB·tang‐Toennies(1,δr)/r
//-----------------------------------------------------------------------------
template<typename T>
inline T qq_t(const T* A, const T* B,
              const T &delta, const T &qA, const T &qB)
{
    T dx  = A[0] - B[0];
    T dy  = A[1] - B[1];
    T dz  = A[2] - B[2];
    T rsq = dx*dx + dy*dy + dz*dz;
    T r   = sqrt(rsq);
    T tt  = kit::tang_toennies_t<T>(1, delta * r);
    return tt * qA * qB / r;
}

inline double qq(const double* A, const double* B,
                 const double &delta, const double &qA, const double &qB)
{
    return qq_t<double>(A, B, delta, qA, qB);
}


//-----------------------------------------------------------------------------
// 4) C₆ and C₈ dispersion with damping
//    C6 term:   C6·t6(r)/r⁶
//    C8 term:   C8·t8(r)/r⁸
//-----------------------------------------------------------------------------
template<typename T>
inline T xC6_t(const T* A, const T* B,
               const T &delta, const T &C6)
{
    T dx  = A[0] - B[0];
    T dy  = A[1] - B[1];
    T dz  = A[2] - B[2];
    T rsq = dx*dx + dy*dy + dz*dz;
    T r   = sqrt(rsq);
    T tt  = kit::tang_toennies_t<T>(6, delta * r);
    return tt * C6 / pow(r, T(6));
}

inline double xC6(const double* A, const double* B,
                  const double &delta, const double &C6)
{
    return xC6_t<double>(A, B, delta, C6);
}

template<typename T>
inline T xC8_t(const T* A, const T* B,
               const T &delta, const T &C8)
{
    T dx  = A[0] - B[0];
    T dy  = A[1] - B[1];
    T dz  = A[2] - B[2];
    T rsq = dx*dx + dy*dy + dz*dz;
    T r   = sqrt(rsq);
    T tt  = kit::tang_toennies_t<T>(8, delta * r);
    return tt * C8 / pow(r, T(8));
}

inline double xC8(const double* A, const double* B,
                  const double &delta, const double &C8)
{
    return xC8_t<double>(A, B, delta, C8);
}


//-----------------------------------------------------------------------------
// 5) Full SAPT‐S dimer energy
//    Combines exponential, charge–charge, C6, C8 terms over all site pairs
//-----------------------------------------------------------------------------
template<typename T>
T sapt_s_t(const T* crd)
{
    // Convert constants to type T
    const T alpha_OO = T(1.1210441e1);
    const T alpha_OC = T(1.1333682e1);
    const T alpha_CC = T(1.1399839e1);

    const T beta_OO = T(4.0202795);
    const T beta_OC = T(4.5074401);
    const T beta_CC = T(5.0932632);
    
    const T qO = T(2.3786535e-1*18.22262373);
    const T qC = T(1.6316722*18.22262373);
    const T qM = T(-1.0537015*18.22262373);
    
    const T delta_1_OO = T(1.4968485);
    const T delta_1_OC = T(1.8797629);
    const T delta_1_CC = T(2.1958809);
    const T delta_1_OM = T(1.9648279);
    const T delta_1_CM = T(2.6032461);
    const T delta_1_MM = T(5.2350982);
    
    const T delta_6_OO = T(2.5924278);
    const T delta_6_OC = T(1.8139783);
    const T delta_6_CC = T(1.7584847);
    
    const T delta_8_OO = T(1.0769328);
    const T delta_8_OC = T(6.7472291e-1);
    const T delta_8_CC = T(3.0176726);
    
    const T C6_OO = T(1.0426642e3);
    const T C6_OC = T(-1.3834479e3);
    const T C6_CC = T(3.4808543e3);
    
    const T C8_OO = T(-1.3516797e4);
    const T C8_OC = T(2.0217414e4);
    const T C8_CC = T(-2.6899552e4);
    
    // atom coords
    const T *Ca   = crd +  0, *Oa1 = crd +  3, *Oa2 = crd +  6;
    const T *Cb   = crd +  9, *Ob1 = crd + 12, *Ob2 = crd + 15;

    // build pseudo‐sites
    T Ba1[3], Ba2[3], Bb1[3], Bb2[3];
    site_t<T>(Ca, Oa1, Ba1);
    site_t<T>(Ca, Oa2, Ba2);
    site_t<T>(Cb, Ob1, Bb1);
    site_t<T>(Cb, Ob2, Bb2);

    // exp
    const T OO_exp =
            exponential_t<T>(Oa1, Ob1, alpha_OO, beta_OO)
          + exponential_t<T>(Oa1, Ob2, alpha_OO, beta_OO)
          + exponential_t<T>(Oa2, Ob1, alpha_OO, beta_OO)
          + exponential_t<T>(Oa2, Ob2, alpha_OO, beta_OO);
    
    const T OC_exp =
            exponential_t<T>(Oa1, Cb, alpha_OC, beta_OC)
          + exponential_t<T>(Oa2, Cb, alpha_OC, beta_OC)
          + exponential_t<T>(Ob1, Ca, alpha_OC, beta_OC)
          + exponential_t<T>(Ob2, Ca, alpha_OC, beta_OC);

    const T CC_exp =
            exponential_t<T>(Ca, Cb, alpha_CC, beta_CC);
    
    // charge-charge
    const T OO_QQ =
            qq_t<T>(Oa1, Ob1, delta_1_OO, qO, qO)
          + qq_t<T>(Oa1, Ob2, delta_1_OO, qO, qO)
          + qq_t<T>(Oa2, Ob1, delta_1_OO, qO, qO)
          + qq_t<T>(Oa2, Ob2, delta_1_OO, qO, qO);
    
    const T OC_QQ =
            qq_t<T>(Oa1, Cb, delta_1_OC, qO, qC)
          + qq_t<T>(Oa2, Cb, delta_1_OC, qO, qC)
          + qq_t<T>(Ob1, Ca, delta_1_OC, qO, qC)
          + qq_t<T>(Ob2, Ca, delta_1_OC, qO, qC);
    
    const T CC_QQ =
            qq_t<T>(Ca, Cb, delta_1_CC, qC, qC);
    
    const T OM_QQ =
            qq_t<T>(Oa1, Bb1, delta_1_OM, qO, qM)
          + qq_t<T>(Oa1, Bb2, delta_1_OM, qO, qM)
          + qq_t<T>(Oa2, Bb1, delta_1_OM, qO, qM)
          + qq_t<T>(Oa2, Bb2, delta_1_OM, qO, qM)
          + qq_t<T>(Ob1, Ba1, delta_1_OM, qO, qM)
          + qq_t<T>(Ob1, Ba2, delta_1_OM, qO, qM)
          + qq_t<T>(Ob2, Ba1, delta_1_OM, qO, qM)
          + qq_t<T>(Ob2, Ba2, delta_1_OM, qO, qM);

    const T CM_QQ =
            qq_t<T>(Ca, Bb1, delta_1_CM, qC, qM)
          + qq_t<T>(Ca, Bb2, delta_1_CM, qC, qM)
          + qq_t<T>(Cb, Ba1, delta_1_CM, qC, qM)
          + qq_t<T>(Cb, Ba2, delta_1_CM, qC, qM);

    const T MM_QQ =
            qq_t<T>(Ba1, Bb1, delta_1_MM, qM, qM)
          + qq_t<T>(Ba1, Bb2, delta_1_MM, qM, qM)
          + qq_t<T>(Ba2, Bb1, delta_1_MM, qM, qM)
          + qq_t<T>(Ba2, Bb2, delta_1_MM, qM, qM);
    
    // C6
    const T OO_C6 =
            xC6_t<T>(Oa1, Ob1, delta_6_OO, C6_OO)
          + xC6_t<T>(Oa1, Ob2, delta_6_OO, C6_OO)
          + xC6_t<T>(Oa2, Ob1, delta_6_OO, C6_OO)
          + xC6_t<T>(Oa2, Ob2, delta_6_OO, C6_OO);
    
    const T OC_C6 =
            xC6_t<T>(Oa1, Cb, delta_6_OC, C6_OC)
          + xC6_t<T>(Oa2, Cb, delta_6_OC, C6_OC)
          + xC6_t<T>(Ob1, Ca, delta_6_OC, C6_OC)
          + xC6_t<T>(Ob2, Ca, delta_6_OC, C6_OC);
    
    const T CC_C6 =
            xC6_t<T>(Ca, Cb, delta_6_CC, C6_CC);
    
    // C8
    const T OO_C8 =
            xC8_t<T>(Oa1, Ob1, delta_8_OO, C8_OO)
          + xC8_t<T>(Oa1, Ob2, delta_8_OO, C8_OO)
          + xC8_t<T>(Oa2, Ob1, delta_8_OO, C8_OO)
          + xC8_t<T>(Oa2, Ob2, delta_8_OO, C8_OO);
    
    const T OC_C8 =
            xC8_t<T>(Oa1, Cb, delta_8_OC, C8_OC)
          + xC8_t<T>(Oa2, Cb, delta_8_OC, C8_OC)
          + xC8_t<T>(Ob1, Ca, delta_8_OC, C8_OC)
          + xC8_t<T>(Ob2, Ca, delta_8_OC, C8_OC);
    
    const T CC_C8 =
            xC8_t<T>(Ca, Cb, delta_8_CC, C8_CC);

    return (OO_exp + OC_exp + CC_exp
          + OO_QQ  + OC_QQ  + CC_QQ
          + OM_QQ  + CM_QQ  + MM_QQ
          - OO_C6  - OC_C6  - CC_C6
          - OO_C8  - OC_C8  - CC_C8);
}

double sapt_s(const double* crd)
{
    return sapt_s_t<double>(crd);
}

// // Template implementation of sapt_s
// template<typename T>
// T sapt_s_t(const T* crd) 
// {


//     // Convert constants to type T
//     const T alpha_OO = T(1.1210441e1);
//     const T alpha_OC = T(1.1333682e1);
//     const T alpha_CC = T(1.1399839e1);

//     const T beta_OO = T(4.0202795);
//     const T beta_OC = T(4.5074401);
//     const T beta_CC = T(5.0932632);
    
//     const T qO = T(2.3786535e-1*18.22262373);
//     const T qC = T(1.6316722*18.22262373);
//     const T qM = T(-1.0537015*18.22262373);
    
//     const T delta_1_OO = T(1.4968485);
//     const T delta_1_OC = T(1.8797629);
//     const T delta_1_CC = T(2.1958809);
//     const T delta_1_OM = T(1.9648279);
//     const T delta_1_CM = T(2.6032461);
//     const T delta_1_MM = T(5.2350982);
    
//     const T delta_6_OO = T(2.5924278);
//     const T delta_6_OC = T(1.8139783);
//     const T delta_6_CC = T(1.7584847);
    
//     const T delta_8_OO = T(1.0769328);
//     const T delta_8_OC = T(6.7472291e-1);
//     const T delta_8_CC = T(3.0176726);
    
//     const T C6_OO = T(1.0426642e3);
//     const T C6_OC = T(-1.3834479e3);
//     const T C6_CC = T(3.4808543e3);
    
//     const T C8_OO = T(-1.3516797e4);
//     const T C8_OC = T(2.0217414e4);
//     const T C8_CC = T(-2.6899552e4);
    
//     const T* Ca  = crd + 0;
//     const T* Oa1 = crd + 3;
//     const T* Oa2 = crd + 6;

//     const T* Cb  = crd + 9;
//     const T* Ob1 = crd + 12;
//     const T* Ob2 = crd + 15;

//     // Use stack-allocated arrays instead of heap for autodiff compatibility
//     T Ba1[3], Ba2[3], Bb1[3], Bb2[3];

//     site_t<T>(Ca, Oa1, Ba1);
//     site_t<T>(Ca, Oa2, Ba2);
//     site_t<T>(Cb, Ob1, Bb1);
//     site_t<T>(Cb, Ob2, Bb2);


//     // exp
//     const T OO_exp =
//             exponential_t<T>(Oa1, Ob1, alpha_OO, beta_OO)
//           + exponential_t<T>(Oa1, Ob2, alpha_OO, beta_OO)
//           + exponential_t<T>(Oa2, Ob1, alpha_OO, beta_OO)
//           + exponential_t<T>(Oa2, Ob2, alpha_OO, beta_OO);
    
//     const T OC_exp =
//             exponential_t<T>(Oa1, Cb, alpha_OC, beta_OC)
//           + exponential_t<T>(Oa2, Cb, alpha_OC, beta_OC)
//           + exponential_t<T>(Ob1, Ca, alpha_OC, beta_OC)
//           + exponential_t<T>(Ob2, Ca, alpha_OC, beta_OC);

//     const T CC_exp =
//             exponential_t<T>(Ca, Cb, alpha_CC, beta_CC);
    
//     // charge-charge
//     const T OO_QQ =
//             qq_t<T>(Oa1, Ob1, delta_1_OO, qO, qO)
//           + qq_t<T>(Oa1, Ob2, delta_1_OO, qO, qO)
//           + qq_t<T>(Oa2, Ob1, delta_1_OO, qO, qO)
//           + qq_t<T>(Oa2, Ob2, delta_1_OO, qO, qO);
    
//     const T OC_QQ =
//             qq_t<T>(Oa1, Cb, delta_1_OC, qO, qC)
//           + qq_t<T>(Oa2, Cb, delta_1_OC, qO, qC)
//           + qq_t<T>(Ob1, Ca, delta_1_OC, qO, qC)
//           + qq_t<T>(Ob2, Ca, delta_1_OC, qO, qC);
    
//     const T CC_QQ =
//             qq_t<T>(Ca, Cb, delta_1_CC, qC, qC);
    
//     const T OM_QQ =
//             qq_t<T>(Oa1, Bb1, delta_1_OM, qO, qM)
//           + qq_t<T>(Oa1, Bb2, delta_1_OM, qO, qM)
//           + qq_t<T>(Oa2, Bb1, delta_1_OM, qO, qM)
//           + qq_t<T>(Oa2, Bb2, delta_1_OM, qO, qM)
//           + qq_t<T>(Ob1, Ba1, delta_1_OM, qO, qM)
//           + qq_t<T>(Ob1, Ba2, delta_1_OM, qO, qM)
//           + qq_t<T>(Ob2, Ba1, delta_1_OM, qO, qM)
//           + qq_t<T>(Ob2, Ba2, delta_1_OM, qO, qM);

//     const T CM_QQ =
//             qq_t<T>(Ca, Bb1, delta_1_CM, qC, qM)
//           + qq_t<T>(Ca, Bb2, delta_1_CM, qC, qM)
//           + qq_t<T>(Cb, Ba1, delta_1_CM, qC, qM)
//           + qq_t<T>(Cb, Ba2, delta_1_CM, qC, qM);

//     const T MM_QQ =
//             qq_t<T>(Ba1, Bb1, delta_1_MM, qM, qM)
//           + qq_t<T>(Ba1, Bb2, delta_1_MM, qM, qM)
//           + qq_t<T>(Ba2, Bb1, delta_1_MM, qM, qM)
//           + qq_t<T>(Ba2, Bb2, delta_1_MM, qM, qM);
    
//     // C6
//     const T OO_C6 =
//             xC6_t<T>(Oa1, Ob1, delta_6_OO, C6_OO)
//           + xC6_t<T>(Oa1, Ob2, delta_6_OO, C6_OO)
//           + xC6_t<T>(Oa2, Ob1, delta_6_OO, C6_OO)
//           + xC6_t<T>(Oa2, Ob2, delta_6_OO, C6_OO);
    
//     const T OC_C6 =
//             xC6_t<T>(Oa1, Cb, delta_6_OC, C6_OC)
//           + xC6_t<T>(Oa2, Cb, delta_6_OC, C6_OC)
//           + xC6_t<T>(Ob1, Ca, delta_6_OC, C6_OC)
//           + xC6_t<T>(Ob2, Ca, delta_6_OC, C6_OC);
    
//     const T CC_C6 =
//             xC6_t<T>(Ca, Cb, delta_6_CC, C6_CC);
    
//     // C8
//     const T OO_C8 =
//             xC8_t<T>(Oa1, Ob1, delta_8_OO, C8_OO)
//           + xC8_t<T>(Oa1, Ob2, delta_8_OO, C8_OO)
//           + xC8_t<T>(Oa2, Ob1, delta_8_OO, C8_OO)
//           + xC8_t<T>(Oa2, Ob2, delta_8_OO, C8_OO);
    
//     const T OC_C8 =
//             xC8_t<T>(Oa1, Cb, delta_8_OC, C8_OC)
//           + xC8_t<T>(Oa2, Cb, delta_8_OC, C8_OC)
//           + xC8_t<T>(Ob1, Ca, delta_8_OC, C8_OC)
//           + xC8_t<T>(Ob2, Ca, delta_8_OC, C8_OC);
    
//     const T CC_C8 =
//             xC8_t<T>(Ca, Cb, delta_8_CC, C8_CC);

//     return (OO_exp + OC_exp + CC_exp
//           + OO_QQ  + OC_QQ  + CC_QQ
//           + OM_QQ  + CM_QQ  + MM_QQ
//           - OO_C6  - OC_C6  - CC_C6
//           - OO_C8  - OC_C8  - CC_C8);
// }

// // Original function for backward compatibility
// double sapt_s(const double* crd) 
// {
//     return sapt_s_t<double>(crd);
// }

//-----------------------------------------------------------------------------
// Explicit template instantiations
//-----------------------------------------------------------------------------
template void    x2b::site_t<autodiff::real>    (const autodiff::real*   , const autodiff::real*   , autodiff::real*   );
template void    x2b::site_t<autodiff::dual2nd> (const autodiff::dual2nd*, const autodiff::dual2nd*, autodiff::dual2nd*);
template void    x2b::site_t<autodiff::var>     (const autodiff::var*    , const autodiff::var*    , autodiff::var*    );

template autodiff::real     x2b::exponential_t<autodiff::real>    (const autodiff::real*   , const autodiff::real*   , const autodiff::real& , const autodiff::real& );
template autodiff::dual2nd  x2b::exponential_t<autodiff::dual2nd> (const autodiff::dual2nd*, const autodiff::dual2nd*, const autodiff::dual2nd&, const autodiff::dual2nd&);
template autodiff::var      x2b::exponential_t<autodiff::var>     (const autodiff::var*    , const autodiff::var*    , const autodiff::var&   , const autodiff::var&   );

template autodiff::real     x2b::qq_t<autodiff::real>    (const autodiff::real*,    const autodiff::real*,    const autodiff::real& , const autodiff::real&, const autodiff::real&);
template autodiff::dual2nd  x2b::qq_t<autodiff::dual2nd> (const autodiff::dual2nd*, const autodiff::dual2nd*, const autodiff::dual2nd&, const autodiff::dual2nd&, const autodiff::dual2nd&);
template autodiff::var      x2b::qq_t<autodiff::var>     (const autodiff::var*,     const autodiff::var*,     const autodiff::var&   , const autodiff::var&   , const autodiff::var&   );

template autodiff::real     x2b::xC6_t<autodiff::real>    (const autodiff::real*,    const autodiff::real*,    const autodiff::real& , const autodiff::real&);
template autodiff::dual2nd  x2b::xC6_t<autodiff::dual2nd> (const autodiff::dual2nd*, const autodiff::dual2nd*, const autodiff::dual2nd&, const autodiff::dual2nd&);
template autodiff::var      x2b::xC6_t<autodiff::var>     (const autodiff::var*,     const autodiff::var*,     const autodiff::var&   , const autodiff::var&   );

template autodiff::real     x2b::xC8_t<autodiff::real>    (const autodiff::real*,    const autodiff::real*,    const autodiff::real& , const autodiff::real&);
template autodiff::dual2nd  x2b::xC8_t<autodiff::dual2nd> (const autodiff::dual2nd*, const autodiff::dual2nd*, const autodiff::dual2nd&, const autodiff::dual2nd&);
template autodiff::var      x2b::xC8_t<autodiff::var>     (const autodiff::var*,     const autodiff::var*,     const autodiff::var&   , const autodiff::var&   );

template autodiff::real     x2b::sapt_s_t<autodiff::real>    (const autodiff::real*);
template autodiff::real2nd  x2b::sapt_s_t<autodiff::real2nd>(const autodiff::real2nd*);
template autodiff::dual2nd  x2b::sapt_s_t<autodiff::dual2nd>(const autodiff::dual2nd*);
template autodiff::var      x2b::sapt_s_t<autodiff::var>    (const autodiff::var*);

// force an out‐of‐line double sapt_s()
template double sapt_s_t<double>(const double*);

} // namespace x2b

