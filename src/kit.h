#ifndef KIT_H
#define KIT_H

#include <cmath>
#include <cassert>
#include <cstddef>
#include <stdexcept>
#include <iomanip>
#include <iostream>
#include <autodiff/forward/real.hpp>
#include <autodiff/forward/dual.hpp> // Add this for dual support
#include <autodiff/reverse/var.hpp>
//#include <autodiff/forward/dual2nd.hpp>

////////////////////////////////////////////////////////////////////////////////

namespace kit {

// Specialization for autodiff::real
// template<>
// struct MathFunctions {
//     static autodiff::real sqrt(const autodiff::real& x) { return sqrt(x); }
//     static autodiff::real exp(const autodiff::real& x) { return exp(x); }
//     static autodiff::real pow(const autodiff::real& x, const autodiff::real& y) { return pow(x, y); }

//     static autodiff::dual sqrt(const autodiff::dual& x) { return sqrt(x); }
//     static autodiff::dual exp(const autodiff::dual& x) { return exp(x); }
//     static autodiff::dual pow(const autodiff::dual& x, const autodiff::dual& y) { return pow(x, y); }

//     static autodiff::real2nd sqrt(const autodiff::real2nd& x) { return sqrt(x); }
//     static autodiff::real2nd exp(const autodiff::real2nd& x) { return exp(x); }
//     static autodiff::real2nd pow(const autodiff::real2nd& x, const autodiff::real2nd& y) { return pow(x, y); }
// };

//Helper to select the right math function
template<typename T>
struct MathFunctions {
    static T sqrt(const T& x) { return std::sqrt(x); }
    static T exp(const T& x) { return std::exp(x); }
    static T pow(const T& x, const T& y) { return std::pow(x, y); }
};

// Specialization for autodiff::real
template<>
struct MathFunctions<autodiff::real> {
    static autodiff::real sqrt(const autodiff::real& x) { return sqrt(x); }
    static autodiff::real exp(const autodiff::real& x) { return exp(x); }
    static autodiff::real pow(const autodiff::real& x, const autodiff::real& y) { return pow(x, y); }
};

// Specialization for autodiff::dual
template<>
struct MathFunctions<autodiff::dual> {
    static autodiff::dual sqrt(const autodiff::dual& x) { return sqrt(x); }
    static autodiff::dual exp(const autodiff::dual& x) { return exp(x); }
    static autodiff::dual pow(const autodiff::dual& x, const autodiff::dual& y) { return pow(x, y); }
};

// Specialization for autodiff::dual
template<>
struct MathFunctions<autodiff::dual2nd> {
    static autodiff::dual2nd sqrt(const autodiff::dual2nd& x) { return sqrt(x); }
    static autodiff::dual2nd exp(const autodiff::dual2nd& x) { return exp(x); }
    static autodiff::dual2nd pow(const autodiff::dual2nd& x, const autodiff::dual2nd& y) { return pow(x, y); }
};

template<>
struct MathFunctions<autodiff::var> {
    static autodiff::var sqrt(const autodiff::var& x) { return sqrt(x); }
    static autodiff::var exp(const autodiff::var& x) { return exp(x); }
    static autodiff::var pow(const autodiff::var& x, const autodiff::var& y) { return pow(x, y); }
};


// Specialization for autodiff::real2nd
template<>
struct MathFunctions<autodiff::real2nd> {
    static autodiff::real2nd sqrt(const autodiff::real2nd& x) { return sqrt(x); }
    static autodiff::real2nd exp(const autodiff::real2nd& x) { return exp(x); }
    static autodiff::real2nd pow(const autodiff::real2nd& x, const autodiff::real2nd& y) { return pow(x, y); }
};

// Template distance function that works with any numeric type
template<typename T>
T distance(const T* a, const T* b)
{
    T dx = a[0] - b[0];
    T dy = a[1] - b[1];
    T dz = a[2] - b[2];

    //return MathFunctions<T>::sqrt(dx*dx + dy*dy + dz*dz);
    return sqrt(dx*dx + dy*dy + dz*dz);
}

// Original distance function for compatibility with existing code
inline double distance(const double* a, const double* b)
{
    return distance<double>(a, b);
}

//----------------------------------------------------------------------------//

// Template version that works with any numeric type
template<typename T>
inline T distance_short_t(const T* xyz)
{
    const T* Ca  = xyz;
    const T* Oa1 = xyz + 3;
    const T* Oa2 = xyz + 6;

    const T* Cb  = xyz + 9;
    const T* Ob1 = xyz + 12;
    const T* Ob2 = xyz + 15;

    T r_small = distance<T>(Ca, Cb);
    if (distance<T>(Ca, Ob1) < r_small) r_small = distance<T>(Ca, Ob1);
    if (distance<T>(Ca, Ob2) < r_small) r_small = distance<T>(Ca, Ob2);
    if (distance<T>(Oa1, Ob1) < r_small) r_small = distance<T>(Oa1, Ob1);
    if (distance<T>(Oa1, Ob2) < r_small) r_small = distance<T>(Oa1, Ob2);
    if (distance<T>(Oa1, Cb) < r_small) r_small = distance<T>(Oa1, Cb);
    if (distance<T>(Oa2, Ob1) < r_small) r_small = distance<T>(Oa2, Ob1);
    if (distance<T>(Oa2, Ob2) < r_small) r_small = distance<T>(Oa2, Ob2);
    if (distance<T>(Oa2, Cb) < r_small) r_small = distance<T>(Oa2, Cb);

    return r_small;
}

// Original function for compatibility with existing code
inline double distance_short(const double* xyz)
{
    return distance_short_t<double>(xyz);
}

//----------------------------------------------------------------------------//

template <int N>
inline int factorial()
{
    return N*factorial<N-1>();
}

template<>
inline int factorial<0>()
{
    return 1;
}

//----------------------------------------------------------------------------//

template<typename T>
inline T tang_toennies_t(const int n, const T& x) {
    T sum = T(1.0);
    T xi = x;
    double fac = 1.0;
    
    for (int i = 1; i <= n; ++i) {
        fac *= i;
        sum += xi/T(fac);
        xi *= x;
    }
    
    return T(1.0) - exp(-x)*sum;
    // return T(1.0) - MathFunctions<T>::exp(-x)*sum;
}

inline double tang_toennies(const int n, const double x) {
    return tang_toennies_t<double>(n, x);
}

template<typename T>
void center_of_mass(const T* const* crd, const double* mass, int n, T* com)
{
    T mtot(0);
    com[0] = com[1] = com[2] = T(0.0);
    
    for (int i = 0; i < n; ++i) {
        const T m = T(mass[i]);
        mtot += m;
        for (int j = 0; j < 3; ++j)
            com[j] += m*crd[i][j];
    }
    
    for (int j = 0; j < 3; ++j)
        com[j] /= mtot;
}

} // namespace kit

////////////////////////////////////////////////////////////////////////////////

#endif // KIT_H
