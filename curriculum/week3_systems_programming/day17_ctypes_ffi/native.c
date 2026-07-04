/* Day 17 — native C routines the homework wraps from Python via ctypes.
 * The grader compiles this into a shared library; you write the Python glue in homework.py. */
#include <stddef.h>

/* Sum an array of doubles. */
double array_sum(const double *a, int n) {
    double s = 0.0;
    for (int i = 0; i < n; i++) s += a[i];
    return s;
}

/* Multiply every element by k, IN PLACE (mutates the caller's buffer). */
void scale(double *a, int n, double k) {
    for (int i = 0; i < n; i++) a[i] *= k;
}

/* Dot product of two equal-length arrays. */
double dot(const double *a, const double *b, int n) {
    double s = 0.0;
    for (int i = 0; i < n; i++) s += a[i] * b[i];
    return s;
}
