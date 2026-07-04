/* Day 16 homework — implement these three functions in C.
 *
 * The grader compiles THIS file into a shared library with gcc and calls each function via
 * Python's ctypes, comparing the result to a Python reference. When `pytest -q` is green, your
 * C matches. Peek at solution.c only after you've tried.
 *
 * C reminders for Python programmers:
 *   - every variable has a fixed type (`long`, `long long`, `int`) and you declare it
 *   - `for (long i = 0; i < n; i++) { ... }` is the classic loop
 *   - integer overflow is silent — use `long long` when a sum can get big
 *   - there's no `**`; multiply explicitly
 */
#include <stdint.h>

/* Sum of i*i for i in [0, n).  e.g. sum_squares(4) = 0 + 1 + 4 + 9 = 14 */
long long sum_squares(long n) {
    /* TODO: accumulate i*i in a `long long` and return it */
    return 0;
}

/* Return 1 if n is prime, else 0.  (n < 2 is not prime) */
int is_prime(long n) {
    /* TODO: trial division — check divisors i where i*i <= n */
    return 0;
}

/* nth Fibonacci number, computed iteratively.  fib(0)=0, fib(1)=1, fib(10)=55 */
long long fib(int n) {
    /* TODO: iterate with two running values; do NOT recurse */
    return 0;
}
