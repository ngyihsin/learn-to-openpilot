/* Day 16 reference solution — the three functions in C. */
#include <stdint.h>

long long sum_squares(long n) {
    long long sum = 0;
    for (long i = 0; i < n; i++) {
        sum += (long long)i * i;
    }
    return sum;
}

int is_prime(long n) {
    if (n < 2) return 0;
    for (long i = 2; i * i <= n; i++) {
        if (n % i == 0) return 0;
    }
    return 1;
}

long long fib(int n) {
    long long a = 0, b = 1;
    for (int i = 0; i < n; i++) {
        long long next = a + b;
        a = b;
        b = next;
    }
    return a;
}
