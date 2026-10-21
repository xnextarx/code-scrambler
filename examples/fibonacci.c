#include <stdio.h>

int fibonacci(int n) {
    if (n <= 0) return 0;
    if (n == 1) return 1;

    int a = 0;
    int b = 1;
    int temp;
    int i;

    for (i = 2; i <= n; i++) {
        temp = a + b;
        a = b;
        b = temp;
    }

    return b;
}

int main() {
    int i;
    printf("Fibonacci sequence:\n");
    for (i = 0; i < 10; i++) {
        printf("  fib(%d) = %d\n", i, fibonacci(i));
    }
    return 0;
}
