#include <stdio.h>

int main() {
    printf("Hello, ");
    printf("World!\n");

    int x = 10;
    int y = 20;
    int sum = x + y;

    printf("Sum of %d + %d = %d\n", x, y, sum);

    if (sum > 25) {
        printf("The sum is large!\n");
    } else {
        printf("The sum is small.\n");
    }

    return 0;
}
