#include <stdio.h>
#include <stdlib.h>

long compare(long a, long b);
long ccompare(long a, long b);

inline long compare(long a, long b)
{
    long ret;
    if (a < b){
        ret = a/b;
    }
    else{
        ret = b/a;
    }

    return ret;
}

inline long ccompare(long a, long b)
{
    long reta = a/b;
    long retb = b/a;
    long test = reta > retb;
    if (test){
        reta = retb;
    }

    return reta;
}


int main(int argc, char **argv)
{
    compare(1, 2);
    ccompare(4, 5);
    return 0;
}