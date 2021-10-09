#include <iostream>
#include <ctime>

void generateMatrix(double **matrix, int row);
//void printMatrix(double **matrix, int row);
void transpose(double **in, double **out, int row);
void transposeBlock(double **in, double **out, int row);

int main(int agrc, char *agrv[])
{
    int row;
    std::cin >> row;
    double **inMatrix = new double*[row];
    double **outMatrix = new double*[row];
    for(int i=0;i<row;i++) {
        inMatrix[i] = new double[row];
        outMatrix[i] = new double[row];
    }
    generateMatrix(inMatrix,row);

    double startTime = (double)clock()/CLOCKS_PER_SEC*1000;

    transpose(inMatrix,outMatrix,row);
    
    double totalTime = (double)clock()/CLOCKS_PER_SEC*1000 - startTime;
    printf("%f",totalTime);
}

void transpose(double **in, double **out, int row)
{
    for(int i=0;i<row;i++) {
        for(int j=0;j<row;j++) {
            out[j][i] = in[i][j];
        }
    }
}

void transposeBlock(double **in, double **out, int row)
{
    int block = 20;
    for(int i=0;i<row;i+=block) {
        for(int j=0;j<row;j++) {
            for(int k=0;k<block;k++) {
                out[j][i] = in[i+k][j];
            }
        }
    }
}

void generateMatrix(double **matrix, int row)
{
    srand(time(NULL));
    for(int i=0;i<row;i++) {
        for(int j=0;j<row;j++) {
            matrix[i][j] = (double)rand()/ (double)rand();
        }
    }
}

/*
void printMatrix(double **matrix, int row)
{
    for(int i=0;i<row;i++) {
        for(int j=0;j<row;j++) {
            printf("%.8f\t",matrix[i][j]);
        }
        printf("\n");
    }
}
*/