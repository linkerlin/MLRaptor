#include "Eigen/Dense"
#include <iostream>
using namespace Eigen;
using namespace std;

typedef Map<ArrayXXd,Aligned> MexMat;
typedef Map<ArrayXd,Aligned> VecMat;
typedef ArrayXXd Mat;
typedef ArrayXd Vec;

void set_seed( int seed );
void select_without_replacement( int N, int K, Vec &chosenIDs);
int discrete_rand( Vec &p );

void init_Mu( MexMat &X, MexMat &Mu, char* initname );
void run_lloyd( MexMat &X, MexMat &Mu, MexMat &Z, int Niter);
