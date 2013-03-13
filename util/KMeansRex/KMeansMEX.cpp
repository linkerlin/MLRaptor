#include "Eigen/Dense"
#include "mex.h"
#include "KMeansRex2.h"
#include <iostream>
using namespace Eigen;
using namespace std;

typedef Map<ArrayXXd,Aligned> MexMat;
typedef Map<ArrayXd,Aligned> VecMat;
typedef ArrayXXd Mat;
typedef ArrayXd Vec;

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) 
{
	if (nrhs != 5 and nrhs != 4) {
		mexErrMsgTxt("Needs 5 args -- Xdata (NxD), K (int), Niter (int), initname (str), seed (int) [OPTIONAL]");
		return;
	}

	double* X_IN = mxGetPr(prhs[0]);	
	int N = mxGetM( prhs[0] );
	int D = mxGetN( prhs[0] );
	
	int K = (int) mxGetScalar(prhs[1]);
	int Niter = (int) mxGetScalar( prhs[2] );
	char* initname = mxArrayToString( prhs[3] );
	
	if (nrhs == 5) {
	  int SEED = (int) mxGetScalar( prhs[4] );
    	set_seed( SEED );
	}
	
	if (K <= 1 ) {
	  mexErrMsgTxt("Need at least two clusters to avoid degeneracy.");
	  return;
	}
	if (K >= N) {
	  mexErrMsgTxt("Cannot request more clusters than data points.");
		return;
	}

	MexMat X (X_IN, N, D);
	
	
  // Create output params!
	plhs[0] = mxCreateDoubleMatrix(K, D, mxREAL);
  MexMat Mu ( mxGetPr(plhs[0]), K, D );
  
	plhs[1] = mxCreateDoubleMatrix(N, 1, mxREAL);
  MexMat Z  ( mxGetPr(plhs[1]), N, 1 );
    
  init_Mu( X, Mu, initname);
  run_lloyd( X, Mu, Z, Niter);
}
