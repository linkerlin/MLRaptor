// $ LD_PRELOAD=/usr/lib/libstdc++.so.6 matlab &
#include "Eigen/Dense"
#include "mersenneTwister2002.c"
#include <iostream>
#include <time.h>

using namespace Eigen;
using namespace std;

typedef Map<ArrayXXd,Aligned> MexMat;
typedef Map<ArrayXd,Aligned> VecMat;
typedef ArrayXXd Mat;
typedef ArrayXd Vec;

#ifdef EIGEN_VECTORIZE
const int doVEC = 1;
#else
const int doVEC = 0;
#endif

#ifdef EIGEN_NO_DEBUG
const int doDEBUG= 1;
#else
const int doDEBUG= 0;
#endif

double get_elapsed_time( clock_t cSTART, clock_t cEND) {
  return (double) ( cEND - cSTART )/CLOCKS_PER_SEC;   
}

void set_seed( int seed ) {
  init_genrand( seed );
}

int discrete_rand( Vec &p ) {
    double total = p.sum();
    int K = (int) p.size();
    
    double r = total*genrand_double();
    double cursum = p(0);
    int newk = 0;
    while ( r >= cursum && newk < K-1) {
        newk++;
        cursum += p[newk];
    }
    if ( newk < 0 || newk >= K ) {
        cerr << "   somethings not right here." << endl;
        return -1;
    }
    return newk;
}

void select_without_replacement( int N, int K, Vec &chosenIDs) {
    Vec p = Vec::Ones(N);
    for (int kk =0; kk<K; kk++) {
      int choice;
      int doKeep = false;
      while ( doKeep==false) {
      
        doKeep=true;
        choice = discrete_rand( p );
      
        for (int previd=0; previd<kk; previd++) {
          if (chosenIDs[previd] == choice ) {
            doKeep = false;
            break;
          }
        }      
      }      
      chosenIDs[kk] = choice;     
    }
}

void pairwise_distance( MexMat &X, MexMat &Mu, Mat &Dist ) {
  int N = X.rows();
  int D = X.cols();
  int K = Mu.rows();

  if ( D <= 16 ) {
    for (int kk=0; kk<K; kk++) {
      Dist.col(kk) = ( X.rowwise() - Mu.row(kk) ).square().rowwise().sum();
    }    
  } else {
    //clock_t starttime = clock();
    Dist = -2*( X.matrix() * Mu.transpose().matrix() );
    //cout << "X*Mu | " << get_elapsed_time( starttime, clock() ) << endl;
  
    //starttime = clock();
    Dist.rowwise() += Mu.square().rowwise().sum().transpose().row(0);
    //cout << "Mu*Mu | " << get_elapsed_time( starttime, clock() ) << endl;
  }
}

void init_Mu( MexMat &X, MexMat &Mu, char* initname ) {
	
	int N = X.rows();
	int K = Mu.rows();
	  
	if ( string( initname ) == "random" ) {
		
		Vec ChosenIDs = Vec::Zero(K);
                select_without_replacement( N, K, ChosenIDs );
		
		for (int kk=0; kk<K; kk++) {
		  Mu.row( kk ) = X.row( ChosenIDs[kk] );
		}		
		
	} else if ( string( initname ) == "plusplus" ) {
    Vec ChosenIDs = Vec::Ones(K);
    int choice = discrete_rand( ChosenIDs );
    Mu.row(0) = X.row( choice );
    ChosenIDs[0] = choice;
    Vec minDist(N);
    Vec curDist(N);
    for (int kk=1; kk<K; kk++) {
      curDist = ( X.rowwise() - Mu.row(kk-1) ).square().rowwise().sum().sqrt();
      if (kk==1) {
        minDist = curDist;
      } else {
        minDist = curDist.min( minDist );
      }      
      choice = discrete_rand( minDist );
      
      ChosenIDs[kk] = choice;
      Mu.row(kk) = X.row( choice );
    }       
	}
}

void calc_Mu( MexMat &X, MexMat &Mu, MexMat &Z) {
  clock_t starttime = clock();
  Mu = Mat::Zero( Mu.rows(), Mu.cols() );
  Vec NperCluster = Vec::Zero( Mu.rows() );
  
  for (int nn=0; nn<X.rows(); nn++) {
    Mu.row( (int) Z(nn,0) ) += X.row( nn );
    NperCluster[ (int) Z(nn,0)] += 1;
  }  
  Mu.colwise() /= NperCluster;
  clock_t endtime = clock();
  //cout << "Mu: " << get_elapsed_time( starttime, endtime ) << " sec." << endl;
}

double assignClosest( MexMat &X, MexMat &Mu, MexMat &Z, Mat &Dist) {
  clock_t starttime = clock();
  double totalDist = 0;
  int minRowID;

  pairwise_distance( X, Mu, Dist );

  for (int nn=0; nn<X.rows(); nn++) {
    totalDist += Dist.row(nn).minCoeff( &minRowID );
    Z(nn,0) = minRowID;
  }
  clock_t endtime = clock();
  //cout << "Z: " << get_elapsed_time( starttime, endtime ) << " sec." << endl;
  return totalDist;
}

void run_lloyd( MexMat &X, MexMat &Mu, MexMat &Z, int Niter )  {
  double prevDist,totalDist = 0;
  //cout << "HELLO!" << doDEBUG << endl;
  Mat Dist = Mat::Zero( X.rows(), Mu.rows() );
  
  //Mu.matrix().transposeInPlace();
  for (int iter=0; iter<Niter; iter++) {
    
    totalDist = assignClosest( X, Mu, Z, Dist );
    calc_Mu( X, Mu, Z );
    if ( prevDist == totalDist ) {
      break;
    }
    prevDist = totalDist;
  }
  //cout << "goodbye!" << endl;
  //Mu.matrix().transposeInPlace();
}

/*
double assignClosestBAD( MexMat &X, MexMat &Mu, MexMat &Z) {
  clock_t starttime = clock();
  double totalDist = 0;
  int minRowID;
  //VectorXd xn = VectorXd::Zero( Mu.cols() );

  VectorXd mTm = ( Mu.square().rowwise().sum() ).matrix();
  for (int nn=0; nn<X.rows(); nn++) {
    VectorXd Dist = -2*( Mu.matrix() * (VectorXd) X.row(nn) ) + mTm;
    Dist.minCoeff( &minRowID );
    Z(nn,0) = minRowID;
  }
  clock_t endtime = clock();
  cout << "Z: " << get_elapsed_time( starttime, endtime ) << " sec." << endl;
  return totalDist;
}
*/
