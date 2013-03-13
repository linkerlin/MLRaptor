#include "Eigen/Dense"
#include "mersenneTwister2002.c"
#include <iostream>
using namespace Eigen;
using namespace std;

typedef Map<ArrayXXd,Aligned> MexMat;
typedef Map<ArrayXd,Aligned> VecMat;
typedef ArrayXXd Mat;
typedef ArrayXd Vec;


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
  Mu = Mat::Zero( Mu.rows(), Mu.cols() );
  Vec NperCluster = Vec::Zero( Mu.rows() );
  
  for (int nn=0; nn<X.rows(); nn++) {
    Mu.row( (int) Z(nn,0) ) += X.row( nn );
    NperCluster[ (int) Z(nn,0)] += 1;
  }  

  Mu.colwise() /= NperCluster;
}

double assignClosest( MexMat &X, MexMat &Mu, MexMat &Z) {
  double totalDist = 0;
  int minRowID;
  for (int nn=0; nn<X.rows(); nn++) {
    totalDist += ( Mu.rowwise() - X.row(nn)  ).square().rowwise().sum().minCoeff( &minRowID );
    Z(nn,0) = minRowID;
  }
  return totalDist;
}

void run_lloyd( MexMat &X, MexMat &Mu, MexMat &Z, int Niter )  {
  double totalDist,prevDist = 0;
  cout << "Init Mu:" << endl<< Mu << endl;
  for (int iter=0; iter<Niter; iter++) {
    
    totalDist = assignClosest( X, Mu, Z );
    cout << totalDist << endl;    
    calc_Mu( X, Mu, Z );
    if ( prevDist == totalDist ) {
      break;
    }
    prevDist = totalDist;
    //cout <<  "Mu:" << endl<< Mu << endl;
  }
}

