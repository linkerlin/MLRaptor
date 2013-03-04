from FracVecOptimizer import FracVecOptimizer


import OptimHDPstick as HDPStick

def main():
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument('--solver', type=str, default='py')

  parser.add_argument('--G', type=int, default=10000)
  parser.add_argument('--K', type=int, default=10)
  parser.add_argument('--gamma', type=float, default=25)
  parser.add_argument('--alpha0', type=float, default=None)

  parser.add_argument('--seed', type=int, default=8675309)
  parser.add_argument('--Ntrial', type=int, default=5)
  parser.add_argument('--initname', type=str, default='rand')

  args=  parser.parse_args()

  if args.alpha0 is None:
    args.alpha0= args.K/2

  Problem = HDPStick.gen_problem( args.G, args.K, args.alpha0, args.gamma, seed=args.seed)
  if args.solver.count('py')>0:
    optEngine = FracVecOptimizer( Problem, LB=1e-6 ) 
  else:
    from MLabFracVecOptimizer import MLabFracVecOptimizer
    optEngine = MLabFracVecOptimizer( Problem, LB=1e-6 ) 
  optEngine.run_many_trials( args.Ntrial, args.initname)

if __name__ == '__main__':
  main()
