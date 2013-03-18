function MatrixProductBenchmark()

runXY( 100, 100, [1e4 2e4 4e4 8e4 16e4] );
runXTX();

end

function [] = runXY( N, M, Krange, nTrial )
if ~exist( 'nTrial', 'var')
    nTrial = 10;
end

fprintf( 'Benchmark: X*Y\n' );
for K = Krange
   X = rand( N, K); 
   Y = rand( K, M);
   tic;
   for rep = 1:nTrial
   X*Y;
   end
   elapsedtime = toc;
   
   fprintf( '  %d x %6d x %d | %.3f sec/trial\n', N, K, M, elapsedtime/nTrial);
end

end

function runXTX( N, nTrial)
N = 2e5;
nTrial = 10;

fprintf( 'Benchmark: X^T * X\n' );
for K = [10 20 40 80 160]
   X = rand( N, K); 
   tic;
   for rep = 1:nTrial
   X'*X;
   end
   elapsedtime = toc;
   
   fprintf( '  %d x %4d | %.3f sec/trial\n', N, K, elapsedtime/nTrial);
end
end