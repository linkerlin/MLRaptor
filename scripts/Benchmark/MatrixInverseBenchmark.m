function MatrixInverseBenchmark( testName, doV )
nTrial =10;

if strcmp( testName, 'invSX' )
    runinvSX( 250000, [10,20,40,80,160], nTrial, doV );
elseif strcmp( testName, 'invSXT' )
    runinvSXT( 250000, [10,20,40,80,160], nTrial, doV );
elseif strcmp( testName, 'invcholSX')
    runinvcholSX( 250000, [10,20,40,80,160], nTrial, doV);
end

end

function [] = runinvcholSX( N, Krange, nTrial, doV)
if doV
        fprintf( 'Benchmark: inv(chol(S)) * X\n');
end
for K = Krange
   S = rand(K,K);
   cholS = chol( S'*S, 'lower');
   X = rand(K,N);
   cholS \ X;

  tic;
  for rep = 1:nTrial
    cholS \ X;
  end
  elapsedtime = toc;

if doV
  fprintf( '  %d x %d x %6d | %.3f sec/trial\n', K,K,N, elapsedtime/nTrial);
else
  fprintf( '%.3f\n', elapsedtime/nTrial);
end
end
end

function [] = runinvSX( N, Krange, nTrial, doV )
if doV
fprintf( 'Benchmark: inv(S) * X\n' );
end
for K = Krange
    S = rand( K,K);
    S = S'*S;
    X = rand( K, N);
    
    S\X; %warmup
    
    tic;
    for rep = 1:nTrial
       S\X;
    end
    elapsedtime = toc;
    
    
    if doV
        fprintf( '  %d x %d x %6d | %.3f sec/trial\n', K,K,N, elapsedtime/nTrial);
    else
        fprintf( '%.3f\n', elapsedtime/nTrial );
    end
end

end

function [] = runinvSXT( N, Krange, nTrial, doV )
if doV
fprintf( 'Benchmark: inv(S)*X^T\n' );
end
for K = Krange
    X = rand( N, K);
    S = rand(K,K);
    S = S'*S;
    S\X'; %warmup
    
    tic;
    for rep = 1:nTrial
        S\X';
    end
    elapsedtime = toc;
  
    if doV
        fprintf( '  %d x %d x %6d | %.3f sec/trial\n', K, K, N, elapsedtime/nTrial);
    else
        fprintf( '%.3f\n', elapsedtime/nTrial );
    end
end

end

function runXTX( N, nTrial, doV)

if doV
fprintf( 'Benchmark: X^T * X\n' );
end
for K = [10 20 40 80 160]
    X = rand( N, K);
    X'*X; % warmup
    
    tic;
    for rep = 1:nTrial
        X'*X;
    end
    elapsedtime = toc;
    

    if doV
        fprintf( '  %d x %6d x %d | %.3f sec/trial\n', N, K, M, elapsedtime/nTrial);
    else
        fprintf( '%.3f\n', elapsedtime/nTrial );
    end
end
end
