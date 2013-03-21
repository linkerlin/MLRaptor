function MatrixProductBenchmark( testName, doV )
nTrial =10;

if strcmp( testName, 'XY' )
    runXY( 100, 100, [1e4 2e4 4e4 8e4 16e4], nTrial, doV );
elseif strcmp( testName, 'XTY' )
    runXTY( 100, 100, [1e4 2e4 4e4 8e4 16e4], nTrial, doV );
elseif strcmp( testName, 'XTX' )
    runXTX( 2e5, nTrial, doV);
end

end

function [] = runXTY( N, M, Krange, nTrial, doV )
if doV
fprintf( 'Benchmark: X^T * Y\n' );
end
for K = Krange
    X = rand( K, N);
    Y = rand( K, M);
    X'*Y; %warmup
    
    tic;
    for rep = 1:nTrial
        X'*Y;
    end
    elapsedtime = toc;
    
    
    if doV
        fprintf( '  %d x %6d x %d | %.3f sec/trial\n', N, K, M, elapsedtime/nTrial);
    else
        fprintf( '%.3f\n', elapsedtime/nTrial );
    end
end

end

function [] = runXY( N, M, Krange, nTrial, doV )
if doV
fprintf( 'Benchmark: X*Y\n' );
end
for K = Krange
    X = rand( N, K);
    Y = rand( K, M);
    X*Y; %warmup
    
    tic;
    for rep = 1:nTrial
        X*Y;
    end
    elapsedtime = toc;
    

    if doV
        fprintf( '  %d x %6d x %d | %.3f sec/trial\n', N, K, M, elapsedtime/nTrial);
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