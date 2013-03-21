function [model,loglik] = runEMforGMM( data, K, Niter, savefilename, seed)
% Run EM algorithm for fitting the Gaussian mixture model with K components
% AUTHOR: Mike Hughes (mike@michaelchughes.com)
% INPUT
%   data  : NxD data matrix (each row is one i.i.d. observation)
%           or string path to .mat file
%   K     : # of mixture components (integer)
%   Niter : # of iterations to run
%   savefilename : string path to .mat file to save model
%   seed  : integer seed for random number generation (used for init only)
% ACKNOWLEDGEMENTS
%  Based on Matlab Central implementation by Michael (Mo) Chen.

if ischar( data )
    X = load( data );
    X = X.X;
else
    X = data;
end

CONVERGE_THR = 1e-6;
loglik = -inf( 1, Niter );

fprintf( 'EM for Mixture of %d Gaussians | seed=%d\n', K, seed);
Resp = init_responsibilities( X, K, seed );
tic;
for t = 1:Niter
    model = Mstep( X, Resp );
    [Resp, loglik(t)] = Estep( X, model );
    
    fprintf( ' %5d/%d after %.0f sec | %.8e \n', t, Niter, toc, loglik(t)  );
    
    if t > 1
        deltaLogLik = loglik(t) - loglik(t-1);
        if deltaLogLik < 0
            fprintf('WARNING: log lik decreased!\n');
        end
        if deltaLogLik < CONVERGE_THR
            break;
        end
    end
end
loglik = loglik( ~isinf(loglik) );
save( savefilename, '-struct', 'model' );
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function Resp = init_responsibilities(X, K, seed)
[N,D] = size(X);
if true    
    % Use python to generate random samples, so we can compare results
    PYCMD = sprintf( 'python -c "import random; random.seed(%d); print random.sample(xrange(%d),%d)"', seed, N, K);
    [status, resultStr] = system( PYCMD );
    rowIDs = str2num( resultStr );
    mu = X( rowIDs+1, :); % python output is 0-idx, so convert to matlab
else
    % Use matlab to generate random samples
    doReplace = false;
    rng(seed);
    rowIDs = randsample( N, K, doReplace );
    mu = X(rowIDs,:);
end
logResp = zeros(N,K);
for k = 1:K
    logResp(:,k) = loggausspdf( X, mu(k,:), eye(D) );
end
logProbPerRow = logsumexp(logResp,2);
Resp = exp( bsxfun(@minus,logResp, logProbPerRow) );
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [Resp, loglik] = Estep( X, model)
w = model.w;
mu = model.mu;
Sigma = model.Sigma;

N = size(X,1);
K = size(mu,1);

logResp = zeros(N,K);
for k = 1:K
    logResp(:,k) = loggausspdf(X, mu(k,:), Sigma(:,:,k) );
end
logResp = bsxfun(@plus, logResp, log(w));
logProbPerRow = logsumexp(logResp,2);

Resp = exp( bsxfun(@minus,logResp, logProbPerRow) );
loglik = sum( logProbPerRow );
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function model = Mstep(X, Resp)
[N,D] = size(X);
K = size(Resp,2);

Nk = sum( Resp,1) + eps;
w = Nk/N;
mu = bsxfun(@times, Resp'*X, 1./Nk' );
Sigma = zeros(D,D,K);
for k = 1:K
    Xdiff = bsxfun( @minus, X, mu(k,:) );
    Xdiff = bsxfun( @times, Xdiff, sqrt(Resp(:,k)) );
    Sigma(:,:,k) = (Xdiff'*Xdiff)/Nk(k) + 1e-8*eye(D);
end
model.w = w;
model.mu = mu;
model.Sigma = Sigma;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function logpdf = loggausspdf(X, mu, Sigma)
% Compute log pdf for each row in matrix X
%   logpdf(n) = log Pr( X(n,:) | mu, Sigma )

[N,D] = size(X);
[distM,cholSigma] = distMahal( X, mu, Sigma );
logdetSigma = 2*sum( log( diag( cholSigma )));
logNormConst = -0.5*D*log(2*pi) - 0.5*logdetSigma;
logpdf = logNormConst - 0.5*distM;
end

function [distPerRow, cholSigma] = distMahal( X, mu, Sigma )
[N,D] = size(X);
Xdiff = bsxfun(@minus,X,mu);
[cholSigma, status]= chol(Sigma, 'lower');
if status ~= 0
    error('ERROR: Sigma is not PD.');
end
Q = cholSigma\Xdiff';
distPerRow = sum( Q.^2, 1);  % quadratic term (M distance)
end

function logS = logsumexp( logA, dim)
% Compute log(sum(exp(logA),dim)) while avoiding numerical underflow.
logAMax = max( logA,[],dim);
logA = bsxfun(@minus,logA,logAMax);
logS = logAMax + log(sum(exp(logA),dim));
end
