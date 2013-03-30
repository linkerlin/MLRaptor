function [] = CalcGMMLogPDF( nProc)
N = 250000;
D = 64;
K = 25;

if nProc > 1
    try
        matlabpool( 'local', nProc);
    catch Exception e
       matlabpool close;
       matlabpool( 'local', nProc);
    end
end

X = randn( N, D);
w = ones( 1, K)./K;
mu = randn( K, D);
Sigma = randn( D, D, K);
for kk = 1:K
    Sigma(:,:,kk) = Sigma(:,:,kk)'* Sigma(:,:,kk);
end

tic;
Estep( X, w, mu, Sigma);
toc

if nProc > 1
    matlabpool close;
end
    
end

function [logResp] = Estep( X, w, mu, Sigma)

N = size(X,1);
K = size(mu,1);

logResp = zeros(N,K);
parfor k = 1:K
    logResp(:,k) = loggausspdf(X, mu(k,:), Sigma(:,:,k) );
end
logResp = bsxfun(@plus, logResp, log(w));
end

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
Xdiff = bsxfun(@minus,X,mu);
[cholSigma, status]= chol(Sigma, 'lower');
if status ~= 0
    error('ERROR: Sigma is not PD.');
end
Q = cholSigma\Xdiff';
distPerRow = sum( Q.^2, 1);  % quadratic term (M distance)
end
