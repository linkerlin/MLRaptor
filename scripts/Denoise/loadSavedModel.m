function GMMStruct = loadSavedModel( dataName, modelName, infName, jobName, taskID)

dumpdir = fullfile( '~/git/MLRaptor/results/', dataName, modelName, infName, jobName, num2str(taskID) );

try
atype = load( fullfile( dumpdir, 'AllocModelType.txt') );
otype = load( fullfile( dumpdir, 'AllocModelType.txt') );
catch e
   atype = modelName;
   otype = 'Gaussian';
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Load global mix weights
if strcmp( atype, 'MixModel' )     
    AStruct = load( fullfile( dumpdir, 'BestAllocModel.mat') );
    if strcmp( infName, 'EM')
        w = AStruct.w;
    elseif strcmp( infName, 'VB') || strcmp( infName, 'oVB' )
        w = AStruct.alpha;
        w = w/sum(w);
    end
elseif strcmp( atype, 'DPMixModel')
    AStruct = load( fullfile( dumpdir, 'BestAllocModel.mat') );
    a1 = AStruct.qalpha1;
    a0 = AStruct.qalpha0;
    vstar = a1./(a0+a1);
    w = vstar;
    w(2:end) = w(2:end) .* cumprod( 1-vstar(1:end-1) );
elseif strcmp( atype, 'AdmixModel') || strcmp( atype, 'HDP')
    w = [];
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Load GMM component mu, Sigma
switch otype
    case 'Gaussian'
        OStruct = load( fullfile(dumpdir, 'BestObsModel.mat') );
        mu = OStruct.m;
        [D,K] = size(mu);
        if strcmp( infName, 'EM')
           invSigma = OStruct.L;
           Sigma = zeros( D, D, K);           
           for k = 1:K
               Sigma(:,:,k) = inv( invSigma(:,:,k) );
           end
        else
           Sigma = zeros( D, D, K);           
           invSigma = zeros( D, D, K);           
           for k = 1:K
               Sigma(:,:,k) = OStruct.invW(:,:,k) / (OStruct.v(k) - D - 1); %E[Sigma] under wishart
           end          
        end
        
        % Ensure exact symmetry!
        for k = 1:K
            Sigma(:,:,k) = 0.5*( Sigma(:,:,k) + Sigma(:,:,k)' );               
            invSigma(:,:,k) = inv( Sigma(:,:,k) );
        end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Package into final MStruct
GMMStruct = struct();
GMMStruct.mixweights = w;
GMMStruct.covs = Sigma;
GMMStruct.invcovs = invSigma;
GMMStruct.means = mu;
GMMStruct.nmodels = K;