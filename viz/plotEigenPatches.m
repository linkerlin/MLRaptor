function [] = plotEigenPatches( dataName, modelName, infName, jobName, taskID )
DUMPROOT = '../results';

D = 64;
nSubplot = 64;

if strcmp( dataName, 'Zoran')
  X = load( '~/code/epll/GSModel_8x8_200_2M_noDC_zeromean.mat');
  K = 200;
else
  DUMPDIR = fullfile( DUMPROOT, dataName, modelName, infName, jobName );
  X = load( fullfile( DUMPDIR, num2str(taskID), 'BestObsModel.mat') );
  K = size(X.L,3);
end

for k = 1:min( K, 5 )
   if strcmp( dataName, 'Zoran')
     Sigma = X.GS.covs(:,:,k);
   elseif strcmp( modelName, 'MixModel' ) && strcmp(infName, 'EM')
     Sigma = inv( X.L(:,:,k) );
   elseif strcmp(infName, 'VB') || strcmp(infName, 'oVB')
     Sigma = X.invW(:,:,k)/( X.v(k) - D - 1 );
   end
   
   [V,D] = eig( Sigma );
   
   figure( k );
   for dd = 1:nSubplot
       eigenPatch = reshape( V(:,dd), 8, 8);
       subplot( sqrt(nSubplot), sqrt(nSubplot), dd);
       imagesc( eigenPatch, [-1 1] );
       set( gca, 'XTick', [], 'YTick', []);   
   end
   colormap( 'bone');
   
end

