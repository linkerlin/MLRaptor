function [] = DenoiseHeldoutData( nTest, dataName, modelName, infName, jobName, taskID )
addpath( genpath( '~/code/epll/' ) );

patchSize = 8;
noiseSD = 25/255;
try
    TestImgDir = '/data/BSR/BSDS500/data/images/test/';
    assert( 0<exist( TestImgDir, 'dir') );
catch e    
    TestImgDir = '/data/liv/mhughes/img/BSDS300/images/test/';
    assert( 0<exist( TestImgDir, 'dir') );
end
TestImgList = dir( fullfile(TestImgDir,'*.jpg') );
TestImgList = TestImgList(1:nTest);

if strcmp( jobName, 'Zoran') || strcmp( jobName, 'zoran')
    GMMstruct = load('~/code/epll/GSModel_8x8_200_2M_noDC_zeromean.mat');
    GMMstruct = GMMstruct.GS;
else
    GMMstruct = loadSavedModel( dataName, modelName, infName, jobName, taskID );
end

doLoadImgSpecificWeights = isempty( GMMstruct.mixweights );

% Config all options for denoising
priorFunc = @(Z,patchSize,noiseSD,imsize) aprxMAPGMM(Z,patchSize,noiseSD,GMMstruct);
betas = (1/noiseSD^2)*[ 1 4 8 16 32 ];
LogLFunc = [];

% Prealloc storage for resulting signal-noise ratio
PSNR = zeros( 1, length(TestImgList)  );
NoisyImgs = cell( nTest, 1);
CleanImgs = cell( nTest, 1);
TrueImgs = cell( nTest, 1);

dumpdir = fullfile( '/home/mhughes/git/MLRaptor/results/', dataName, modelName, infName, jobName);
[~,~] = mkdir( dumpdir);
dumpdir = fullfile( dumpdir, num2str(taskID) );
[~,~] = mkdir( dumpdir);
[~,~] = mkdir( dumpdir, 'denoise');

fprintf( 'Denoise Experimental Setup\n');
fprintf( '  Patch Size: %d\n', patchSize);
fprintf( '  Noise Std Dev: %.3f\n', noiseSD);

fprintf( 'Heldout Data: Berkeley Segmentation Test Set\n');
fprintf( '  Num. Test Images: %d\n', nTest);

fprintf( 'Model Info\n');
fprintf( '  AllocModel: %s\n', modelName);
fprintf( '  K= %d\n', GMMstruct.nmodels);

fprintf( 'Running Denoising inference...\n');
for aa = 1:length( TestImgList )

   % Load image + preprocess so matrix has values between (0,1)
   curimpath = fullfile( TestImgDir, TestImgList(aa).name );
   I = double(rgb2gray(imread( curimpath )))/255;    

   % Add gaussian noise (ensuring reproduceability w/ rand seed)
   rng(8675309 + aa);
   noisyI = I + noiseSD*randn(size(I));
   
   if doLoadImgSpecificWeights
       [~,shortname,ext] = fileparts( TestImgList(aa).name );
       wHat = load( fullfile( dumpdir, 'heldout', [shortname '.wHat.txt']) );
       assert( ~isempty( wHat) && length(wHat) == GMMstruct.nmodels );
       GMMstruct.mixweights = wHat;
       priorFunc = @(Z,patchSize,noiseSD,imsize) aprxMAPGMM(Z,patchSize,noiseSD,GMMstruct);
   end

   tic;
   cleanI = EPLLhalfQuadraticSplit(noisyI, patchSize^2/noiseSD^2, patchSize, betas, 1, priorFunc, I, LogLFunc);

   PSNR( aa ) = 20*log10( 1/std2( cleanI-I)  );
   
   fprintf(' Image %d/%d | PSNR: %.2f | %.0f sec\n', aa, length(TestImgList), PSNR(aa), toc );
   
   NoisyImgs{aa} = noisyI;
   TrueImgs{aa} = I;
   CleanImgs{aa} = cleanI;
   
   outmatfile = fullfile( dumpdir, 'denoise', 'Results.mat');
   save( outmatfile, 'PSNR', 'betas', 'patchSize', 'noiseSD', 'TestImgList', 'TestImgDir', 'NoisyImgs', 'TrueImgs', 'CleanImgs');
   
end
fprintf( 'Wrote results to %s\n', outmatfile);