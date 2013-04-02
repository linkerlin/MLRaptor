dumpdir = '/home/mhughes/git/MLRaptor/results/BerkSeg/';

Rb = load( fullfile( dumpdir, 'MixModel', 'VB', 'mikefast', '2', 'denoise', 'Results.mat'), 'PSNR');
Ro = load( fullfile( dumpdir, 'MixModel', 'oVB', 'mikefast', '4', 'denoise', 'Results.mat'), 'PSNR');

figure(501);
plot( Rb.PSNR, Ro.PSNR, 'k.', 'MarkerSize', 16);
hold on;
xs=linspace( 24, 33, 8);
plot( xs, xs, 'r--');
xlabel( 'Batch VB K=25 GMM', 'FontSize', 16);
ylabel( 'Online VB K=25 GMM', 'FontSize', 16);
title( 'Head-to-head compare PSNR for 30 test images', 'FontSize', 16);

bGMM = loadSavedModel( 'BerkSeg', 'MixModel', 'VB', 'mikefast', 1);
oGMM = loadSavedModel( 'BerkSeg', 'MixModel', 'oVB', 'mikefast', 1);

figure(601);
subplot( 2,1,1);
bar( bGMM.mixweights );
title( 'Batch VB Mix Weights');
subplot( 2,1,2);
bar( oGMM.mixweights );
title( 'Online VB Mix Weights');


TestImgDir = '/data/liv/mhughes/img/BSDS300/images/test/';
TestImgList = dir( fullfile(TestImgDir,'*.jpg') );
for aa = 1:5
[~,shortname,ext] = fileparts( TestImgList(aa).name );
bGMM.mixweights = load( fullfile( dumpdir, 'AdmixModel', 'VB', 'mikefast', '2', 'heldout', [shortname '.wHat.txt']) );
oGMM.mixweights = load( fullfile( dumpdir, 'AdmixModel', 'oVB', 'mikefast', '4', 'heldout', [shortname '.wHat.txt']) );

figure(700+aa);
subplot( 2,1,1);
bar( bGMM.mixweights );
title( 'Batch VB Mix Weights');
subplot( 2,1,2);
bar( oGMM.mixweights );
title( 'Online VB Mix Weights');
end