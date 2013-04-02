dumpdir = '/home/mhughes/git/MLRaptor/results/BerkSeg/';

modelNames = {'MixModel', 'DPMixModel', 'AdmixModel'};
infNames = {'EM', 'VB', 'oVB'};
jobNames = {'zoran', 'mikefast'};

jitter = linspace(-0.05, 0.05, 4);

plotID = 0;
figure(101);
clf;
hold all;
plotNames = {};
minscore = 10000;
maxscore = 0;
for mm = 1:length( modelNames)
    for ii = 1:length( infNames)
        for jj = 1:length(jobNames)
            didWork = false;
            for taskID = 1:4

                try
                    fpath=fullfile( dumpdir, modelNames{mm}, infNames{ii}, jobNames{jj}, num2str(taskID), 'denoise', 'Results.mat');
                    Results = load( fpath, 'PSNR'  );
                    if taskID ==1
                        plotID = plotID+1;
                        plotNames{plotID} = [modelNames{mm} '-' infNames{ii} '-' jobNames{jj}];
                    end
                    
                catch e
                    continue; 
                end
                               
                score = mean( Results.PSNR );
                
                plot( score, plotID+ jitter(taskID), 'k+', 'MarkerSize', 14 );
                
                
                if score > maxscore
                    maxscore = ceil(score);
                end
                if score < minscore
                    minscore = floor(score);
                end
            end                        
        end
    end
end
set( gca, 'YTick', 1:plotID, 'YTickLabel', plotNames);
ylim( [0 plotID+1] );
xlim( [minscore maxscore]);
set( gca, 'FontSize', 14);
xlabel( 'PSNR (dB)', 'FontSize', 16);


R = load( fullfile( dumpdir, 'MixModel', 'EM', 'mikefast', '1', 'denoise', 'Results.mat'), 'PSNR');
R2 = load( fullfile( dumpdir, 'MixModel', 'EM', 'zoran', '1', 'denoise', 'Results.mat'), 'PSNR');
figure( 301);
plot( R.PSNR, R2.PSNR, 'k.', 'MarkerSize', 16);
hold on;
xs=linspace( 24, 33, 8);
plot( xs, xs, 'r--');
xlabel( 'My K=25 GMM', 'FontSize', 16);
ylabel( 'Zoran K=200 GMM', 'FontSize', 16);
title( 'Head-to-head compare PSNR for 30 test images', 'FontSize', 16);