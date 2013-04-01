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
                               
                score = median( Results.PSNR );
                
                plot( score, plotID+ jitter(taskID), 'k+', 'MarkerSize', 14 );
                
            end                        
        end
    end
end
set( gca, 'YTick', 1:plotID, 'YTickLabel', plotNames);
ylim( [0 plotID+1] );
set( gca, 'FontSize', 14);
xlabel( 'PSNR (dB)', 'FontSize', 16);