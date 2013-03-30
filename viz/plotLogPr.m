function [] = plotLogPr( dataName, modelName, infName, jobName, taskIDs )
DUMPROOT = '../results';


DUMPDIR = fullfile( DUMPROOT, dataName, modelName, infName, jobName );

if ~exist( 'taskIDs', 'var')
    queryDir = fullfile(DUMPDIR,'*')
    dList = dir(queryDir );
    dList = dList(3:end)
    taskIDs = 1:length( dList )
end

hold all;
for taskID = taskIDs
    
    curDUMPDIR = fullfile(DUMPDIR,  num2str(taskID) );
    
    try
        iters  = load( fullfile(curDUMPDIR, 'iters.txt') );
        ev     = load( fullfile(curDUMPDIR, 'evidence.txt'));
        %iters = load( fullfile(curDUMPDIR, 'trace.iters') );
        %ev    = load( fullfile(curDUMPDIR, 'trace.evidence') );
        plot( iters, ev, '.-', 'MarkerSize', 10, 'LineWidth', 2);
    catch e
        continue;
    end
end

