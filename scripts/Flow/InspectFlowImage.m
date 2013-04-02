fList = dir(  '/data/liv/visiondatasets/sintel/training/flow/ambush_4/*.flo' );

RawImg = imread( '/data/liv/visiondatasets/sintel/training/clean/ambush_4/frame_0013.png');
Flow = readFlowFile( '/data/liv/visiondatasets/sintel/training/flow/ambush_4/frame_0013.flo');
PatchData = load( '/data/liv/visiondatasets/sintel/patches/ambush_4/frame_0013.dat');
[H,W,~] = size(Flow);

figure(101);
clf;
set( gcf, 'Units', 'normalized', 'Position', [0.5 0 1 1]);
hold on;

RawImH = subplot( 'Position', [0.1 0.65 0.8 0.35] );
imagesc( RawImg );
set( gca, 'XTick',[],'YTick',[]);

FlowImH = subplot( 'Position', [0.1 0.25 0.8 0.35] );
imagesc( flowToColor(Flow, 50) );
set( gca, 'XTick',[],'YTick',[]);

TruePatchImH = subplot( 'Position', [0.1 0.02 0.2 0.2] );
%imagesc( flowToColor(Flow(1:10, 1:10,:), 50) );
%set( gca, 'XTick',[],'YTick',[]);
%axis image;

PyPatchImH = subplot( 'Position', [0.5 0.02 0.2 0.2] );
%imagesc( flowToColor( reshape( PatchData(1,:), [10 10 2]), 50) );
%set( gca, 'XTick',[],'YTick',[]);
%axis image;

LEFT = 28;
RIGHT = 29;
ESC = 27;

button=0;
curPos = 1;
doStart = 1;
while button ~= ESC
       
    [x,y,button] = ginput(1);
    fprintf( '%.2f %.2f\n', x,y);
    
    if button == ESC
        break;
    end
    
    didChange = 0;
    if button == LEFT
        curPos = curPos-1;
        curPos = max( curPos, 1);   
        didChange = 1;
    elseif button == RIGHT        
        curPos = curPos+1;
        curPos = min( curPos, length(fList) );
        didChange = 1;
    end
    
    if didChange || doStart
        doStart = 0;
        shortname = fList( curPos ).name;
        [~,shortname,ext] = fileparts( shortname );
        RawImg = imread( ['/data/liv/visiondatasets/sintel/training/clean/ambush_4/' shortname '.png'] );
        Flow = readFlowFile( ['/data/liv/visiondatasets/sintel/training/flow/ambush_4/' shortname '.flo']);
        PatchData = load( ['/data/liv/visiondatasets/sintel/patches/ambush_4/' shortname '.dat'] );
        
        subplot( RawImH );
        imagesc( RawImg );
        set( gca, 'XTick',[],'YTick',[]);
        
        
        subplot( FlowImH );
        imagesc( flowToColor(Flow, 50) );
        set( gca, 'XTick',[],'YTick',[]);
        shortname
        continue;
    end
    
    xLoc = 10*floor( x/10 )+1;
    yLoc = 10*floor( y/10 )+1;

    subplot( TruePatchImH);
    imagesc( flowToColor(Flow(yLoc:yLoc+9, xLoc:xLoc+9,:), 50) );
    set( gca, 'XTick',[],'YTick',[]);
    axis image;
    
    %Debug printout
    %Flow(yLoc:yLoc+9, xLoc:xLoc+9, 1)
    
    
    subplot( PyPatchImH);
    xLoc = floor( x/10 )+1;
    yLoc = floor( y/10 )+1;
    
    %yLoc = floor(H/10)-yLoc; % flipupdown
    pLoc = (yLoc-1)*floor( W/10 ) + xLoc;
    fX = reshape( PatchData(pLoc,1:100), [10 10])';    
    fY = reshape( PatchData(pLoc,101:end), [10 10])';
    flowStack(:,:,1) = fX;
    flowStack(:,:,2) = fY;
    
    imagesc( flowToColor( flowStack, 50) );
    set( gca, 'XTick',[],'YTick',[]);
    axis image;
    
   fprintf( '         %d %d %d\n', xLoc, yLoc, pLoc);
    
    
end
