function PlotBenchmarkResults( filename )

if strcmp( filename(1:3), 'XTY' ) || strcmp( filename(1:2), 'XY' )
    Krange = [1e4 2e4 4e4 8e4 16e4]
else
    Krange = [10 20 40 80 160]
end

fid = fopen( filename );
W = textscan( fid, '%.3f', 'CommentStyle', '#');
fclose(fid);
nTests = length( W{1} )/length(Krange);
ElapsedTime = reshape( W{1}, length(Krange), nTests);

plot( Krange, ElapsedTime, '.-', 'LineWidth', 2, 'MarkerSize', 20 );

fid = fopen( filename );
W = textscan( fid, '%s', 'Delimiter','\n');
fclose(fid);

W = W{1}
LegNames = {};
locID = 1;
for tt = 1:nTests
    LegNames{tt} = W{ locID }( 15:end );
    locID = locID + length(Krange) + 1
end
legend( LegNames, 'Location', 'NorthWest');

if strcmp( filename(1:2), 'XY' )
    titleStr = 'X * Y';%,   where X : 100 xD, Y = D x 100';
elseif strcmp( filename(1:3), 'XTY' )   
    titleStr = 'X^T * Y' %,   where X : D x 100, Y = D x 100';
else
    titleStr = 'X^T * X'; %,   where X : 2x10^5 x D';
end
title( titleStr, 'FontSize', 20);
xlabel( 'Dimension D', 'FontSize', 18);
ylabel( 'Time (sec)', 'FontSize', 18);