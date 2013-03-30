figure;
hold on;
tNames = {'XY', 'XTY', 'XTX'};
for tt = 1:2
        fname = sprintf( '%s.dat', tNames{tt} );
        subplot( 1, 2, tt );
        PlotBenchmarkResults( fname );
        ylim( [0 0.65] );
end

% tNames = {'XY', 'XTY', 'XTX'};
% mNames = {'single', 'multi'};
% for mm = 1:2
%     for tt = 1:3
%         fname = sprintf( '%scompare_%s.dat', tNames{tt}, mNames{mm} );
%         subplot( 2, 3, 3*(mm-1) + tt );
%         PlotBenchmarkResults( fname );
%     end
% end