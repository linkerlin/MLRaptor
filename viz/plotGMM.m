function [] = plotGMM( dataName, modelName, infName, jobName, taskID)

GMMstruct = loadSavedModel( dataName, modelName, infName, jobName, taskID);


MyColors = jet( GMMstruct.nmodels );

t = -pi:.01:pi;
x = sin(t);
y = cos(t);
    
figure;
hold all;

for kk = 1:GMMstruct.nmodels
    
    Mu    = GMMstruct.means( 1:2,kk );
    Sigma = GMMstruct.covs( 1:2, 1:2,kk);
        
    [V,D] = eig( Sigma );
    sqrtSigma = real( V*sqrt(D) )';    
    Zraw = [x' y']*sqrtSigma;
    
    for Rad = linspace(0.1, 2, 4)
        
        Z = Rad.* Zraw;
        Z = bsxfun( @plus, Z, Mu' );
        plot( Z(:,1), Z(:,2), '.', 'Color', MyColors(kk,:) );
    end
    
end