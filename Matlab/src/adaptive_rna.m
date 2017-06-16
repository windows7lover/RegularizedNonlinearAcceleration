function [x_extrapolated,c,info] = adaptive_rna(funval,iterations,param)
% [x_extrapolated,c,info] = ADAPTIVE_RNA(funval,iterations,param)
%
% Apply the regularized nonlinear acceleration algorithm on the given 
% iterations, with several heuristics, to extrapolate a new (better) point.
% The regularization parameter is automatically found using a grid search.
%
%   Example of use
%   --------------
%   % minimization of function "f" using rna+gradient method
%   x = x0;
%   x_history = zeros(d,k);
%   for i=1:N
%       for j=1:k 
%           % perform k gradient step
%           x = x-stepSize*gradient(x); % gradient step
%           x_history(:,j) = x;
%       end
%       % Restart using x0 = x_extrapolated
%       x = adaptive_rna(f,x_history);
%   end
%
%   List of outputs
%   ---------------
%
%   - x_extrapolated: computed using RegularizedNonlinearAcceleration, with
%     additionnal adaptive heuristics listed below.
%   - c: coefficients of the weigthed mean (see the help of function
%     RegularizedNonlinearAcceleration for more information)
%   - info: structure which gives additionnal information
%       * info.nfcalls: number of call of funval(x).
%       * info.lambdaUsed: lambda used for computing c.
%       * info.stepSizeMultiplier: multiplicative factor in front of the 
%                                  extrapolation step.
%       * info.param: param used in the algorithm.
%
%   List of inputs
%   --------------
%
% 	- funval: a function handle, returning the value of the objective 
%     function to minimize.
% 	- iterations: Matrix of size d x (k+1) , where d is the dimention of 
%     the space and k+1 the number of points to extrapolate.
% 	- param: a structure which contains the options of the acceleration 
%     algorithm.
%
%   Description of the parameters
%   -----------------------------
%
% * param.doAdaptiveLambda (default: true); 
% 		Determine if lambda should change over time
%
%
% * param.doLineSearch (default: true); 
% 		Determine if we should perform a line search on the stepsize at the
%       end
%
% * param.forceDecrease (default: true); 
%       Optionnal, check if the extrapolated value is smaller than the last
%       iterate of gradient method. If not, discard the extrapolation.
%
% * param.lambdaRange (default: NaN);
%       /!\ Active only if param.doAdaptiveLambda == true /!\
%       param.lambdaRange = [lambda_min,lambda_max]. Determine the range of
%       the extremal value of the grid search for lambda. If lambdaRange is
%       NaN, then the range is computed automatically.
%
% * param.sizeGrid (default: NaN);
%       /!\ Active only if param.doAdaptiveLambda == true /!\
%       /!\ Active only if param.lambdaRange != NaN /!\
%       The value param.sizeGrid fixed the number of different trials for
%       the grid search.
%
% * param.lambda (default 1e-6); 
%       /!\ Active only if param.doAdaptiveLambda == false /!\
%       The value param.lambda fixes the regularization.       
%
%
%   Related paper:
%   - Scieur, d'Aspremont and Bach. Regularized Nonlinear Acceleration.
%
% See also:
% RegularizedNonlinearAcceleration

%% Retrieve info
k = size(iterations,2)-1;
info.nfcalls = 0;

%% Inputs check

if(nargin<3)
    param = struct();
end

if(~isfield(param,'doLineSearch'))
    param.doLineSearch = true;
end

if(~isfield(param,'doAdaptiveLambda'))
    param.doAdaptiveLambda = true;
end

if(~isfield(param,'forceDecrease'))
    param.forceDecrease = true;
end


if(param.doAdaptiveLambda)
    
    if(~isfield(param,'lambdaRange'))
        param.lambdaRange = NaN;
    end
    
    if(any(isnan(param.lambdaRange)))
        lambdavec = NaN;
    else
        if(length(param.lambdaRange) ~=2)
            error('Size of param.lambdaRange should be 2.')
        end
        lambda_range = param.lambdaRange;
        if(~isfield(param,'sizeGrid'))
            param.sizeGrid = k+1;
        end
        min_l = min(lambda_range);
        max_l = max(lambda_range);
        lambdavec = [0, logspace(log10(min_l),log10(max_l),k)];
    end
    
    
else
    if(~isfield(param,'lambda'))
        param.lambda = 1e-6;
    end
    lambdavec = param.lambda;
end


%% Initializations

algo_x = iterations; % sequence
UU = diff(algo_x,1,2);
UU = UU'*UU;
UU_norm = UU/norm(UU); % normalized matrix U

if(isnan(lambdavec))
    % compute the grid using SVD
    vec = log(svd(UU_norm));
    vec = (vec(1:end-1)+vec(2:end))/2;
    lambdavec=[0 ; exp(vec)];
end
fvalvec = zeros(size(lambdavec)); % for the grid search

%% Grid search on lambda

for i=1:length(lambdavec) % length(lambdavec) = 1 if not adaptive
    [x_extrapolated,c] = RegularizedNonlinearAcceleration(algo_x, UU_norm, lambdavec(i)); 
    % extrapolation using differents values of lambda
    
    if(param.doAdaptiveLambda)
        fvalvec(i) = funval(x_extrapolated);
        info.nfcalls = info.nfcalls +1;
    else
        fvalvec = 0;
    end
end

if( param.forceDecrease && min(fvalvec) > funval(algo_x(:,end)) ) % Force decrease
    info.nfcalls = info.nfcalls +1;
    x_extrapolated = algo_x(:,end); % If bad extrapolation, return the last point of the algorithm
else
    [~, idx_min] = min(fvalvec);
    lambdamin = lambdavec(idx_min);
    [x_extrapolated,c] = RegularizedNonlinearAcceleration(algo_x, UU_norm, lambdamin);
    info.lambdaUsed = lambdamin;
end

% Line search on the stepsize
info.stepSizeMultiplier = 1;
if(param.doLineSearch)
    % Find a good stepsize, i.e a good alpha such that
    %           f( x_0 - alpha*(step) )
    % is small, where step = (x_extr-x_0).
    x = iterations(:,1);
    step = x_extrapolated-x;
    sizestep = 1;
    fold = funval(x+sizestep*step);
    info.nfcalls = info.nfcalls +1;
    sizestep = 2*sizestep;

    while(true)
        fnew = funval(x+sizestep*step);
        info.nfcalls = info.nfcalls +1;
        if(fold>fnew)
            fold = fnew;
            sizestep = 2*sizestep;
        else
            break;
        end
    end
    sizestep = sizestep/2;
    info.stepSizeMultiplier = sizestep;
    x_extrapolated = x+sizestep*step;
end

info.param = param;
