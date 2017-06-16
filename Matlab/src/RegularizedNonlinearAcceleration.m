function [ x_extr, c ] = RegularizedNonlinearAcceleration( x, UU, lambda )
%REGULARIZEDNONLINEARACCELERATION
%    [x_extr] = REGULARIZEDNONLINEARACCELERATION(x) extrapolate the limit
%    oft he  sequence of column-vectors [x(:,1) ... x(:,end)] with a 
%    weigthed mean of coefficients c, computed by solving the problem 
%	    c^* = argmin_c |Uc|^2
%	    s.t. sum(c) == 1
%   where U(:,i) = x(:,i+1) - x(:,i). 
%   The output x_extr is equal to sum_i=1^k c^*_i x_i.
%
%   [x_extr,c] = REGULARIZEDNONLINEARACCELERATION(x) also output the
%   computed coefficients of the weighted mean.
%
%   [...] = REGULARIZEDNONLINEARACCELERATION(x,UU) assume the matrix UU to
%   be U'*U/norm(U'*U), where U = diff(x,1,2).
%
%   [...] = REGULARIZEDNONLINEARACCELERATION(x,UU,lambda) extrapolates the 
%   limit using the regularization parameter lambda, i.e., it computes the
%   coefficients c by solving
%	    c^* = argmin_c |Uc|^2 + lambda |c|^2
%	    s.t. sum(c) == 1
%   This usually leads to better performances if lambda is well chosen.
%
%   Related paper:
%   - Scieur, d'Aspremont and Bach. Regularized Nonlinear Acceleration.
% 
%   See also:
%   adaptive_rna

if(nargin<2)
    UU = diff(x,1,2);
    UU = UU'*UU;
    UU = UU/norm(UU);
end

if(nargin<3)
    lambda = 0;
end

if(size(UU,2)==0)
    x_extr = x;
    return;
end

% U = diff(x,1,2);
k = size(UU,2);

matrix = (UU + eye(k)*lambda);

c = matrix\ones(k,1);
c = c/sum(c);

x_extr=x(:,1:end-1)*c;
