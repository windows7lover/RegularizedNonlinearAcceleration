% We want to minimize a quadratic function using RNA + gradient descend.

clear all;
close all;
clc;

%% Parameters
dim = 50;
solution_radius = 1000; % how far is x^* from x0
k = 10; % inner-loop
N = 1000; % outer-loop

%% Starting point and solution
x0 = rand(dim,1);
xstar = rand(dim,1);
x0 = xstar + solution_radius*(x0-xstar)/norm(x0-xstar);

%% Definition of the quadratic function
A = rand(dim);
AA = A'*A;

f = @(x) 0.5*norm(A*(x-xstar))^2; % min_x |Ax-b|^2, where b = Ax*
gradient = @(x) AA*(x-xstar);
L = norm(AA); % Lipchitz constant


%% Minimization using accelerated gradient descend

% Initialization
error_history = zeros(k*N,1);
x_history = zeros(dim,k);
iter = 0;

% Set x = x0
x = x0;

% Optimal step size for convex functions
stepSize = 1/L;

for i=1:N
    
    % perform k gradient steps
    for j=1:k
        % gradient step
        x = x-stepSize*gradient(x); 
        
        % Store iterate x for extrapolation
        x_history(:,j) = x;
                
        % Record norm(x-xstar)
        iter = iter + 1;
        error_history(iter) = norm(x-xstar);
    end
    % Restart using x0 = x_extrapolated
    warning off % avoid ('Matrix is close to singular or badly scaled')
    x = adaptive_rna(f,x_history);
    warning on
end

%% We also use basic gradient descend for the comparison

% Initialization
error_history_gradient = zeros(k*N,1);

% Set x = x0
x = x0;

% Optimal step size for convex functions
stepSize = 1/L;

% perform k gradient steps
for i=1:k*N
    % gradient step
    x = x-stepSize*gradient(x); 

    % Record norm(x-xstar)
    error_history_gradient(i) = norm(x-xstar);
end

%% Plot
figure
semilogy(1:k*N,error_history_gradient,'r');
hold on
semilogy(1:k*N,error_history,'b');
legend({'Gradient Descend', 'RNA + Gradient Descend'},'location','SW')