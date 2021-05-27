function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

%cost function
for i = 1:m
    x_ = theta'*X(i,:)';          
    J = J + (-1*y(i)*log(sigmoid(x_)) - (1-y(i))*log(1-sigmoid(x_)));       
end
J = J /m;

[r,c] = size(X);
aux = 0;
for j = 2:c
   aux = aux + theta(j)^2;   
end
aux =  aux*lambda/(2*m);
J = J + aux;


for j = 1:c    
   for i = 1:r      
        x_ = theta'*X(i,:)';                
        grad(j) = grad(j) +  (sigmoid(x_)-y(i))*X(i,j);
   end
   grad(j) = grad(j)/m;
  if j ~= 1
     grad(j) = grad(j) + lambda*theta(j)/m;
  end
end


% =============================================================

end
