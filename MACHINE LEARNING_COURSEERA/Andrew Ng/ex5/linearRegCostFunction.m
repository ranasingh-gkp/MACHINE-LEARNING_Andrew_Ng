function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
theta_t=theta(2:end,1);
grad = zeros(size(theta));
h_theta=zeros(size(y));
h_t=h_theta;

h_theta=X*theta;
h_t=(h_theta-y).^2;
J=sum(h_t(:));
theta_t=theta_t.^2;
J=J+sum(theta_t(:))*lambda;
J=J/(2*m);

grad(1,1)=X(:,1)'*(h_theta-y);
grad(1,1)=grad(1,1)/m;
for i=2:length(grad)
	grad(i,1)=X(:,i)'*(h_theta-y)+lambda*theta(i,1);
	grad(i,1)=grad(i,1)/m;
end;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%












% =========================================================================

grad = grad(:);

end
