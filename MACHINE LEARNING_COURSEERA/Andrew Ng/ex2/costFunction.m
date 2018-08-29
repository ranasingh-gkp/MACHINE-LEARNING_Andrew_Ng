function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples
n=length(theta);
h_theta=0;
% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

for i=1:m
	for j=1:n
		h_theta=h_theta+theta(j,1)*X(i,j);
	end; 
h_t=sigmoid(h_theta);
h_theta=0;
J=J+y(i,1)*log(h_t)+(1-y(i,1))*log(1-h_t);
end;
J=-1*J/m;
for k=1:n
for i=1:m
	for j=1:n
	h_theta=h_theta+theta(j,1)*X(i,j);
	end;
	h_t=sigmoid(h_theta);
	grad(k)=grad(k)+(h_t-y(i,1))*X(i,k);
	h_theta=0;
end;
grad(k)=grad(k)/m;
end;
		
% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%








% =============================================================

end
