function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples
n=length(theta);
% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));
h_theta=0;
h_t=0;
sum=0;

	for k=2:n
	sum=sum+theta(k,1)^2;
	end;	
	sum=sum*lambda/2;
for i=1:m
	for j=1:n
	h_theta=h_theta+X(i,j)*theta(j,1);	
	end;
	h_t=sigmoid(h_theta);
	h_theta=0;
	J=J-(y(i,1)*log(h_t)+(1-y(i,1))*log(1-h_t));	
end;

J=J+sum;
J=J/m;
sum=0;
for i=1:n
	for j=1:m
	for k=1:n
	h_theta=h_theta+X(j,k)*theta(k,1);
	end;
	
	h_t=sigmoid(h_theta);
	h_theta=0;
	grad(i)=grad(i)+(h_t-y(j,1))*X(j,i);
	end;
if(i!=1)
sum=lambda*theta(i,1);
end;
grad(i)=grad(i)+sum;
grad(i)=grad(i)/m;

end;
% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta






% =============================================================

end
