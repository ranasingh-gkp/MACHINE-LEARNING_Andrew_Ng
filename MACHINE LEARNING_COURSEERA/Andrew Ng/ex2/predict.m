function p = predict(theta, X)
%PREDICT Predict whether the label is 0 or 1 using learned logistic 
%regression parameters theta
%   p = PREDICT(theta, X) computes the predictions for X using a 
%   threshold at 0.5 (i.e., if sigmoid(theta'*x) >= 0.5, predict 1)

m = size(X, 1); % Number of training examples

% You need to return the following variables correctly
p = zeros(m, 1);
h_t=0;
n=length(theta);
k=0;
for i=1:m
	for j=1:n
	k=k+theta(j,1)*X(i,j);
	end;
h_t=sigmoid(k);
if(h_t<0.5)
p(i,1)=0;
else
p(i,1)=1;
end;
k=0;
end;
% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned logistic regression parameters. 
%               You should set p to a vector of 0's and 1's
%







% =========================================================================


end
