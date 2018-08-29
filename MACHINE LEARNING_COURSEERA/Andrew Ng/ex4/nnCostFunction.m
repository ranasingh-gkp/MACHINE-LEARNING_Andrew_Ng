function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
X=[ones(m,1) X];
[k1 l1]=size(Theta1);
[k2 l2]=size(Theta2);


a2=zeros(k1,1);

a2_new=zeros(k1+1,1);
d2=zeros(length(a2_new),1);
k2=d2;

a3=zeros(num_labels,1);
d3=zeros(num_labels,1);
delta2=zeros(size(Theta2));
delta1=zeros(length(d2)-1,length(X(1,:)));


res=zeros(num_labels, 1);
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));
sum=0;
for i=1:m
	res(y(i))=1;
	a2=Theta1*X(i,:)';
	z2=[1;a2];
	a2_new=[1; sigmoid(a2)];
	a3=Theta2*a2_new;
	a3=sigmoid(a3);
	for j=1:num_labels
	sum=sum+log(a3(j,1))*res(j,1);
	sum=sum+log(1-a3(j,1))*(1-res(j,1));
	end;
	d3=a3-res;
	d2=(Theta2)'*d3.*sigmoidGradient(z2);
	d2=d2(2:end);
	Theta2_grad=Theta2_grad+d3*(a2_new)';
	Theta1_grad=Theta1_grad+d2*X(i,1:end);
	d2=k2;
	res(y(i))=0;
end;
if(lambda~=0)
	for i=1:k1
		for j=2:l1
Theta1_grad(i,j)=Theta1_grad(i,j)+lambda*Theta1(i,j);
		end;
	end;
	for i=1:num_labels
		for j=2:k1+1
Theta2_grad(i,j)=Theta2_grad(i,j)+lambda*Theta2(i,j);
		end;
	end;
end;

Theta1_grad=Theta1_grad/m;
Theta2_grad=Theta2_grad/m;
J=J-sum;
sum1=0;
for i=1:k1
	for j=2:l1
		sum1=sum1+Theta1(i,j)^2;
	end;
end;
sum2=0;
for i=1:num_labels
	for j=2:k1+1
		sum2=sum2+Theta2(i,j)^2;
	end;
end; 
J=J+(sum1+sum2)*lambda/2;
J=J/m;


% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%



















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
