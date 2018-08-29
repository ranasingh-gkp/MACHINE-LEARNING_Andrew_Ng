function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);
X=[ones(m,1) X];
[k1,l1]=size(Theta1);
[k2,l2]=size(Theta2);
a2=zeros(k1+1,1);

a3=zeros(k2,1);
for i=1:m
	for j=1:k1
		sum=0;
		for k=1:l1
		sum=sum+Theta1(j,k)*X(i,k);
		end;
	a2(j+1,1)=sum;
	end;
	a2=sigmoid(a2);
	a2(1,1)=1;
	for j=1:k2
		sum=0;
		for k=1:l2
			sum=sum+Theta2(j,k)*a2(k,1);
		end;
		a3(j,1)=sum;
	end;
	a3=sigmoid(a3);
	ma=0;
	for j=1:num_labels
		if(a3(j,1)>ma)
			ma=a3(j,1);
			mark=j;
		end;
	end;
p(i,1)=mark;
end;

	

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%









% =========================================================================


end
