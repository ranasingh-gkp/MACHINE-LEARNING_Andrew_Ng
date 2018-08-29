function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%
% You need to return the following variables correctly.
set=zeros(8,1);
[m,n]=size(Xval);
b_sigma=0;
b_C=0;
m_k=9999;
x1 = [1 2 1]; x2 = [0 4 -1];
set =[0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];
for i=1:8
	C=set(i,1);
	for j=1:8
		sigma=set(j,1); 
	model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
		pred = svmPredict(model, Xval);
		k=mean(double(pred ~= yval));
		if(k<m_k)
			m_k=k;
			b_C=C;
			b_sigma=sigma;
		end;
	end;
end;		
C=b_C;
sigma=b_sigma;		
		
% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%







% =========================================================================

end
