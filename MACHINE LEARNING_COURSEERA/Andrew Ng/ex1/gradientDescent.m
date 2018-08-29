function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);
temp=zeros(2,1);
temp(1,1)=theta(1,1);
temp(2,1)=theta(2,1);
for iter = 1:num_iters
		sum=0;
		for i=1:m
			h_theta=0;
			for j=1:2
				h_theta=h_theta+temp(j,1)*X(i,j);
			end;
		sum=sum+(h_theta-y(i))*X(i,1);
		end;
       theta(1,1)=theta(1,1)-sum*alpha/m;
		
	sum=0;
		for i=1:m
			h_theta=0;
			for j=1:2
				h_theta=h_theta+temp(j,1)*X(i,j);
			end;
		sum=sum+(h_theta-y(i))*X(i,2);
		end;
       theta(2,1)=theta(2,1)-sum*alpha/m;
		temp(2,1)=theta(2,1);
		temp(1,1)=theta(1,1);

			
    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %

    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);
	

end

end
