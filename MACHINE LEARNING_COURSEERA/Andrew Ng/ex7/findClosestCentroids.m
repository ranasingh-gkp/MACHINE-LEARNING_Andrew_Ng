function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1 
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

% Set K
K = size(centroids, 1);
m=length(X(:,1));
z=zeros(length(X(1,:)),1);
% You need to return the following variables correctly.
idx = zeros(size(X,1), 1);
min=9999;
for i=1:m
	z=X(i,:);
	dist=0;
	for j=1:K
		for k=1:length(z)
			dist=dist+(X(i,k)-centroids(j,k))^2;
		end;
	if(min>dist)
		min=dist;
		idx(i)=j;
	end;
	dist=0;
	end;
	min=9999;
end;
% ====================== YOUR CODE HERE ======================
% Instructions: Go over every example, find its closest centroid, and store
%               the index inside idx at the appropriate location.
%               Concretely, idx(i) should contain the index of the centroid
%               closest to example i. Hence, it should be a value in the 
%               range 1..K
%
% Note: You can use a for-loop over the examples to compute this.
%







% =============================================================

end

