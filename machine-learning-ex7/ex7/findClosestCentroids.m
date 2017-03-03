function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1 
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

% Set K
K = size(centroids, 1);

% You need to return the following variables correctly.
idx = zeros(size(X,1), 1);

sz = size(X,1);
sz1 = size(centroids,1);

%size(centroids)
%size(X)
%size(idx)
%size(X)
%sz
%Loop over every example X
for j=1:sz
    
    dist = zeros(K,1);
    %dist
    for k=1:sz1
        dist(k) = sum((X(j,:)-centroids(k,:)).^2);
        
        %((X(j,1)-centroids(k,1))^2+(X(j,2)-centroids(k,2))^2);
    end
    
    %dist 
    %find(dist == min(dist))
    %fprintf('HERE\n')
    %size(dist)
    %idx(j)
    %find(dist == min(dist))
    %min(dist)
    minindx=find(dist == min(dist));
    %j
    minindx = minindx(1);
    %if (ndims(minindx) > 1)
    %    ndims(minindx)
    %    minindx
    %    minindx=minindx(1)
    %end
    
    %fprintf('Yep Above\n');
    %if (size(idx(j)) > size(find(dist == min(dist))))
    %    pause;
    %    end
    %size(minindx)
    idx(j) = minindx;
end


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

