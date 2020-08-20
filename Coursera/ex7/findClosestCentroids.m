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
distance=zeros(size(X,1),K);

% ====================== YOUR CODE HERE ======================
% Instructions: Go over every example, find its closest centroid, and store
%               the index inside idx at the appropriate location.
%               Concretely, idx(i) should contain the index of the centroid
%               closest to example i. Hence, it should be a value in the 
%               range 1..K
%
% Note: You can use a for-loop over the examples to compute this.
%


%for i = 1:size(X,1)
	% sum(A,1)        %表示对每一列求所有元素的和
    % sum(A,2)        %表示对每一行求所有元素的和
	% 利用for迴圈,計算X各個資料和每個群心的差值(格式會是1xn - kxn,會自動對應成kxn的結果)
  	% 再各項取平方後列加總,就能得到第i項資料和各群心的歐式距離的平方
  %	K = sum((X(i,:) - centroids).^2, 2);
  	% 再找出哪一個群心距離最近(值最小),存進idx即可
 % 	[value, idx(i)] = min(K);
%endfor

for i= 1:K
  D = bsxfun(@minus, X, centroids(i,:));
  distance(:,i) =sum(D.^2,2);
end

[minValues, idx] = min(distance,[],2);


% =============================================================

end

