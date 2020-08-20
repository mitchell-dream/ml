function [mu sigma2] = estimateGaussian(X)
%ESTIMATEGAUSSIAN This function estimates the parameters of a 
%Gaussian distribution using the data in X
%   [mu sigma2] = estimateGaussian(X), 
%   The input X is the dataset with each n-dimensional data point in one row
%   The output is an n-dimensional vector mu, the mean of the data set
%   and the variances sigma^2, an n x 1 vector
% 

% Useful variables
[m, n] = size(X);

% You should return these values correctly
mu = zeros(n, 1);
sigma2 = zeros(n, 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the mean of the data and the variances
%               In particular, mu(i) should contain the mean of
%               the data for the i-th feature and sigma2(i)
%               should contain variance of the i-th feature.
%


%mu=mean(X,2);
%size=rows(X)
%columns(X) 返回列  rows(X) 返回行
%for i=1:size
%	mu(i)=mean(X(i));
	%fprintf('mu(i):  %f\n', mu(i));
	%sigma2(i)=mu(i);
%	Y=X(i,:)-mu(i);
	%fprintf('Y:  %e\n', Y);
%	sigma2(i)=1/(size-1) * sum(Y.^2);
	%fprintf('sigma2(i):  %f\n', sigma2(i));
%end


% 求每列的平均数
mu=mean(X);
% 求每列的方差
sigma2=var(X,1);


sigma2
fprintf('rows():  %f\n', rows(sigma2));
fprintf('columns():  %f\n', columns(sigma2));




% =============================================================


end
