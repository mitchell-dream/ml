function [X_norm, mu, sigma] = featureNormalize(X)
%FEATURENORMALIZE Normalizes the features in X 
%   FEATURENORMALIZE(X) returns a normalized version of X where
%   the mean value of each feature is 0 and the standard deviation
%   is 1. This is often a good preprocessing step to do when
%   working with learning algorithms.


% mu = mean(X) 返回沿数组中不同维（每列）的元素的平均值   mu = mean(X,2); 返回矩阵中每一行的均值
mu = mean(X);

% bsxfun(@function,A,B): 根据function的不同对矩阵A和B 进行运算： X 中每个元素 - mu
X_norm = bsxfun(@minus, X, mu);

%  std (x, flag,dim) 计算每一列/行的标准差  
% flag==0.........是除以n－1 
% flag==1.........是除以n  dim表示维数
% dim==1..........是按照列分
% dim==2..........是按照行分
% 若是三维的矩阵，dim=＝3就按照第三维来分数据

% 求每列标准差
sigma = std(X_norm);
% X 特征 除以标准差
X_norm = bsxfun(@rdivide, X_norm, sigma);


% ============================================================

end
