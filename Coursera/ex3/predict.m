function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

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



% size（X，1） 获取x 的行数，ones(m,n) 生成一个m行 n 列，并且所有元素都为 1 的矩阵
% [ones(size(X, 1), 1) X] X 添加上生成的元素都为 1 的矩阵
X = [ones(size(X, 1), 1) X];
H1 = sigmoid(X * Theta1');
H1 = [ones(size(H1,1),1) H1];
H2 = sigmoid(H1 *Theta2');
% [x, ix] = max ([1, 3, 5, 2, 5])  x = 5 ix = 3  x 返回最大值，ix 返回最大值所在位置
[val, index] = max(H2,[],2);
p = index;




% =========================================================================


end
