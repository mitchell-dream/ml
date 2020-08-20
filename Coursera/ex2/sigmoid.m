function g = sigmoid(z)
%SIGMOID Compute sigmoid function
%   g = SIGMOID(z) computes the sigmoid of z.

% You need to return the following variables correctly 
g = zeros(size(z));
%g=1/ (1+exp(-z));
% 1、exp 代表 e，/ 后面这里可能是一个矩阵，如果是矩阵运算的话，需要添加 .+ 运算符，后面的运算都依赖这个函数
g = 1./(exp(-z)+1)

end
