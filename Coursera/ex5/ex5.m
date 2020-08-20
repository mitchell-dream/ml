%% Machine Learning Online Class
%  Exercise 5 | Regularized Linear Regression and Bias-Variance
%
%  Instructions
%  ------------
% 
%  This file contains code that helps you get started on the
%  exercise. You will need to complete the following functions:
%
%     linearRegCostFunction.m
%     learningCurve.m
%     validationCurve.m
%
%  For this exercise, you will not need to change any code in this file,
%  or any other files other than those mentioned above.
%

%% Initialization
clear ; close all; clc

%% =========== Part 1: Loading and Visualizing Data =============
%  We start the exercise by first loading and visualizing the dataset. 
%  The following code will load the dataset into your environment and plot
%  the data.
%

% Load Training Data
fprintf('Loading and Visualizing Data ...\n')

% Load from ex5data1:  读取数据
% You will have X, y, Xval, yval, Xtest, ytest in your environment
load ('ex5data1.mat');

% m = Number of examples
m = size(X, 1);
 
% Plot training data 画曲线
plot(X, y, 'rx', 'MarkerSize', 10, 'LineWidth', 1.5);
xlabel('Change in water level (x)');
ylabel('Water flowing out of the dam (y)');

fprintf('Program paused. Press enter to continue.\n');
pause;

%% =========== Part 2: Regularized Linear Regression Cost =============
%  You should now implement the cost function for regularized linear 
%  regression. 
% 实现代价函数 求 J 和 J 的偏导 grad
theta = [1 ; 1];
J = linearRegCostFunction([ones(m, 1) X], y, theta, 1);

% 预计代价为 303.993192
fprintf(['Cost at theta = [1 ; 1]: %f '...
         '\n(this value should be about 303.993192)\n'], J);

fprintf('Program paused. Press enter to continue.\n');
pause;

%% =========== Part 3: Regularized Linear Regression Gradient =============
%  You should now implement the gradient for regularized linear 
%  regression.
%  正则化 求 J 和 J 的偏导 grad

theta = [1 ; 1];
[J, grad] = linearRegCostFunction([ones(m, 1) X], y, theta, 1);


% 预计梯度为 [-15.303016; 598.250744]
fprintf(['Gradient at theta = [1 ; 1]:  [%f; %f] '...
         '\n(this value should be about [-15.303016; 598.250744])\n'], ...
         grad(1), grad(2));

fprintf('Program paused. Press enter to continue.\n');
pause;


%% =========== Part 4: Train Linear Regression =============
%  Once you have implemented the cost and gradient correctly, the
%  trainLinearReg function will use your cost function to train 
%  regularized linear regression.
% 
%  Write Up Note: The data is non-linear, so this will not give a great 
%                 fit.
% 
% 使用线性回归函数你和样本数据 λ = 0 求 θ
%  Train linear regression with lambda = 0  
lambda = 0;
[theta] = trainLinearReg([ones(m, 1) X], y, lambda);

%  Plot fit over the data 画出拟合后的曲线
plot(X, y, 'rx', 'MarkerSize', 10, 'LineWidth', 1.5);
xlabel('Change in water level (x)');
ylabel('Water flowing out of the dam (y)');
hold on;
plot(X, [ones(m, 1) X]*theta, '--', 'LineWidth', 2)
hold off;

fprintf('Program paused. Press enter to continue.\n');
pause;


%% =========== Part 5: Learning Curve for Linear Regression =============
%  Next, you should implement the learningCurve function. 
%
%  Write Up Note: Since the model is underfitting the data, we expect to
%                 see a graph with "high bias" -- Figure 3 in ex5.pdf 
%

% 画出学习误差曲线（高偏差） ，观察是过拟合还是欠拟合
lambda = 0;
[error_train, error_val] = ...
    learningCurve([ones(m, 1) X], y, ...
                  [ones(size(Xval, 1), 1) Xval], yval, ...
                  lambda);

plot(1:m, error_train, 1:m, error_val);
title('Learning curve for linear regression')
legend('Train', 'Cross Validation')
xlabel('Number of training examples')
ylabel('Error')
axis([0 13 0 150])

fprintf('# Training Examples\tTrain Error\tCross Validation Error\n');
for i = 1:m
    fprintf('  \t%d\t\t%f\t%f\n', i, error_train(i), error_val(i));
end

fprintf('Program paused. Press enter to continue.\n');
pause;

%% =========== Part 6: Feature Mapping for Polynomial Regression =============
%  One solution to this is to use polynomial regression. You should now
%  complete polyFeatures to map each example into its powers
%

p = 8;


% 增加特征数
% Map X onto Polynomial Features and Normalize
X_poly = polyFeatures(X, p);
[X_poly, mu, sigma] = featureNormalize(X_poly);  % Normalize
X_poly = [ones(m, 1), X_poly];                   % Add Ones


% 特征归一化(特征缩放)
% Map X_poly_test and normalize (using mu and sigma)
X_poly_test = polyFeatures(Xtest, p);
X_poly_test = bsxfun(@minus, X_poly_test, mu);
X_poly_test = bsxfun(@rdivide, X_poly_test, sigma);
X_poly_test = [ones(size(X_poly_test, 1), 1), X_poly_test];         % Add Ones

% Map X_poly_val and normalize (using mu and sigma)
X_poly_val = polyFeatures(Xval, p);
X_poly_val = bsxfun(@minus, X_poly_val, mu);
X_poly_val = bsxfun(@rdivide, X_poly_val, sigma);
X_poly_val = [ones(size(X_poly_val, 1), 1), X_poly_val];           % Add Ones

fprintf('Normalized Training Example 1:\n');
fprintf('  %f  \n', X_poly(1, :));

fprintf('\nProgram paused. Press enter to continue.\n');
pause;



%% =========== Part 7: Learning Curve for Polynomial Regression =============
%  Now, you will get to experiment with polynomial regression with multiple
%  values of lambda. The code below runs polynomial regression with 
%  lambda = 0. You should try running the code with different values of
%  lambda to see how the fit and learning curve change.
%


% 经过特征缩放后 再次进行训练
lambda = 0;
[theta] = trainLinearReg(X_poly, y, lambda);


% 线性回归最优学习参数 θ
% Plot training data and fit
figure(1);
plot(X, y, 'rx', 'MarkerSize', 10, 'LineWidth', 1.5);
plotFit(min(X), max(X), mu, sigma, theta, p);
xlabel('Change in water level (x)');
ylabel('Water flowing out of the dam (y)');
title (sprintf('Polynomial Regression Fit (lambda = %f)', lambda));

% 绘制误差曲线，查看过拟合或欠拟合
% 這邊值得注意的一點是,繪製出來的結果很可能會跟作業的pdf檔中給的答案不一樣
% 根據課程中的論壇說明
% 可能是數值精度導致的問題
% 因為這次作業中,特徵映射到x的8次方,實際上其實很少會取到這麼大的次方數
% 其次是訓練集的數量過少
% 基於這些理由才導致這樣的問題
% 此外,使用的是matlab還是octave,還有使用的版本等等,都可能在這樣的狀況下出現不同的差異
% (其實在polyFeatures.m,用的是.^還是和前一行.*答案就不同了)
% 不過歸根究柢皆是因為8次方的映射和過少的訓練集所導致
% 這也是在正式使用時不該會發生的狀況
% 所以其實最後submit有過的話就也不用太在意這邊的結果跟pdf的標準答案不同就是了
figure(2);
[error_train, error_val] = ...
    learningCurve(X_poly, y, X_poly_val, yval, lambda);
plot(1:m, error_train, 1:m, error_val);

title(sprintf('Polynomial Regression Learning Curve (lambda = %f)', lambda));
xlabel('Number of training examples')
ylabel('Error')
axis([0 13 0 100])
legend('Train', 'Cross Validation')

fprintf('Polynomial Regression (lambda = %f)\n\n', lambda);
fprintf('# Training Examples\tTrain Error\tCross Validation Error\n');
for i = 1:m
    fprintf('  \t%d\t\t%f\t%f\n', i, error_train(i), error_val(i));
end

fprintf('Program paused. Press enter to continue.\n');
pause;

%% =========== Part 8: Validation for Selecting Lambda =============
%  You will now implement validationCurve to test various values of 
%  lambda on a validation set. You will then use this to select the
%  "best" lambda value.
%

% 自动选择 lambda
[lambda_vec, error_train, error_val] = ...
    validationCurve(X_poly, y, X_poly_val, yval);

close all;
plot(lambda_vec, error_train, lambda_vec, error_val);
legend('Train', 'Cross Validation');
xlabel('lambda');
ylabel('Error');

fprintf('lambda\t\tTrain Error\tValidation Error\n');
for i = 1:length(lambda_vec)
	fprintf(' %f\t%f\t%f\n', ...
            lambda_vec(i), error_train(i), error_val(i));
end

fprintf('Program paused. Press enter to continue.\n');
pause;


%% =========== Part 9: Computing test set error =============
% bestEVal是最小值,setLidx是第幾項(之前的作業已經用過的方法)
[bestEVal, setLidx] = min(error_val);
bestTheta = trainLinearReg(X_poly, y, lambda_vec(setLidx));
lambda = 3
J = linearRegCostFunction(X_poly_test, ytest, bestTheta, 0);
fprintf('J Suppose to be 3.8599 (J = %f)\n\n', J);
fprintf('\n');
fprintf('Program paused. Press enter to continue.\n');
pause;


%% =========== Part 10: Plotting learning curves with randomly selected examples =============
%1、在训练集和交叉验证集中随机选出 i 个实例。
%2、在随机选择的训练集中训练 θ
%3、然后用训练出的 θ 在随机选择的训练集和交叉验证集上评估
%4、以上的步骤需要重复多次（50次）然后应该使用其名均值来确定 训练集的错误率 和 交叉验证集的错误率
%5、供参考,图10显示了学习曲线我们获得了多项式回归和λ= 0.01。由于随机选择的例子，您的数字可能略有不同。

lambda = 0.01;
exm = size(X_poly, 1);
exvalm = size(X_poly_val, 1);
errTra = zeros(exm,1);
errVal = zeros(exm,1);
for i=1:exm
    for j=1:50
        randomIdx=randperm(exm);
        randomvalIdx=randperm(exvalm);
        rantheta = trainLinearReg(X_poly(randomIdx(1:i), :), y(randomIdx(1:i)), lambda);
        errTra(i) = errTra(i) + ...
            linearRegCostFunction(X_poly(randomIdx(1:i), :), y(randomIdx(1:i)), rantheta, 0);
        errVal(i) = errVal(i) + ...
            linearRegCostFunction(X_poly_val(randomvalIdx(1:i), :), yval(randomvalIdx(1:i)), rantheta, 0);
    endfor
    errTra(i) = errTra(i)/50;
    errVal(i) = errVal(i)/50;
endfor

% plot(1:exm, errTra, 1:exm, errVal);
%plot(1:exm, errTra, 1:exm, errVal);

title(sprintf('Polynomial Regression Learning Curve (lambda = %f)', lambda));
xlabel('Number of training examples')
ylabel('Error')
axis([0 13 0 100])
legend('Train', 'Cross Validation')

fprintf('Polynomial Regression (lambda = %f)\n\n', lambda);
fprintf('# Training Examples\tTrain Error\tCross Validation Error\n');
for i = 1:m
    fprintf('  \t%d\t\t%f\t%f\n', i, errTra(i), errVal(i));
end

fprintf('Program paused. Press enter to continue.\n');



