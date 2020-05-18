%目标函数，求其全局最小
%注意区分实参与形参
function f = fun(X, Y, W_1, W_2, b_1, b_2)
[n, ~] = size(Y);
y = softmax(W_2 * ReLU(W_1 * X' + b_1) + b_2);
f = - sum(diag(Y * log(y))) / n;
end