%将目标函数转换为
%对softmax的结果y，也即神经网络的输出
%取每一列（对应每个样本）的对应分类位置的值再求和
%的形式
function f = fun2(X, assistant_array, W_1, W_2, b_1, b_2)
[n, ~] = size(X);
y = softmax(W_2 * ReLU(W_1 * X' + b_1) + b_2);
tmp = 0;%临时变量，用于求和
for i = 1:n
    tmp = tmp + log(y(assistant_array(i), i));
end
f = - tmp / n;
end