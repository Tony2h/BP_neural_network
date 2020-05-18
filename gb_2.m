%对b_2的偏导数
function gb_2 = gb_2(X, Y, W_1, W_2, b_1, b_2)
[n, ~] = size(X);
tmp1 = W_2 * ReLU(W_1 * X' + b_1) + b_2;%临时变量，为softmax函数输入，size为l*n
gb_2 = sum(Y, 1)';
%sum()函数的参数1表示矩阵的每列元素求和得到一个行向量，当前只完成了第一部分求导，gb_2是一个k维列向量
for i = 1:n
    gb_2 = gb_2 - exp(tmp1(:, i)) ./ sum(exp(tmp1(:, i)));
end
gb_2 = -1/n * gb_2;
end