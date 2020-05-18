%在fun2的基础上，将log与softmax函数结合，方便求导推演
function f = fun3(X, assistant_array, W_1, W_2, b_1, b_2)
[n, ~] = size(X);
tmp1 = W_2 * ReLU(W_1 * X' + b_1) + b_2;%临时变量，为softmax函数输入
%我其实知道，softmax输出的矩阵中，究竟哪些位置的元素会被用于计算损失函数
%所以其它位置的元素并不需要进行计算
%但是整个softmax的输入矩阵参数是全都需要的，也就是tmp1中存储的内容
tmp2 = zeros(n, 1);
for i = 1:n
    tmp2(i) = tmp1(assistant_array(i), i) - ...
        log( sum( exp( tmp1(:, i) ) ) );
end
f = - sum(tmp2) / n;
end