%Ŀ�꺯��������ȫ����С
%ע������ʵ�����β�
function f = fun(X, Y, W_1, W_2, b_1, b_2)
[n, ~] = size(Y);
y = softmax(W_2 * ReLU(W_1 * X' + b_1) + b_2);
f = - sum(diag(Y * log(y))) / n;
end