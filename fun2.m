%��Ŀ�꺯��ת��Ϊ
%��softmax�Ľ��y��Ҳ������������
%ȡÿһ�У���Ӧÿ���������Ķ�Ӧ����λ�õ�ֵ�����
%����ʽ
function f = fun2(X, assistant_array, W_1, W_2, b_1, b_2)
[n, ~] = size(X);
y = softmax(W_2 * ReLU(W_1 * X' + b_1) + b_2);
tmp = 0;%��ʱ�������������
for i = 1:n
    tmp = tmp + log(y(assistant_array(i), i));
end
f = - tmp / n;
end