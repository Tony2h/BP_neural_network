%��fun2�Ļ����ϣ���log��softmax������ϣ�����������
function f = fun3(X, assistant_array, W_1, W_2, b_1, b_2)
[n, ~] = size(X);
tmp1 = W_2 * ReLU(W_1 * X' + b_1) + b_2;%��ʱ������Ϊsoftmax��������
%����ʵ֪����softmax����ľ����У�������Щλ�õ�Ԫ�ػᱻ���ڼ�����ʧ����
%��������λ�õ�Ԫ�ز�����Ҫ���м���
%��������softmax��������������ȫ����Ҫ�ģ�Ҳ����tmp1�д洢������
tmp2 = zeros(n, 1);
for i = 1:n
    tmp2(i) = tmp1(assistant_array(i), i) - ...
        log( sum( exp( tmp1(:, i) ) ) );
end
f = - sum(tmp2) / n;
end