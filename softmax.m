%softmax����������һ�����������ͬ����С�ľ���
function s = softmax(x)
[K, N] = size(x);
s = zeros(K, N);
for i=1:K
    for j = 1:N
        s(i, j) = exp(x(i, j)) / sum(exp(x(:, j)));%�������
    end
end
end