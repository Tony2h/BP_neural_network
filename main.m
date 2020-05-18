clc;
clear;
%% ��ȡ����
%100������400������10����
data = csvread("mnist_test.csv");
n = 1000;%������
m = 784;%������
k = 10;%������
l = 50;%������Ԫ��
X = data(1:n, 2:m+1);%������������

assistant_array = data(1:n, 1) + 1;%�������飬��¼ÿ�������ķ���
Y = zeros(n, k); 
for i = 1:n
    Y(i, assistant_array(i)) = 1;%��ǩ��1-10���࣬Y�ĵ�i�е�ĳ��λ����1��ʾ��i�������Ǹ�λ�ö�Ӧ�ķ���
end

%��ʼ��ϵ������ͳ�������
W_1 = randn(l, m);
b_1 = zeros(l, 1);
W_2 = randn(k, l);
b_2 = rands(k, 1);

%X��һ��
X = mapminmax(X, 0, 1);
%% main code
kmax = 1000;
eps = 1e-3;

disp(correct(X, assistant_array, W_1, W_2, b_1, b_2));

[W_1, b_1, W_2, b_2,f,iter,time] = ...
    steepest_descent(W_1, b_1, W_2, b_2, X, Y, assistant_array, kmax,eps);

disp(correct(X, assistant_array, W_1, W_2, b_1, b_2))



%% check the correctness of the gradient function of the parameters.

% disp(fun(X, Y, W_1, W_2, b_1, b_2));
% disp(fun2(X, assistant_array, W_1, W_2, b_1, b_2));
% disp(fun3(X, assistant_array, W_1, W_2, b_1, b_2));
% 
% W_1 = ones(4, 2);
% b_1 = [1; 2; 3; 4];
% W_2 = ones(2, 4);W_2(2, 4) = 0;
% b_2 = [1; 2];
% 
% X = [1, -1; -1, -1];
% Y = [1, 0; 0, 1];
% 
% assistant_array = [1, 2];
% 
% gb_2(X, Y, W_1, W_2, b_1, b_2);%bingo
% gW_2(X, Y, W_1, W_2, b_1, b_2);%bingo
% gb_1(X, assistant_array, W_1, W_2, b_1, b_2);%bingo
% gW_1(X, assistant_array, W_1, W_2, b_1, b_2);
