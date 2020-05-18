clc;
clear;
%% 读取数据
%100样本，400特征，10分类
data = csvread("mnist_test.csv");
n = 1000;%样本数
m = 784;%特征数
k = 10;%分类数
l = 50;%隐层神经元数
X = data(1:n, 2:m+1);%样本特征数据

assistant_array = data(1:n, 1) + 1;%辅助数组，记录每个样本的分类
Y = zeros(n, k); 
for i = 1:n
    Y(i, assistant_array(i)) = 1;%标签，1-10分类，Y的第i行的某个位置是1表示第i个样本是该位置对应的分类
end

%初始化系数矩阵和常数向量
W_1 = randn(l, m);
b_1 = zeros(l, 1);
W_2 = randn(k, l);
b_2 = rands(k, 1);

%X归一化
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
