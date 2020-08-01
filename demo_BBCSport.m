% Code for our TMM paper:
% Jie Wen, Ke Yan, Zheng Zhang, Yong Xu, Junqian Wang, Lunke Fei, Bob Zhang,
% Adaptive Graph Completion Based Incomplete Multi-view Clustering [J],
% IEEE Transactions on Multimedia, 2020.
% written by Jerry (Email: jiewen_pr@126.com)
% If you used the code, please cite the above papers.
% More codes on Incomplete multi-view clustering are released at:  
% https://sites.google.com/view/jerry-wen-hit/publications
clear all
clc

Dataname = 'bbcsport4vbigRnSp'
percentDel = 0.3
Datafold = [Dataname,'_percentDel_',num2str(percentDel),'.mat'];
load(Dataname);
load(Datafold);

lambda1 = 10
lambda3 = 100
f = 1
r = 2
ind_folds = folds{f};
load(Dataname);
truthF = truth;   % 真实类标
clear truth
numInst = length(truthF);
num_cluster = length(unique(truthF));
for iv = 1:length(X)
    X1 = X{iv};     % bbc这里和其他库有区别 转置和不转置
    X1 = NormalizeFea(X1,0);    % 0为列归一化
    ind_0 = find(ind_folds(:,iv) == 0);  % indexes of misssing instances
    X1(:,ind_0) = [];           % 去除缺失视角  
    % -------------- 图初始化 ----------------- %
    options = [];
    options.NeighborMode = 'KNN';
    options.k = 11;
    options.WeightMode = 'Binary';       % Binary
    So{iv} = constructW(X1',options);                  % 论文提出的 图初始化方法    
    G = diag(ind_folds(:,iv));
    G(:,ind_0) = [];
    Sor{iv} = G*So{iv}*G';
end
clear X X1 ind_0 G So

max_iter = 50
% % ---------------- 初始化 U ------------- %
LSv = 0;
alpha = ones(length(Sor),1)/length(Sor);
alpha_r = alpha.^r;
for iv = 1:length(Sor)
    linshi_S = 0.5*(Sor{iv}+Sor{iv}');
    LSv = LSv + (diag(sum(linshi_S))-linshi_S)*alpha_r(iv);
end
[U_ini, ~, ev]=eig1(LSv, num_cluster, 0);
clear LSv ev linshi_S
[Nsor,U,obj] = AGC_IMC(Sor,U_ini,r,num_cluster,numInst,lambda1,lambda3,max_iter,ind_folds);
new_F = U;
norm_mat = repmat(sqrt(sum(new_F.*new_F,2)),1,size(new_F,2));
for i = 1:size(norm_mat,1)
    if (norm_mat(i,1)==0)
        norm_mat(i,:) = 1;
    end
end
new_F = new_F./norm_mat; 
pre_labels    = kmeans(real(new_F),num_cluster,'emptyaction','singleton','replicates',20,'display','off');
result_cluster = ClusteringMeasure(truthF, pre_labels)*100       

 