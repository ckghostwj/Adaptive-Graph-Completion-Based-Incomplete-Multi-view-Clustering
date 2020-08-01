% Code for our TMM paper:
% Jie Wen, Ke Yan, Zheng Zhang, Yong Xu, Junqian Wang, Lunke Fei, Bob Zhang,
% Adaptive Graph Completion Based Incomplete Multi-view Clustering [J],
% IEEE Transactions on Multimedia, 2020.
% written by Jerry (Email: jiewen_pr@126.com)
% If you used the code, please cite the above papers.
% More codes on Incomplete multi-view clustering are released at:  
% https://sites.google.com/view/jerry-wen-hit/publications
function [Sor,U,obj] = AGC_IMC(Sor_ini,U_ini,r,num_cluster,numInst,lambda1,lambda3,max_iter,ind_folds)

So = Sor_ini;
Sor = Sor_ini;
U = U_ini;

% % -------- 根据 Sr 初始化表示系数矩阵A ---------- %
A = rand(length(Sor),length(Sor));   % 初始化所有视角的重构稀疏权重为1
A = A-diag(diag(A)); % 去除自表示的影响
for m = 1:length(Sor)
    indx = [1:length(Sor)];
    indx(m) = [];
    A(indx',m) = (ProjSimplex(A(indx',m)'))'; % 将A的每一列按照列和归一化
end
alpha = ones(length(Sor),1)/length(Sor);
alpha_r = alpha.^r;

for iter = 1:max_iter
    if iter == 1
        Sor = Sor_ini;
    end
    % ---------------- U ------------- %
    LSv = 0;
    for iv = 1:length(Sor)
        linshi_S = 0.5*(Sor{iv}+Sor{iv}');
        LSv = LSv + (diag(sum(linshi_S))-linshi_S)*alpha_r(iv);
    end
    [U, ~, ~] = eig1(LSv, num_cluster, 0);
    clear LSv linshi_S 
    
    % ---------------- A (B)---------------- %
    vec_S = [];
    for iv = 1:length(Sor)
        vec_S = [vec_S,(Sor{iv}(:))];
    end    

    for iv = 1:length(Sor)
        indv = [1:length(Sor)];
        indv(iv) = [];
        [A(indv',iv),~] = SimplexRepresentation_acc(vec_S(:,indv), vec_S(:,iv));
        %  min  || Ax - y||^2
        %  s.t. x>=0, 1'x=1
    end 
    % ---------- alpha_v ------------- %
    Rec_error = zeros(1,length(Sor));
    vec_S = [];
    for iv = 1:length(Sor)
        vec_S = [vec_S,(Sor{iv}(:))];
    end  
    for iv = 1:length(Sor)
        % ------- obj reconstructed error ------------ %
        W = ones(numInst,numInst);
        ind_0 = find(ind_folds(:,iv) == 0);  % indexes of misssing instances
        W(:,ind_0) = 0;
        W(ind_0,:) = 0;
        linshi_S = 0.5*(Sor{iv}+Sor{iv}');
        LSv = diag(sum(linshi_S))-linshi_S;
        
        Rec_error(iv) = norm((Sor{iv}-So{iv}).*W,'fro')^2+lambda3*trace(U'*LSv*U)+lambda1*norm(vec_S(:,iv)-vec_S*A(:,iv))^2;
    end
    clear Linshi_S LSv W 
    H = bsxfun(@power,Rec_error, 1/(1-r));     % h = h.^(1/(1-r));
    alpha = bsxfun(@rdivide,H,sum(H)); % alpha = H./sum(H);
    alpha_r = alpha.^r; 
    clear H
    % ------------------ Sr{iv} ------------------ %
    Z = L2_distance_1(U',U');
    vec_S = [];
    for iv = 1:length(Sor)
        vec_S = [vec_S,(Sor{iv}(:))];
    end
    for iv = 1:length(Sor)
        W = ones(numInst,numInst);
        ind_0 = find(ind_folds(:,iv) == 0);  % indexes of misssing instances
        W(:,ind_0) = 0;
        W(ind_0,:) = 0;
        M_iv = vec_S*A(:,iv);
        M_iv = reshape(M_iv,numInst,numInst);
        sum_Y = 0;
        coeef = 0;
        for iv2 = 1:length(Sor)
            if iv2 ~= iv
                Y_iv2 = vec_S(:,iv2)-vec_S*A(:,iv2)+A(iv,iv2)*vec_S(:,iv);
                sum_Y = sum_Y + alpha_r(iv2)*A(iv,iv2)*lambda1*Y_iv2;
                coeef = coeef +  A(iv,iv2)^2*alpha_r(iv2);
                clear Y_iv2
            end
        end
        clear iv2
        matrix_sum_Y = reshape(sum_Y,numInst,numInst);
        clear sum_Y
        Linshi_L = (alpha_r(iv)*Sor{iv}.*W+alpha_r(iv)*lambda1*M_iv-0.25*lambda3*Z*alpha_r(iv)+matrix_sum_Y)./(alpha_r(iv)*W+lambda1*alpha_r(iv)+coeef*lambda1);
        for num = 1:numInst
            indnum = [1:numInst];
            indnum(num) = [];
            Sor{iv}(indnum',num) = (EProjSimplex_new(Linshi_L(indnum',num)'))';
        end
        clear Linshi_L matrix_sum_Y coeef
    end
    clear vec_S 
    
    % --------------  obj -------------- %
    vec_S = [];
    for iv = 1:length(Sor)
        vec_S = [vec_S,(Sor{iv}(:))];
    end
    for iv = 1:length(Sor)
        % ------- obj reconstructed error ------------ %
        W = ones(numInst,numInst);
        ind_0 = find(ind_folds(:,iv) == 0);  % indexes of misssing instances
        W(:,ind_0) = 0;
        W(ind_0,:) = 0;
        linshi_S = 0.5*(Sor{iv}+Sor{iv}');
        LSv = diag(sum(linshi_S))-linshi_S;
        
        Rec_error(iv) = norm((Sor{iv}-So{iv}).*W,'fro')^2+lambda3*trace(U'*LSv*U)+lambda1*norm(vec_S(:,iv)-vec_S*A(:,iv))^2;
    end
    clear W ind_0 linshi_S LSv
    obj(iter) = alpha_r*Rec_error';
    clear vec_S
    if iter > 2 && abs(obj(iter)-obj(iter-1))<1e-6
        iter
        break;
    end     
end
end
