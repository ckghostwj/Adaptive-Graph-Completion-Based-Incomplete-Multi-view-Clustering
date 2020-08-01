function S =RelevantViews(L)
% function to find 
%       min_S   \|L-SL\|^2_2
%       s.t.    diag(S)=0
%               \|S(i,:)\| \le 1
%               S(i,j)\ge

[M,N2]=size(L);
S=zeros(M,M);
for m=1:1:M
   
    H =[];
    f=[];
    TempL=L([1:m-1,m+1:M],:);
    H= TempL*TempL';
    f=-TempL*L(m,:)';
   
    [tempS, FVAL,optv] = quadprog(H,f,[],[],ones(1,M-1),1,zeros(M-1,1),[]);
    S(m,[1:m-1,m+1:M])=tempS;
end
    
end
