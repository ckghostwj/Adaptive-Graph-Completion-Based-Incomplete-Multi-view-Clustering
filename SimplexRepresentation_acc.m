% An out-of-date version, just for reference

% J. Huang, F. Nie, and H. Huang, A new simplex sparse learning model to measure data similarity for clustering [C]. 
% In: Proceedings of the International Conference on Artificial Intelligence, 2015, 3569-3575.
%  min  || Ax - y||^2
%  s.t. x>=0, 1'x=1
function [x obj]=SimplexRepresentation_acc(A, y, x0)


NIter = 500;
NStop = 20;

[dim n] = size(A);              % һ��һ������
%AA = A'*A; 
Ay = A'*y;
if nargin < 3
    x = 1/n*ones(n,1);          % ��ʼ����ʾϵ��
else
    x = x0;
end;

x1 = x;
t = 1;
t1 = 0;
r = 0.5;                                        % ��ʽ�еĳ�ʼֵL
%obj = zeros(NIter,1);
for iter = 1:NIter
    p = (t1-1)/t;                               % ��ʽ15�еĵ�������
    s = x + p*(x-x1);                           % ��ʽ15
    x1 = x;                                     % x1��ǰһ����ϵ��ֵ
    g = A'*(A*s) - Ay; % g = AA*s - Ay;         % �ݶ�
    ob1 = norm(A*x - y);
    for it = 1:NStop
        z = s - r*g;                            % ��ʽ13�е�v
        z = EProjSimplex(z);                    % ��ʽ13
        ob = norm(A*z - y);
        if ob1 < ob
            r = 0.5*r;                          % ��̩��ϵ��L��������Ӧѡȡ
        else
            break;
        end;
    end;
    if it == NStop
        obj(iter) = ob;
        %disp('not');
        break;
    end;
    x = z;
    t1 = t;                     % ��һ����t
    t = (1+sqrt(1+4*t^2))/2;
    
    
    obj(iter) = ob;
end
   
1;
    
