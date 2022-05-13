function W = UpdateSOM(S,W,evaluated,evaluation,LDis,sigma0,tau0)
% Update SOM
%------------------------------- Copyright --------------------------------
% Copyright (c) 2018-2019 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    for s = 1 : size(S,1)
            sigma  = sigma0*(1-(evaluated+s)/evaluation);
            tau    = tau0*(1-(evaluated+s)/evaluation);
            [~,u1] = min(pdist2(S(s,:),W));
            U      = LDis(u1,:) < sigma;
            W(U,:) = W(U,:) + tau.*repmat(exp(-LDis(u1,U))',1,size(W,2)).*(repmat(S(s,:),sum(U),1)-W(U,:));
    end
end