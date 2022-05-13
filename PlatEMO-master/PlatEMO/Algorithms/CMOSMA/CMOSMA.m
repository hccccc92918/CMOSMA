function CMOSMA(Global)
% <algorithm> <C>
% D    ---     --- Number of neurons in each dimension of the latent space
% tau0 --- 0.7 --- Initial learning rate
% H    ---   5 --- Size of neighborhood mating pools
% A self-organizing map approach for constrained multi-objective optimization problems
%------------------------------- R  eference --------------------------------
% C. He, M. Li, C. Zhang et al,A self-organizing map approach for constrained multi-objective
% optimization problems, Complex & Intelligent Systems,2022.
%--------------------------------------------------------------------------
    %% Parameter setting
    [D,tau0,H] = Global.ParameterSet(repmat(ceil(Global.N.^(1/(Global.M-1))),1,Global.M-1),0.7,5);
    D1=D;
    Global.N   = prod(D);
    sigma0     = sqrt(sum(D.^2)/(Global.M-1))/2;
    
    %% Generate random population
    FP = Global.Initialization();
    AP = Global.Initialization();
    
    %% Initialize the SOM1 
    % Training set
    S = FP.decs;
    % Weight vector of each neuron
    W = S;
    [LDis,B]=Initialize_SOM(S,D,H);

    %% Initialize the SOM2
    % Training set
    S2 = AP.decs;
    % Weight vector of each neuron
    W2 = S2;
    [LDis2,B2]=Initialize_SOM(S2,D1,H);
   
    %% Optimization
    while Global.NotTermination(FP)
        % Update SOM1
        W = UpdateSOM(S,W,Global.evaluated,Global.evaluation,LDis,sigma0,tau0);
        W2 = UpdateSOM(S2,W2,Global.evaluated,Global.evaluation,LDis2,sigma0,tau0);      
        % Associate each solution with a neuron
        [XU] = Associate(FP,W,Global.N);
        [XU2] = Associate(AP,W2,Global.N); 
        %Construct  matingPool
        [MatingPool1] = MatingPool(XU,Global.N,B);
        [MatingPool2] = MatingPool(XU2,Global.N,B2);
        % Evolution
        A1 = FP.decs;
        Offspring1  =GA([FP,FP(MatingPool1)]);%GA
        A2 = AP.decs;
        Offspring2  =GA([AP,AP(MatingPool2)]);%GA
        [FP] = EnvironmentalSelection([FP,Offspring1,Offspring2],Global.N,true);
        [AP] = EnvironmentalSelection([AP,Offspring1,Offspring2],Global.N,false);
         % Update the training set
        S = setdiff(FP.decs,A1,'rows');
        S2 = setdiff(AP.decs,A2,'rows'); 
    end
end
