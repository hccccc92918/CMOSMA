function DCNSGAIII(Global)
% <algorithm> <C>
% Nondominated sorting genetic algorithm III

%------------------------------- Reference --------------------------------
% K. Deb and H. Jain, An evolutionary many-objective optimization algorithm
% using reference-point based non-dominated sorting approach, part I:
% Solving problems with box constraints, IEEE Transactions on Evolutionary
% Computation, 2014, 18(4): 577-601.
%------------------------------- Copyright --------------------------------
% Copyright (c) 2018-2019 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------
            cp                    = Global.ParameterSet(5);
            %% Generate the reference points and random population
            [Z,Global.N]         = UniformPoint(Global.N, Global.M);
            Population            = Global.Initialization();
            
            %% Calculate the initial dynamic constraint boundary
            [initialE, ~]         = max(max(0,Population.cons), [], 1);
            initialE(initialE==0) = 1;
            
            %% Optimization
            while Global.NotTermination(Population)
                % Reduce the dynamic constraint boundry
                epsn       = ReduceBoundary(initialE, ceil(Global.gen/Global.N), ceil(Global.maxgen/Global.N)-1, cp);
                % Mating selection which prefers to select epsilon-feasible solutions
                MatingPool = TournamentSelection(2,Global.N,sum(max(0, Population.cons-epsn), 2));
                Offspring  = GA(Population(MatingPool));
                % Update the reference points only consider the epsilon-feasible solution set
                Zmin       = min([Population(sum(max(0, Population.cons)<=epsn, 2)==size(Population.cons, 2)).objs; Offspring(sum(max(0,Offspring.cons)<=epsn, 2)==size(Offspring.cons, 2)).objs], [], 1);
                % Environment selection
                Population = EnvironmentalSelection([Population, Offspring], Global.N, Z, Zmin, initialE, epsn);
            end
end