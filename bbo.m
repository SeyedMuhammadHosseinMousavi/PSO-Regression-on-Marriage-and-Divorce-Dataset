function [x,err,cost]=bbo(CostFunction,nVar)

%% Problem 
VarSize = [1 nVar];   % Decision Variables Matrix Size
VarMin = -5;         % Decision Variables Lower Bound
VarMax = 5;         % Decision Variables Upper Bound
%% BBO 
MaxIt = 100;          % Maximum Number of Iterations
nPop = 20;            % Number of Habitats (Population Size)

KeepRate = 0.2;                   % Keep Rate
nKeep = round(KeepRate*nPop);     % Number of Kept Habitats
nNew = nPop-nKeep;                % Number of New Habitats
% Migration Rates
mu = linspace(1, 0, nPop);          % Emmigration Rates
lambda = 1-mu;                    % Immigration Rates
alpha = 0.9;
pMutation = 0.1;
sigma = 0.02*(VarMax-VarMin);

%% Start
% Empty Habitat
habitat.Position = [];
habitat.Cost = [];
% Create Habitats Array
pop = repmat(habitat, nPop, 1);
% Initialize Habitats
for i = 1:nPop
pop(i).Position = unifrnd(VarMin, VarMax, VarSize);
pop(i).Cost = CostFunction(pop(i).Position);
end
% Sort Population
[~, SortOrder] = sort([pop.Cost]);
pop = pop(SortOrder);
% Best Solution Ever Found
BestSol = pop(1);
% Array to Hold Best Costs
BestCost = zeros(MaxIt, 1);

%% BBO 
for it = 1:MaxIt
newpop = pop;
for i = 1:nPop
for k = 1:nVar
% Migration
if rand <= lambda(i)
% Emmigration Probabilities
EP = mu;
EP(i) = 0;
EP = EP/sum(EP);
% Select Source Habitat
j = RouletteWheelSelection(EP);
% Migration
newpop(i).Position(k) = pop(i).Position(k) ...
+alpha*(pop(j).Position(k)-pop(i).Position(k));
end
% Mutation
if rand <= pMutation
newpop(i).Position(k) = newpop(i).Position(k)+sigma*randn;
end
end
% Apply Lower and Upper Bound Limits
newpop(i).Position = max(newpop(i).Position, VarMin);
newpop(i).Position = min(newpop(i).Position, VarMax);
% Evaluation
newpop(i).Cost = CostFunction(newpop(i).Position);
end
% Sort New Population
[~, SortOrder] = sort([newpop.Cost]);
newpop = newpop(SortOrder);
% Select Next Iteration Population
pop = [pop(1:nKeep)
newpop(1:nNew)];
% Sort Population
[~, SortOrder] = sort([pop.Cost]);
pop = pop(SortOrder);
% Update Best Solution Ever Found
BestSol = pop(1);
% Store Best Cost Ever Found
BestCost(it) = BestSol.Cost;
% Show Iteration Information
disp(['In Iteration ' num2str(it) ': BBO Cost is = ' num2str(BestCost(it))]);
end

x=BestSol.Position';
err=BestSol.Cost;
cost=BestCost;

% Plot
plot(BestCost,'-oy', 'LineWidth', 2);
xlabel(' Iteration');
ylabel('Best Cost Value');
ax = gca; 
ax.FontSize = 12; 
set(gca,'Color','r')
legend({'BBO Regression'},'FontSize',12,'FontWeight','bold','TextColor','y');


