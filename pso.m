function [x,err,cost]=pso(CostFunction,nVar)

%% Problem 
VarSize = [1 nVar];   % Decision Variables Matrix Size
VarMin = -5;         % Decision Variables Lower Bound
VarMax = 5;         % Decision Variables Upper Bound
% PSO Parameters
MaxIt = 300;      % Maximum Number of Iterations
nPop = 50;        % Population Size (Swarm Size)

% PSO Parameters
w = 1;            % Inertia Weight
wdamp = 0.99;     % Inertia Weight Damping Ratio
c1 = 1.5;         % Personal Learning Coefficient
c2 = 2.0;         % Global Learning Coefficient
% Velocity Limits
VelMax = 0.1*(VarMax-VarMin);
VelMin = -VelMax;

%% Initialization

empty_particle.Position = [];
empty_particle.Cost = [];
empty_particle.Velocity = [];
empty_particle.Best.Position = [];
empty_particle.Best.Cost = [];

particle = repmat(empty_particle, nPop, 1);
GlobalBest.Cost = inf;
for i = 1:nPop
% Initialize Position
particle(i).Position = unifrnd(VarMin, VarMax, VarSize);
% Initialize Velocity
particle(i).Velocity = zeros(VarSize);
% Evaluation
particle(i).Cost = CostFunction(particle(i).Position);
% Update Personal Best
particle(i).Best.Position = particle(i).Position;
particle(i).Best.Cost = particle(i).Cost;
% Update Global Best
if particle(i).Best.Cost<GlobalBest.Cost
GlobalBest = particle(i).Best;
end
end
BestCost = zeros(MaxIt, 1);

%% PSO  
for it = 1:MaxIt
for i = 1:nPop
% Update Velocity
particle(i).Velocity = w*particle(i).Velocity ...
+c1*rand(VarSize).*(particle(i).Best.Position-particle(i).Position) ...
+c2*rand(VarSize).*(GlobalBest.Position-particle(i).Position);
% Apply Velocity Limits
particle(i).Velocity = max(particle(i).Velocity, VelMin);
particle(i).Velocity = min(particle(i).Velocity, VelMax);
% Update Position
particle(i).Position = particle(i).Position + particle(i).Velocity;
% Velocity Mirror Effect
IsOutside = (particle(i).Position<VarMin | particle(i).Position>VarMax);
particle(i).Velocity(IsOutside) = -particle(i).Velocity(IsOutside);
% Apply Position Limits
particle(i).Position = max(particle(i).Position, VarMin);
particle(i).Position = min(particle(i).Position, VarMax);
% Evaluation
particle(i).Cost = CostFunction(particle(i).Position);
% Update Personal Best
if particle(i).Cost<particle(i).Best.Cost
particle(i).Best.Position = particle(i).Position;
particle(i).Best.Cost = particle(i).Cost;
% Update Global Best
if particle(i).Best.Cost<GlobalBest.Cost
GlobalBest = particle(i).Best;
end
end
end
BestCost(it) = GlobalBest.Cost;
disp(['In Iteration ' num2str(it) ': PSO Cost is = ' num2str(BestCost(it))]);
w = w*wdamp;
end
BestSol = GlobalBest;

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
legend({'PSO Regression'},'FontSize',12,'FontWeight','bold','TextColor','y');
