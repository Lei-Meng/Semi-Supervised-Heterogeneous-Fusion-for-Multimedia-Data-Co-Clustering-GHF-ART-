function [CV,CT,Confu_matrix,Precision,Assign,Keywords]=FusionART(M,N,Label)

% [M]m*n1 is the pattern matrix of visual features, rows refer patterns and 
%columns refer features
% [N]m*n2 is the corresponding matrix of textual features
% [Label]m*1 contains groundtruth labels for all patterns, label values for
% classes should be 1,2,3...


%parameters
alpha = 0.01;
beta = 0.6;
gamma = 0.8;
rho = 0.01;

tic;

%complement coding
M = [M,1-M]; 
%N = [N,1-N]; no complement coding for textual feature vector

% data structure
[rowV, colV] = size(M); %patterns, features
[rowT, colT] = size(N); 


%a cluster has two feature vectors CV and CT
CV = zeros(rowV,colV); %visual features of clusters
CT = zeros(rowT,colT); %textual features of clusters

J = 0; %number of cluster

%record the sizes of clusters for textual feature learning purpose
Cluster_size = zeros(rowV,1); 

Assign = zeros(rowV,1); %cluster assignment of patterns, note rowV = rowT


% first cluster
CV = M(1,:);
CT = N(1,:);

J = 1;
Assign(1)=1;
Cluster_size(1)=1;
%record the time for generating the structure
time_structure_generation = toc;

tic;

fprintf('Algorithm starts!\n');

%encode other patterns
for n=2:rowV     
     T_max =0; %maximum choice value
     winner=0; %index of winner cluster     
     % to find the winner cluster for the nth pattern
    for j=1:J     
      %numerator of match function for visual features of current cluster j
        Mj_numerator_V = 0; 
        Mj_numerator_T = 0; %for textual features
        
        for i=1:colV
               Mj_numerator_V = Mj_numerator_V + min(M(n,i),CV(j,i));   
        end
        
        for i=1:colT
               Mj_numerator_T = Mj_numerator_T + min(N(n,i),CT(j,i));   
        end
        
        Mj_V = Mj_numerator_V/sum(M(n,:));% match value for visual features
         % match value for textual features
        Mj_T = Mj_numerator_T/(0.00001+sum(N(n,:)));
   
        if Mj_V >= rho && Mj_T >= rho             
            Tj = (1-gamma)*Mj_numerator_V/(alpha+sum(CV(j,:))) +...
           gamma*Mj_numerator_T/(alpha+sum(CT(j,:))); % Tj is choice value         
            if Tj > T_max
                T_max = Tj; % T_max is the maximum choice value
                winner = j;
            end
        end
    end    
    %if no winner, create a new cluster
    if winner == 0
        J = J+1;
        CV(J,:) = M(n,:);
        CT(J,:) = N(n,:);
        Assign(n)=J;
        Cluster_size(J)=1;
    else         %else, update the cluster prototype of the winner  
        for i=1:colV
          CV(winner,i)=beta*min(CV(winner,i),M(n,i))+(1-beta)*CV(winner,i); 
        end
        
        for i=1:colT
            CT(winner,i) = Cluster_size(winner)/(Cluster_size(winner)+1)...
                *(CT(winner,i)+N(n,i)/Cluster_size(winner)); 
        end       
        Assign(n)=winner;
        Cluster_size(winner)=Cluster_size(winner)+1;
    end    
end

time_algorithm = toc; % record time for generating clusters
%fprintf('%s: abnormal status detected!\n', record{i,1});
fprintf('Algorithm ends!\n');

%statistics

Distri = zeros(J,1); %cluster sizes

for i=1:rowV
    
    Distri(Assign(i))= Distri(Assign(i))+1;
    
end
     

Cate_num = max(Label); %find the number of classes
           
Confu_matrix = zeros(J,Cate_num); %confusion matrix

%build the confusion matrix
for i=1:rowV
    
    Confu_matrix(Assign(i),Label(i))= Confu_matrix(Assign(i),Label(i))+1;
    
end

[Max_value,Index]=max(Confu_matrix,[],2);


%calculate the precision of clusters
%Precision = zeros(J,1); 

Precision = Max_value ./ Distri;


%calculate the recall of clusters
%calculate the sizes of the major classes corresponding to clusters
 Class_size=zeros(J,1);

for i=1:J
    
    Class_size(i) = sum(Confu_matrix(:,Index(i)),1);
    
end

Recall = Max_value ./ Class_size;

Recall = Recall;
           


%assign images to real folders in terms of clusters           

fprintf('Assigning images to the respective cluster folders!\n');

load('index_selected_images.mat');
load('imageList.mat');

root = 'C:\Users\Meng LEI\Desktop\lily\lily_commerce';
root_destiny =  strcat(root,'\codes for lilycommerce\images\results');

for i = 1:J %generate folders for clusters
    clusterName = strcat('cluster',num2str(i));
    mkdir(fullfile(root_destiny,clusterName));
end




for i = 1:length(Assign) %copy images to the respective folders
    copyfile(imageList{index_selected_images(i)},...
    fullfile(root_destiny, strcat('cluster',num2str(Assign(i)))));
end  



% plot statistics of features 

fprintf('Obtaining feature distributions of clusters!\n');

featurePlot;

% extract five keywords for each cluster

KeywordExtract;

end
      




