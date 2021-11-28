function [Ip_data, Ip_label]=SMOTENaNDE(Minority_data,Minority_label,Majority_data,Majority_label,Synthetic_samples,Synthetic_label)
%% SMOTE-NaN-DE：用natural neighbors and DE去处理SMOTE-based method中的噪声
%% 原始不平衡数据
data=[Minority_data;Majority_data];
t=[Minority_label;Majority_label];
%% 改进的数据集
improve_data=[data;Synthetic_samples];
improve_t=[t;Synthetic_label];
NaN=NaN_Search(improve_data);
%% 用自然近邻去识别噪声
Noises=[]; %可疑噪声的集合，将被DE优化
for i=1:size(improve_data,1)
    labels=improve_t(NaN{i}); %第i个样本的自然近邻的类标签集合
    pos=find(labels~=improve_t(i));
%% 如果一个样本是可疑的噪声，当且仅当，其中一个自然近邻与它类别不一致，或者这个样本是离群点
    if length(unique(pos))>=1 || length (NaN{i})==0 
        Noises=[Noises;i];
    end
end
%% 计算可疑噪声集合和正常样本（非噪声样本）的集合
Non_Noise=setdiff([1:1:length(improve_t)],Noises);  % 非噪声样本的序号
Non_data=improve_data(Non_Noise,:);                 % 噪声样本的数据
Non_t=improve_t(Non_Noise,:);                       % 噪声样本的标签
Noises_data=improve_data(Noises,:);                 % 噪声样本的数据
Noises_data_label=improve_t(Noises);                % 噪声样本的标签
%% 基于非噪声数据，用差分计算去优化噪声数据
if length(Noises)~=0
    [GS,label_GS,Accuracy_global]=DE_adjustment(Noises_data,Noises_data_label,Non_data,Non_t,200) ;
end
%% 合并非噪声数据和被优化的噪声数据
Ip_data=[Non_data;GS];
Ip_label=[Non_t;label_GS];
end

