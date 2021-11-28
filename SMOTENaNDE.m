function [Ip_data, Ip_label]=SMOTENaNDE(Minority_data,Minority_label,Majority_data,Majority_label,Synthetic_samples,Synthetic_label)
%% SMOTE-NaN-DE����natural neighbors and DEȥ����SMOTE-based method�е�����
%% ԭʼ��ƽ������
data=[Minority_data;Majority_data];
t=[Minority_label;Majority_label];
%% �Ľ������ݼ�
improve_data=[data;Synthetic_samples];
improve_t=[t;Synthetic_label];
NaN=NaN_Search(improve_data);
%% ����Ȼ����ȥʶ������
Noises=[]; %���������ļ��ϣ�����DE�Ż�
for i=1:size(improve_data,1)
    labels=improve_t(NaN{i}); %��i����������Ȼ���ڵ����ǩ����
    pos=find(labels~=improve_t(i));
%% ���һ�������ǿ��ɵ����������ҽ���������һ����Ȼ�����������һ�£����������������Ⱥ��
    if length(unique(pos))>=1 || length (NaN{i})==0 
        Noises=[Noises;i];
    end
end
%% ��������������Ϻ������������������������ļ���
Non_Noise=setdiff([1:1:length(improve_t)],Noises);  % ���������������
Non_data=improve_data(Non_Noise,:);                 % ��������������
Non_t=improve_t(Non_Noise,:);                       % ���������ı�ǩ
Noises_data=improve_data(Noises,:);                 % ��������������
Noises_data_label=improve_t(Noises);                % ���������ı�ǩ
%% ���ڷ��������ݣ��ò�ּ���ȥ�Ż���������
if length(Noises)~=0
    [GS,label_GS,Accuracy_global]=DE_adjustment(Noises_data,Noises_data_label,Non_data,Non_t,200) ;
end
%% �ϲ����������ݺͱ��Ż�����������
Ip_data=[Non_data;GS];
Ip_label=[Non_t;label_GS];
end

