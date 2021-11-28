function [GS,label_GS,Accuracy_global]=DE_adjustment(GS,label_GS,TR,label_TR,Gm)  
%%
% Gm  迭代次数
% [GS,label_GS] 是要被优化的样本
% [TR, label_TR] 是正常样本
%% ---------把输入的数据集转换成按类别标签顺序排列-------------
[label_GS,I] = sort(label_GS);
GS_m=GS(I,:);
GS=GS_m;
[label_TR,I] = sort(label_TR);
TR_m=TR(I,:);
TR=TR_m;
%% ---------把输入的数据集转换成按类别标签顺序排列-------------
G=1; %初始化代数
X=GS;
label_X=label_GS;
[X_n,D]=size(X);%求目标向量X的个数（行）和维数（列）；
%% ----------求每列的最大值和最小值-------------
for j=1:D
    max_col_X(j)=max(X(:,j));
    min_col_X(j)=min(X(:,j));
end
%% ----------求每列的最大值和最小值-------------
%% 评判规格，可选KNN，Cart 或者neural network
KNN_model = ClassificationKNN.fit(TR,label_TR,'NumNeighbors',3);
Predict_label=predict(KNN_model, X);
Accuracy_global=sum(label_X==Predict_label)/size(X,1); %计算GS上的分类正确率
fprintf('-----------------Start accuracy (should be 0): %g------------------------\n',Accuracy_global);
%% correctIdx 记录GS中被分类正确的样本， 分类正确记为1，错位记为0
correctIdx=zeros(length(label_GS),1); 
pos=find(label_X==Predict_label)';
correctIdx(pos)=1; %正确的为1，不正确的为0
class_n1=length(unique(label_GS));
class_n2=length(unique(label_TR));
class_n=max(class_n1,class_n2);
%% 分别求X,TR两个数据集中各个类的数目开始
label_num_X=unique(label_X);
for jj=1:class_n1
    class_ni_X(jj)=sum(label_X==label_num_X(jj));
end
label_num_TR=unique(label_TR);
for jj=1:class_n2
    class_ni_TR(jj)=sum(label_TR==label_num_TR(jj));
end
%% 突变和交叉开始, 目的是迭代改变噪声属性和为止，使得它们能被正常样本分类正确
while( Accuracy_global<1 && G<Gm)
    if G>Gm
        break;
    end
    Temp_TR=TR;
    label_TR_Temp=label_TR;
    Accuracy_U=0;
    fprintf('--------------这是第%g次迭代前，分类正确率是%g,%g----------------------\n',G,Accuracy_global,Accuracy_U);   
    for i=1:X_n %针对每一个目标向量X生成一个试验向量U
        %%%%判断X(i）是第几类开始
        temporary=0;
        class_number_X=1;
        for j=1:length(class_ni_X)
            temporary=temporary+class_ni_X(j);  
            if i>temporary
               class_number_X=class_number_X+1;
            end
        end
        %%%%判断X(i）是第几类完 
        %%%%%从同一类中抽取样本实现“突变”和“交叉”
        temporary=0;
        for j=1:class_number_X-1
            temporary=temporary+class_ni_TR(j); 
        end
        rand_r=temporary+randperm(class_ni_TR(class_number_X));
        [~,rand_ri]=ismember(X(i,:),TR(rand_r,:),'rows');%查找是否有rand_r(1-3)=Xi
        if rand_ri>0
            rand_r(rand_ri)=[];%%%,对于遗传算法要求rand_r(1-3)≠Xi；没有考虑小于3个的情况，后面更新
        end
        %可以用交集函数[xx,a,b]=intersect(X,Y,'rows')
        %%%%%%核心突变和交叉公式开始          
        %%%%%%自适应参数确定F（SFLSDE）算法
        tau1=0.1;tau2=0.1;tau3=0.03;tau4=0.07;
        SFGSS=8;SFHC=20;Fl=0.1;Fu=0.9;
        Fi=rand(1,1);
        rand_Fi=rand(1,5);%自适应SFLSDE算法的5个随机数；
        if rand_Fi(5)<tau3
            Fi=SFGSS;
            else if tau3<=rand_Fi(5)&&rand_Fi(5)<tau4
                Fi=SFHC;
                else if rand_Fi(2)<tau1
                    Fi=Fl+Fu*rand_Fi(1);
                    end
                end
        end
        %%%%%%自适应参数确定F（SFLSDE）算法完
        KK=rand(1,1);%%突变操作的系数
        % 如果发现还是噪声，那么就改变其值
        if length(rand_r)>=3
            U(i,:)=X(i,:)+KK*(TR(rand_r(1),:)-X(i,:))+Fi*(TR(rand_r(2),:)-TR(rand_r(3),:));
        else                
            P_r1=X(i,:)+X(i,:).*((rand(1,D)-0.5)/5);
            P_r2=X(i,:)+X(i,:).*((rand(1,D)-0.5)/5);
            P_r3=X(i,:)+X(i,:).*((rand(1,D)-0.5)/5);
            U(i,:)=X(i,:)+KK*(P_r1-X(i,:))+Fi*(P_r2-P_r3);
        end       
%% U的值，大于每列的最大值设置为该列的最大值，小于最小值设置为最小值，开始；
        Position_more1=find(U(i,:)>max_col_X);
        U(i,Position_more1)=max_col_X(Position_more1);
        Position_less0=find(U(i,:)<min_col_X);
        U(i,Position_less0)=min_col_X(Position_less0);
        label_U(i,1)=label_X(i);
    end
%%  选择条件：如歌GS能被正常样本分类正确，即correctIdx=1，那么就不在改变，否则就改变
    for i=1:length(correctIdx)
        if correctIdx(i)==0
            X(i,:)=U(i,:);
            GS(i,:)=U(i,:);
        end
    end
%% 更新correctIdx和分类正确率
    KNN_model = ClassificationKNN.fit(TR,label_TR,'NumNeighbors',3);
    Predict_label=predict(KNN_model, X);
    pos=find(label_X==Predict_label)';
    Accuracy_U=sum(label_X==Predict_label)/size(X,1);
    correctIdx(pos)=1;
    if (Accuracy_U>Accuracy_global)
        Accuracy_global=Accuracy_U;
    end
    fprintf('--------------这是第%g次迭代后，分类正确率是%g,%g----------------------\n',G,Accuracy_global,Accuracy_U);
    G=G+1;
end
        
    
    
    
    
    
    
    