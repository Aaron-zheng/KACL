import numpy as np
from scipy.sparse.construct import rand
import torch
import torch.nn as nn

import torch.nn.functional as F
import dgl

import dgl.function as fn
from conv import myGATConv, DropLearner

class Contrast_2view(nn.Module):
    # 。这个类专门设计用于处理来自两个不同视图（例如用户-物品交互和知识图谱）的数据，并计算这两种数据表达的相似度
    # cf_dim: 用户-物品交互特征的维度。
    # kg_dim: 知识图谱特征的维度。
    # hidden_dim: 网络中隐藏层的维度。
    # tau: 温度系数，用于控制对比学习中相似度计算的敏感性。
    # cl_size: 正样本的数量，用于创建对角矩阵。
    def __init__(self, cf_dim, kg_dim, hidden_dim, tau, cl_size):
        super(Contrast_2view, self).__init__()
        # 定义了两个投影层 projcf 和 projkg，分别用于用户-物品交互和知识图谱。
        self.projcf = nn.Sequential(
            nn.Linear(cf_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.projkg = nn.Sequential(
            nn.Linear(kg_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        # 使用对角矩阵 pos 表示正样本。
        self.pos = torch.eye(cl_size).cuda()
        # 设置超参数 tau 作为温度系数，用于对比学习中的相似度计算。
        self.tau = tau
        for model in self.projcf:
            if isinstance(model, nn.Linear):
                nn.init.xavier_normal_(model.weight, gain=1.414)
        for model in self.projkg:
            if isinstance(model, nn.Linear):
                nn.init.xavier_normal_(model.weight, gain=1.414)

    def sim(self, z1, z2):
        # 分别计算 z1 和 z2 的L2范数。
        # keepdim=True 确保结果的维度与原始数据一致，便于后续的矩阵运算。
        z1_norm = torch.norm(z1, dim=-1, keepdim=True)
        z2_norm = torch.norm(z2, dim=-1, keepdim=True)
        # 计算 z1 和 z2 的点积，其中 z2.t() 是 z2 的转置。这个点积给出了嵌入向量之间的相似度分数。
        dot_numerator = torch.mm(z1, z2.t())
        # 计算 z1 和 z2 范数的外积，用于规范化点积结果，确保相似度分数不受向量长度的影响。
        dot_denominator = torch.mm(z1_norm, z2_norm.t())
        #  通过对点积除以范数外积的结果进行缩放并应用指数函数，计算相似度矩阵
        #  。这里 self.tau 作为一个温度参数，用于控制相似度分布的平滑程度。温度参数较低会使得输出更加尖锐（即相似度高的项更突出）。
        sim_matrix = torch.exp(dot_numerator / dot_denominator / self.tau)
        # 进行行归一化，确保每行的相似度值加和为1。1e-8 是为了防止分母为零。
        sim_matrix = sim_matrix/(torch.sum(sim_matrix, dim=1).view(-1, 1) + 1e-8)
        assert sim_matrix.size(0) == sim_matrix.size(1)
        # 首先通过 sim_matrix.mul(self.pos) 应用一个掩码 self.pos（通常是单位矩阵，标识正样本位置），然后对每行的结果求和和对数，取负值并计算平均值。
        # 这样计算出的损失旨在最小化正样本对的对数似然损失，这是一种常用的方法来进行对比学习，尤其是在处理如自监督学习的场景中。
        lori_mp = -torch.log(sim_matrix.mul(self.pos).sum(dim=-1)).mean()
        # 返回基于正样本的对比学习损失。
        return lori_mp

    def forward(self, z1, z2):
        # 根据输入计算两个嵌入的投影。
        multi_loss = False
        # 分别通过两个神经网络层序列 projcf 和 projkg 进行投影。
        # 这些层序列通常包含线性层和激活函数（例如这里使用了 nn.ELU()），用于将嵌入转换到一个适合进行相似度计算的新空间。
        z1_proj = self.projcf(z1)
        z2_proj = self.projkg(z2)
        if multi_loss:
            # 如果 multi_loss 为真，计算多个损失，包括自身的相似度损失。
            loss1 = self.sim(z1_proj, z2_proj)
            # 计算 z1_proj 自身的相似度损失，这种自比较可以帮助模型学习到更加紧密、一致的内部结构。
            loss2 = self.sim(z1_proj, z1_proj)
            # 计算 z2_proj 自身的相似度损失，同样目的是加强模型的内部一致性。
            loss3 = self.sim(z2_proj, z2_proj)
            # 最终损失是这三个损失的平均值：(loss1 + loss2 + loss3) / 3。
            # 通过结合不同的损失，模型能够从不同角度学习嵌入的相似性和区分性，从而提高其泛化能力和鲁棒性。
            return (loss1 + loss2 + loss3) / 3
        else:
            # 返回对比学习损失。
            return self.sim(z1_proj, z2_proj)


# 是一种知识图谱中用于链接预测的经典模型，利用三线性形式来计算实体之间的关系得分。
# 这个类的实现非常适用于处理实体和关系嵌入的任务，如知识图谱的链接预测。
class DistMult(nn.Module):
    # num_rel这个参数指定了关系的总数，对应知识图谱中不同类型的关系。
    # dim: 这是每个关系嵌入的维度。
    def __init__(self, num_rel, dim):
        super(DistMult, self).__init__()
        # 形状为 (num_rel, dim, dim) 的三维权重矩阵，其中每个 dim x dim 的矩阵代表一个特定关系的权重。
        # 这个矩阵用于捕捉和编码关系特定的语义信息。
        self.W = nn.Parameter(torch.FloatTensor(size=(num_rel, dim, dim)))
        # 使用 Xavier 初始化权重。
        # 这行代码使用 Xavier 正态初始化方法来初始化权重矩阵 self.W。
        # gain=1.414 是初始化时的比例因子，这个值通常用于优化 ReLU 和其变体（这里为 ELU）激活函数后的权重初始化，确保网络在训练初期有较好的性能。
        nn.init.xavier_normal_(self.W, gain=1.414)

    def forward(self, left_emb, right_emb, r_id):
        # left_emb: 输入的左侧实体嵌入。
        # right_emb: 输入的右侧实体嵌入。
        # r_id: 关系的索引，指定使用哪一个关系的嵌入进行计算。
        # 接受左右嵌入和关系ID，进行双向矩阵乘法，以计算关系预测的结果。
        thW = self.W[r_id]
        # 这两行代码将输入的嵌入向量增加一个维度，使其适合进行矩阵乘法。
        # unsqueeze 操作是为了将嵌入向量转化为矩阵形式，以便与关系矩阵 thW 进行三线性乘法。
        left_emb = torch.unsqueeze(left_emb, 1)
        right_emb = torch.unsqueeze(right_emb, 2)
        # 这行代码首先计算左侧实体嵌入与关系矩阵的乘积，然后将结果与右侧实体嵌入相乘，
        # 最终通过 squeeze() 方法移除单维条目，得到最终的关系得分。
        return torch.bmm(torch.bmm(left_emb, thW), right_emb).squeeze()
    
    
class myGAT(nn.Module):
    def __init__(self, args, num_entity, num_etypes, num_hidden, num_classes, num_layers,
                 heads, activation, feat_drop, attn_drop, negative_slope, residual, pretrain=None):
        super(myGAT, self).__init__()
        # 定义多个GAT层：输入层、隐藏层、输出层。
        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList()
        self.sub_gat_layers = nn.ModuleList()
        self.kg_gat_layers = nn.ModuleList()
        
        self.drop_learner = False
        
        self.activation = activation
        
        self.cfe_size = args.embed_size
        self.kge_size = args.kge_size
        self.edge_dim = self.kge_size
        self.cl_alpha = args.cl_alpha
        alpha = args.alpha
        cl_dim = self.cfe_size
        
        tau = args.temperature
        self.weight_decay = args.weight_decay
        self.kg_weight_decay = args.kg_weight_decay
        self.batch_size = args.batch_size
        
        if pretrain is not None:
            user_embed = pretrain['user_embed']
            item_embed = pretrain['item_embed']
            self.user_size = user_embed.shape[0]
            self.item_size = item_embed.shape[0]
            self.ret_num = self.user_size + self.item_size
            self.embed = nn.Parameter(torch.zeros((self.ret_num, self.cfe_size)))
            self.cl_embed = nn.Parameter(torch.zeros((self.ret_num, self.cfe_size)))
            nn.init.xavier_normal_(self.embed, gain=1.414)
            nn.init.xavier_normal_(self.cl_embed, gain=1.414)
            self.ini = torch.FloatTensor(np.concatenate([user_embed, item_embed], axis=0)).cuda()

        self.kg_embed = nn.Parameter(torch.zeros((num_entity, args.kge_size)))
        self.user_embed = nn.Parameter(torch.zeros((self.user_size, args.kge_size + 48)))
        
        nn.init.xavier_normal_(self.kg_embed, gain=1.414)
        #nn.init.xavier_normal_(self.user_embed, gain=1.414)
        # input projection (no residual)
        self.gat_layers.append(myGATConv(self.cfe_size, num_hidden, heads[0],
            feat_drop, attn_drop, negative_slope, False, self.activation, bias=True, alpha=alpha))
        # hidden layers
        for l in range(1, num_layers):
            # due to multi-head, the in_dim = num_hidden * num_heads
            self.gat_layers.append(myGATConv(num_hidden * heads[l-1],
                 num_hidden, heads[l],
                feat_drop, attn_drop, negative_slope, residual, self.activation, bias=True, alpha=alpha))
        # output projection
        self.gat_layers.append(myGATConv(num_hidden * heads[-2],
             num_classes, heads[-1],
            feat_drop, attn_drop, negative_slope, residual, None, bias=True, alpha=alpha))
        
        
        # input projection (no residual)
        self.sub_gat_layers.append(myGATConv(self.cfe_size, num_hidden, heads[0],
            feat_drop, attn_drop, negative_slope, False, self.activation, bias=True, alpha=alpha))
        # hidden layers
        for l in range(1, num_layers):
            # due to multi-head, the in_dim = num_hidden * num_heads
            self.sub_gat_layers.append(myGATConv(num_hidden * heads[l-1],
                 num_hidden, heads[l],
                feat_drop, attn_drop, negative_slope, residual, self.activation, bias=True, alpha=alpha))
        # output projection
        self.sub_gat_layers.append(myGATConv(num_hidden * heads[-2],
             num_classes, heads[-1],
            feat_drop, attn_drop, negative_slope, residual, None, bias=True, alpha=alpha))
        
        # input projection (no residual)
        self.kg_gat_layers.append(myGATConv(self.kge_size, num_hidden, heads[0],
            feat_drop, attn_drop, negative_slope, False, self.activation, bias=True, alpha=alpha))
        # hidden layers
        for l in range(1, num_layers):
            # due to multi-head, the in_dim = num_hidden * num_heads
            self.kg_gat_layers.append(myGATConv(num_hidden * heads[l-1],
                 num_hidden, heads[l],
                feat_drop, attn_drop, negative_slope, residual, self.activation, bias=True, alpha=alpha))
        # output projection
        self.kg_gat_layers.append(myGATConv(num_hidden * heads[-2],
             num_classes, heads[-1],
            feat_drop, attn_drop, negative_slope, residual, None, bias=True, alpha=alpha))
        self.epsilon = torch.FloatTensor([1e-12]).cuda()
        self.contrast = Contrast_2view(self.cfe_size + 48, self.kge_size + 48, cl_dim, tau, args.batch_size_cl)
        self.decoder = DistMult(num_etypes, self.kge_size + 48)
        self.learner1 = DropLearner(self.cfe_size, self.cfe_size)
        self.learner2 = DropLearner(self.kge_size, self.kge_size, self.edge_dim)
        self.cf_edge_weight = None
        self.kg_edge_weight = None

    # 计算用户-物品嵌入，包含多层GAT操作，使用标准化确保嵌入的稳定性。
    def calc_ui_emb(self, g):
        all_embed = []
        h = self.embed
        tmp = (h / (torch.max(torch.norm(h, dim=1, keepdim=True),self.epsilon)))
        all_embed.append(tmp)
        res_attn = None
        for l in range(self.num_layers):
            h, res_attn = self.gat_layers[l](g, h, res_attn=res_attn)
            h = h.flatten(1)
            tmp = (h / (torch.max(torch.norm(h, dim=1, keepdim=True),self.epsilon)))
            all_embed.append(tmp)
        # output projection
        logits, _ = self.gat_layers[-1](g, h, res_attn=res_attn)
        logits = logits.mean(1)
        all_embed.append(logits / (torch.max(torch.norm(logits, dim=1, keepdim=True),self.epsilon)))
        all_embed = torch.cat(all_embed, 1)
        return all_embed
    
    def calc_cl_emb(self, g, drop_learn = False):
        # 计算用于对比学习的嵌入，支持 Dropout 学习。
        all_embed = []
        h = self.cl_embed
        tmp = (h / (torch.max(torch.norm(h, dim=1, keepdim=True),self.epsilon)))
        edge_weight = None
        reg = 0
        if drop_learn:
            reg, edge_weight = self.learner1(tmp, g, temperature = 0.7)
            self.cf_edge_weight = edge_weight.detach()
        else:
            edge_weight = self.cf_edge_weight
        all_embed.append(tmp)
        res_attn = None
        for l in range(self.num_layers):
            h, res_attn = self.sub_gat_layers[l](g, h, res_attn=res_attn, edge_weight = edge_weight)
            h = h.flatten(1)
            tmp = (h / (torch.max(torch.norm(h, dim=1, keepdim=True),self.epsilon)))
            all_embed.append(tmp)
        # output projection
        logits, _ = self.sub_gat_layers[-1](g, h, res_attn=res_attn, edge_weight = edge_weight)
        logits = logits.mean(1)
        all_embed.append(logits / (torch.max(torch.norm(logits, dim=1, keepdim=True),self.epsilon)))
        all_embed = torch.cat(all_embed, 1)
        if drop_learn:
            return all_embed, reg
        else:
            return all_embed
    
    def calc_kg_emb(self, g, drop_learn = False):
        # 计算知识图谱嵌入，与 calc_cl_emb() 类似，使用 Dropout 学习。
        all_embed = []
        h = self.kg_embed
        tmp = (h / (torch.max(torch.norm(h, dim=1, keepdim=True),self.epsilon)))
        edge_weight = None
        reg = 0
        if drop_learn:
            reg, edge_weight = self.learner2(tmp, g, temperature = 0.7)
            self.kg_edge_weight = edge_weight.detach()
        else:
            edge_weight = self.kg_edge_weight
        all_embed.append(tmp)
        res_attn = None
        for l in range(self.num_layers):
            h, res_attn = self.kg_gat_layers[l](g, h, res_attn=res_attn, edge_weight = edge_weight)
            h = h.flatten(1)
            tmp = (h / (torch.max(torch.norm(h, dim=1, keepdim=True),self.epsilon)))
            all_embed.append(tmp)
        # output projection
        logits, _ = self.kg_gat_layers[-1](g, h, res_attn=res_attn, edge_weight = edge_weight)
        logits = logits.mean(1)
        all_embed.append(logits / (torch.max(torch.norm(logits, dim=1, keepdim=True),self.epsilon)))
        all_embed = torch.cat(all_embed, 1)
        if drop_learn:
            return all_embed, reg
        else:
            return all_embed
    
    def calc_cf_loss(self, g, sub_g, kg, user_id, pos_item, neg_item):#(self, g, user_id, item_id, pos_mat):
        # 计算推荐系统的损失，包括基础损失和正则化损失。
        # 使用正例和负例的嵌入，计算 Softplus 损失。
        embedding_cf = self.calc_ui_emb(g)
        #embedding_cf = self.calc_cl_emb(g)
        reg_cl, reg_kg = 0, 0
        """
        reg_cl, reg_kg = 0, 0
        #embedding_cl, reg_cl = self.calc_cl_emb(sub_g, True)
        embedding_cl = self.calc_cl_emb(sub_g, False)
        
        #embedding_kg = self.calc_kg_emb(kg, e_feat)[:self.item_size]
        #embedding_kg, reg_kg = self.calc_kg_emb(kg, True)
        embedding_kg = self.calc_kg_emb(kg, False)
        
        embedding_kg = torch.cat([self.user_embed, embedding_kg[:self.item_size]], 0)
        
        embedding = torch.cat([embedding_cf, embedding_cl, embedding_kg, self.ini], 1)
        """
        embedding = torch.cat([embedding_cf, self.ini], 1)
        #embedding = torch.cat([embedding_cl, embedding_kg, self.ini], 1)
        #embedding = torch.cat([embedding_cf, embedding_kg, self.ini], 1)
        
        u_emb = embedding[user_id]
        p_emb = embedding[pos_item]
        n_emb = embedding[neg_item]
        pos_scores = (u_emb * p_emb).sum(dim=1)
        neg_scores = (u_emb * n_emb).sum(dim=1)
        base_loss = F.softplus(neg_scores - pos_scores).mean()
        reg_loss = self.weight_decay * ((u_emb*u_emb).sum()/2 + (p_emb*p_emb).sum()/2 + (n_emb*n_emb).sum()/2) / self.batch_size
        loss = base_loss + reg_loss
        return loss, reg_cl, reg_kg

    def calc_cl_loss(self, g, kg, item):
        # 计算对比学习损失，使用 Contrast_2view 进行相似度比较。
        embedding = self.calc_cl_emb(g)
        #kg_embedding = self.calc_kg_emb(kg, e_feat)
        kg_embedding = self.calc_kg_emb(kg)
        kg_emb = kg_embedding[item]
        item = item + np.array([self.user_size])
        cf_emb = embedding[item]
        cl_loss = self.contrast(cf_emb, kg_emb)
        loss = self.cl_alpha*cl_loss
        return loss

    def calc_kg_loss(self, g, h, r, pos_t, neg_t):
        # 计算知识图谱损失，使用 DistMult 计算关系预测的损失。
        #embedding = self.calc_kg_emb(g, e_feat)
        weight = False
        embedding = self.calc_kg_emb(g)
        
        h_emb = embedding[h]
        pos_t_emb = embedding[pos_t]
        neg_t_emb = embedding[neg_t]

        pos_score = self.decoder(h_emb, pos_t_emb, r)
        neg_score = self.decoder(h_emb, neg_t_emb, r)
        aug_edge_weight = 1
        if weight:
            emb = self.kg_embed
            emb = (emb / (torch.max(torch.norm(emb, dim=1, keepdim=True),self.epsilon)))
            _, aug_edge_weight = self.learner2.get_weight(emb[h], emb[pos_t], temperature = 0.7)
            #print(aug_edge_weight.size(), neg_score.size())
        #loss
        base_loss = (aug_edge_weight * F.softplus(-neg_score + pos_score)).mean()
        return base_loss

    def forward(self, mode, *input):
        # 根据模式(cf, kg, cl, test)调用不同的方法。
        # 提供了对比学习、知识图谱、推荐系统的训练和测试的前向计算。
        if mode == "cf":
            return self.calc_cf_loss(*input)
        elif mode == "kg":
            return self.calc_kg_loss(*input)
        elif mode == "cl":
            return self.calc_cl_loss(*input)
        elif mode == "test":
            #g, kg, e_feat = input
            g, kg = input
            self.kg_edge_weight = None
            self.cf_edge_weight = None
            embedding_cf = self.calc_ui_emb(g)
            #embedding_cf = self.calc_cl_emb(g)

            embedding_cl = self.calc_cl_emb(g)
            embedding_kg = self.calc_kg_emb(kg)
            
            embedding_kg = torch.cat([self.user_embed, embedding_kg[:self.item_size]], 0)
            embedding = torch.cat([embedding_cf, embedding_cl, embedding_kg, self.ini], 1)  

            #embedding = torch.cat([embedding_cf, self.ini], 1)        
            
            return embedding