import numpy as np
from utility.load_data import Data

import scipy.sparse as sp
import random as rd
import collections
from time import time

class KGAT_loader(Data):
    def __init__(self, args, path):
        super().__init__(args, path)
        self.all_kg_dict = self._get_all_kg_dict()
        # generate the sparse adjacency matrices for user-item interaction.
        self.adj_list= self._get_cf_adj_list()
        self.kg_adj_list, self.adj_r_list = self._get_kg_adj_list()

        # generate the sparse laplacian matrices.
        self.lap_list = self._get_lap_list()
        self.kg_lap_list = self._get_kg_lap_list()
        
        # generate the triples dictionary, key is 'head', value is '(tail, relation)'.
    
    def _get_cf_adj_list(self, is_subgraph = False, dropout_rate = None):
        #  np_mat（NumPy 数组表示的稀疏矩阵）、row_pre 和 col_pre（分别是行和列的偏移量）。
        def _np_mat2sp_adj(np_mat, row_pre, col_pre):
            # 函数首先计算了节点总数 n_all，这个值是用户数和物品数之和
            n_all = self.n_users + self.n_items
            # single-direction
            # 提取了行和列的索引，并添加了行和列的偏移量
            a_rows = np_mat[:, 0] + row_pre
            a_cols = np_mat[:, 1] + col_pre
            if is_subgraph is True:
                # 子图抽样。
                subgraph_idx = np.arange(len(a_rows))
                # 代码随机选择一部分边（基于 dropout_rate），然后对 a_rows 和 a_cols 进行子集抽样
                subgraph_id = np.random.choice(subgraph_idx, size = int(dropout_rate * len(a_rows)), replace = False)
                a_rows = a_rows[subgraph_id]
                a_cols = a_cols[subgraph_id]
            # 设置矩阵的权重，这里所有边的权重都设置为1。
            vals = [1.] * len(a_rows) * 2
            # 将原来的行和列索引合并，以构建对称的图。
            rows = np.concatenate((a_rows, a_cols))
            # 将原来的行和列索引合并，以构建对称的图。
            cols = np.concatenate((a_cols, a_rows))
            #  创建了一个稀疏矩阵。COO格式是稀疏矩阵的一种常见格式
            adj = sp.coo_matrix((vals, (rows, cols)), shape=(n_all, n_all))
            # 函数返回了创建的稀疏矩阵 adj，其大小是 (n_all, n_all)，表示整个用户-物品图。
            return adj
        R = _np_mat2sp_adj(self.train_data, row_pre=0, col_pre=self.n_users)
        return R

    def _get_kg_adj_list(self, is_subgraph = False, dropout_rate = None):
        adj_mat_list = []
        adj_r_list = []
        # 用于将一个NumPy矩阵转换成两个稀疏矩阵，通常用于图或网络结构的构建。
        # 主要逻辑是从输入矩阵中提取行和列索引，然后根据需要进行子图抽样，并生成对应的稀疏矩阵
        def _np_mat2sp_adj(np_mat):
            # 这行代码获取总节点数，通常表示整个网络中可能的节点数。这可能用于确定生成的稀疏矩阵的大小。
            n_all = self.n_entities
            # single-direction
            # 从输入矩阵中提取对应的行和列索引。这些索引可能代表图中的节点间的关系。
            a_rows = np_mat[:, 0]
            a_cols = np_mat[:, 1]
            if is_subgraph is True:
                # 如果 is_subgraph 是 True，代码会执行子图抽样。
                # 首先创建一个索引数组 subgraph_idx = np.arange(len(a_rows))，
                subgraph_idx = np.arange(len(a_rows))
                # 然后通过 np.random.choice 按照 dropout_rate 的比例随机选择一些索引，形成子图。
                # 这可以用于生成更小的图结构，适合于大型数据集或需要随机抽样的场景。
                subgraph_id = np.random.choice(subgraph_idx, size = int(dropout_rate * len(a_rows)), replace = False)
                #print(subgraph_id[:10])
                a_rows = a_rows[subgraph_id]
                a_cols = a_cols[subgraph_id]
            # 为第一个稀疏矩阵中的每个边设置权重，所有值为1。
            a_vals = [1.] * len(a_rows)

            # 交换行和列，以形成对称关系。这种设计用于创建一个双向图或确保两个矩阵相互对应。
            b_rows = a_cols
            b_cols = a_rows
            # 为第二个稀疏矩阵中的每个边设置权重，也全为1。
            b_vals = [1.] * len(b_rows)
            # 创建了第一个稀疏矩阵，代表从 a_rows 到 a_cols 的关系。
            a_adj = sp.coo_matrix((a_vals, (a_rows, a_cols)), shape=(n_all, n_all))
            # 创建了第二个稀疏矩阵，代表从 b_rows 到 b_cols 的关系。
            b_adj = sp.coo_matrix((b_vals, (b_rows, b_cols)), shape=(n_all, n_all))
            # 函数返回了两个稀疏矩阵 a_adj 和 b_adj
            return a_adj, b_adj

        # 一个循环中处理一组关系数据，并生成一系列稀疏矩阵，同时更新关系的计数。
        # 主要作用是将关系数据转换成稀疏矩阵，并为每个矩阵分配相应的关系ID
        for r_id in self.relation_dict.keys():
            #print(r_id)
            # 通过将字典中的关系数据转换成NumPy数组
            # 这两个矩阵通常代表某个关系的正向和逆向结构
            K, K_inv = _np_mat2sp_adj(np.array(self.relation_dict[r_id]))
            # 将生成的第一个稀疏矩阵追加到列表
            adj_mat_list.append(K)
            # 将当前关系的ID追加到列表
            adj_r_list.append(r_id)
            # 将第二个稀疏矩阵追加到
            adj_mat_list.append(K_inv)
            # 为逆向关系分配一个新的ID，并追加到
            adj_r_list.append(r_id + self.n_relations)
        # 更新总关系数为原来的两倍。这一操作用于确保正向和逆向关系之间的独立性。
        self.n_relations = self.n_relations * 2
        #print(adj_r_list)
        # 一个是生成的稀疏矩阵 adj_mat_list，另一个是对应的关系ID adj_r_list
        return adj_mat_list, adj_r_list

    # 计算双向正则化拉普拉斯矩阵
    def _bi_norm_lap(self, adj):
        # 计算稀疏矩阵 adj 的行和，这代表每个节点的度数。该行和是稀疏矩阵在图结构中非常重要的属性。
        rowsum = np.array(adj.sum(1))
        # 计算每个度数的倒数的平方根。这是为了在拉普拉斯矩阵的正则化过程中使用。
        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        # 将无穷大（由于除以零）设置为0，以避免异常值。
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        # 创建一个对角稀疏矩阵，值为 d_inv_sqrt。这个对角矩阵用于将拉普拉斯矩阵中的行和列的影响正则化
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
        # 是计算双向正则化拉普拉斯矩阵
        # adj.dot(d_mat_inv_sqrt) 将 adj 与对角稀疏矩阵 d_mat_inv_sqrt 相乘，以调整行的权重。
        # transpose() 对结果进行转置，以便调整列的权重。
        # dot(d_mat_inv_sqrt) 再次与 d_mat_inv_sqrt 相乘，完成正则化。
        bi_lap = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)
        # 返回拉普拉斯矩阵的COO（Coordinate list）格式
        return bi_lap.tocoo()

    def _si_norm_lap(self, adj):
        # 计算稀疏矩阵 adj 的行和。这通常代表每个节点的度数，表示该节点在图中的连接数量。
        rowsum = np.array(adj.sum(1))
        # 计算每个度数的倒数。这用于单向正则化，因为在标准拉普拉斯矩阵中，度数的倒数可以用来调整每个节点的影响。
        d_inv = np.power(rowsum, -1).flatten()
        # 如果某个节点的度数为零，度数的倒数会变成无穷大，这里将其设置为0，以避免计算错误。
        d_inv[np.isinf(d_inv)] = 0.
        # 基于 d_inv 创建一个对角稀疏矩阵。
        d_mat_inv = sp.diags(d_inv)
        # 将对角稀疏矩阵 d_mat_inv 与稀疏矩阵 adj 相乘，
        norm_adj = d_mat_inv.dot(adj)
        # 返回COO格式的单向正则化拉普拉斯矩阵
        return norm_adj.tocoo()
    
    def _get_kg_lap_list(self, is_subgraph = False, subgraph_adj = None):
        if is_subgraph is True:
            adj_list = subgraph_adj
        else:
            adj_list = self.kg_adj_list
        if self.args.adj_type == 'bi':
            lap_list = [self._bi_norm_lap(adj) for adj in adj_list]
        else:
            lap_list = [self._si_norm_lap(adj) for adj in adj_list]
        return lap_list
        
    def _get_lap_list(self, is_subgraph = False, subgraph_adj = None):
        if is_subgraph is True:
            # 如果 is_subgraph 为 True，则使用 subgraph_adj 作为输入的稀疏矩阵。这个条件通常用于处理子图。
            adj = subgraph_adj
        else:
            #  否则，默认使用 self.adj_list 作为稀疏矩阵
            adj = self.adj_list
        if self.args.adj_type == 'bi':
            # 如果是 'bi'，调用 _bi_norm_lap 函数来计算双向正则化的拉普拉斯矩阵
            lap_list = self._bi_norm_lap(adj)
        else:
            # 如果 adj_type 不为 'bi'，则默认使用单向正则化的拉普拉斯矩阵。
            lap_list = self._si_norm_lap(adj)
        # 返回生成的拉普拉斯矩阵列表
        return lap_list

    # 这个函数的作用是构建一个知识图谱的表示，其中实体通过不同关系相互连接。
    # 返回的 all_kg_dict 提供了一种结构化的方式来存储和访问知识图谱中的实体和关系
    def _get_all_kg_dict(self):
        # 创建一个 defaultdict，其中默认值是一个空列表。
        all_kg_dict = collections.defaultdict(list)
        # 循环遍历 self.relation_dict 的所有键
        for relation in self.relation_dict.keys():
            # head 和 tail 分别代表关系的起点和终点
            for head, tail in self.relation_dict[relation]:
                # 表示 head 通过这个关系连接到 tail。
                all_kg_dict[head].append((tail, relation))
                # 通过为 relation 加上 self.n_relations，确保反向关系被处理。这一操作常用于表示无向图或反向关系。
                all_kg_dict[tail].append((head, relation + self.n_relations))
        return all_kg_dict

    def _generate_train_cf_batch(self):
        if self.batch_size <= self.n_users:
            # 如果批处理大小小于等于存在用户的总数，则从存在用户中随机选择相应数量的用户
            users = rd.sample(self.exist_users, self.batch_size)
        else:
            # 如果批处理大小大于存在用户的总数，则从存在用户中重复地随机选择用户，以填满批处理大小
            users_list = list(self.exist_users)
            users = [rd.choice(users_list) for _ in range(self.batch_size)]

        def sample_pos_items_for_u(u, num):
            # 从train_user_dict字典中获取用户u的所有正样本项目。这个字典可能是一个包含用户ID和对应正样本项目ID列表的数据结构。
            pos_items = self.train_user_dict[u]
            # 计算用户u的正样本项目数量
            n_pos_items = len(pos_items)
            # 初始化一个空列表，用于存储从正样本项目中抽取的项目
            pos_batch = []
            while True:
                # 检查是否已经抽取了指定数量的项目，如果是，则跳出循环。
                if len(pos_batch) == num: break
                # 随机生成一个正样本项目的索引，范围从0到n_pos_items-1。
                pos_id = np.random.randint(low=0, high=n_pos_items, size=1)[0]
                # 根据随机生成的索引，从正样本项目列表中获取一个具体的项目ID。
                pos_i_id = pos_items[pos_id]
                # 检查该项目是否已经在抽取的项目列表中。
                if pos_i_id not in pos_batch:
                    pos_batch.append(pos_i_id)
            # 返回抽取的正样本项目列表，直到列表中包含了指定数量的项目为止。
            return pos_batch
        
        def sample_neg_items_for_u(u, num):
            # 初始化一个空列表，用于存储从负样本项目中抽取的项目。
            neg_items = []
            while True:
                # 检查是否已经抽取了指定数量的项目，如果是，则跳出循环。
                if len(neg_items) == num: break
                # 随机生成一个负样本项目的索引，范围从0到self.n_items-1
                neg_i_id = np.random.randint(low=0, high=self.n_items,size=1)[0]
                # 检查随机生成的负样本项目是否同时不在给定用户u的正样本项目中（避免了采样到正样本项目）且不在已经抽取的负样本项目列表中。
                if neg_i_id not in self.train_user_dict[u] and neg_i_id not in neg_items:
                    neg_items.append(neg_i_id)


            return neg_items
        # 返回抽取的负样本项目列表，直到列表中包含了指定数量的项目为止。
        pos_items, neg_items = [], []
        # 对每个用户抽取一个正样本项目和一个负样本项目，并将它们分别添加到 pos_items 和 neg_items 列表中。
        for u in users:
            pos_items += sample_pos_items_for_u(u, 1)
            neg_items += sample_neg_items_for_u(u, 1)
        return users, pos_items, neg_items


    # 生成用于训练对比学习(Contrastive Learning)的批次数据
    # 据给定的批次大小和现有的物品集合，从中随机选择一组物品作为批次
    def _generate_train_cl_batch(self):
        # 这一行代码检查对比学习的批次大小 self.batch_size_cl 是否小于等于现有的物品数量 self.exist_items。
        # 这是为了确保从物品集合中采样时不会超过可用的物品数量。
        if self.batch_size_cl <= len(self.exist_items):
            # 如果批次大小小于或等于现有物品数量，则从 self.exist_items 中随机采样 self.batch_size_cl 个物品
            items = rd.sample(self.exist_items, self.batch_size_cl)
        else:
            # 处理批次大小大于现有物品的情况
            # 首先将 self.exist_items 转换为列表，以确保可以使用索引操作。
            items_list = list(self.exist_items)
            #  items_list 中选择 self.batch_size_cl 次。此处允许重复选择，因此可能会多次选择相同的物品。
            items = [rd.choice(items_list) for _ in range(self.batch_size_cl)]
        return items
    
    def _generate_train_kg_batch(self):

        exist_heads = self.all_kg_dict.keys()

        # 从给定的字典中选择一组键，确保所选的键数量达到或接近所需的批处理大小，并且在键的数量不足时，通过随机选择来填充
        if self.batch_size_kg <= len(exist_heads):
            heads = rd.sample(exist_heads, self.batch_size_kg)
        else:
            heads = [rd.choice(exist_heads) for _ in range(self.batch_size_kg)]

        def sample_pos_triples_for_h(h, num):
            # 这一行代码从已知的知识图谱字典中获取实体 h 的所有正向三元组，并将它们存储在 pos_triples 中。
            pos_triples = self.all_kg_dict[h]
            # 这一行计算实体 h 的正向三元组数量，并将结果存储在 n_pos_triples 中。
            n_pos_triples = len(pos_triples)

            pos_rs, pos_ts = [], []
            while True:
                # 这个条件检查是否已经收集到了足够数量的关系。如果是，则退出循环。
                if len(pos_rs) == num: break
                # 随机选择一个正向三元组的索引，范围在0到 n_pos_triples-1 之间。
                pos_id = np.random.randint(low=0, high=n_pos_triples, size=1)[0]
                # 两行代码分别从选择的正向三元组中提取目标实体和关系，并将它们分别赋值给变量 t 和 r。
                t = pos_triples[pos_id][0]
                r = pos_triples[pos_id][1]
                # 这个条件检查是否已经选择了相同的关系或目标实体
                if r not in pos_rs and t not in pos_ts:
                    pos_rs.append(r)
                    pos_ts.append(t)
            return pos_rs, pos_ts

        def sample_neg_triples_for_h(h, r, num):
            # 这一行代码创建了一个空列表 neg_ts，用于存储生成的负向目标实体。
            neg_ts = []
            while True:
                if len(neg_ts) == num: break
                # 随机选择一个目标实体的索引，范围在0到 self.n_entities-1 之间，其中 self.n_entities 表示知识图谱中实体的总数。
                t = np.random.randint(low=0, high=self.n_entities, size=1)[0]
                # 这个条件检查生成的目标实体和给定关系 (t, r) 是否在知识图谱中。
                # 如果不在，并且目标实体不在已经生成的负向目标实体列表中，则将该目标实体添加到 neg_ts 列表中
                if (t, r) not in self.all_kg_dict[h] and t not in neg_ts:
                    neg_ts.append(t)
            return neg_ts
        
        pos_r_batch, pos_t_batch, neg_t_batch = [], [], []

        for h in heads:
            pos_rs, pos_ts = sample_pos_triples_for_h(h, 1)
            pos_r_batch += pos_rs
            pos_t_batch += pos_ts

            neg_ts = sample_neg_triples_for_h(h, pos_rs[0], 1)
            neg_t_batch += neg_ts

        return heads, pos_r_batch, pos_t_batch, neg_t_batch

    def generate_train_batch(self):
        
        users, pos_items, neg_items = self._generate_train_cf_batch()

        batch_data = {}
        batch_data['users'] = users
        batch_data['pos_items'] = pos_items
        batch_data['neg_items'] = neg_items
        return batch_data
        

    def generate_train_kg_batch(self):
        heads, relations, pos_tails, neg_tails = self._generate_train_kg_batch()

        batch_data = {}

        batch_data['heads'] = heads
        batch_data['relations'] = relations
        batch_data['pos_tails'] = pos_tails
        batch_data['neg_tails'] = neg_tails
        return batch_data
    
    def generate_train_cl_batch(self):
        items = self._generate_train_cl_batch()
        batch_data = {}
        batch_data['items'] = items
        return batch_data


