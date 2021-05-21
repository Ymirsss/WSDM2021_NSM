import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.sparse as sparse
import math
import time
VERY_NEG_NUMBER = -100000000000
VERY_SMALL_NUMBER = 1e-10

##主要实现4.2.2节实体的初始化和更新
#似乎文章只用到了TypeLayer（初始化） 没有用到STLayer（推理更新），因为在reasoning模块 又写了一遍推理过程
class TypeLayer(nn.Module):
    """
    Sparse version GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, linear_drop, device):
        super(TypeLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.linear_drop = linear_drop
        # self.kb_head_linear = nn.Linear(in_features, out_features)
        self.kb_self_linear = nn.Linear(in_features, out_features)
        # self.kb_tail_linear = nn.Linear(out_features, out_features)
        self.device = device

####论文4.2.2公式（2）------用relation初始化整个batch的entities
    def forward(self, local_entity, edge_list, rel_features):
        '''
        input_vector/local_entity: (batch_size, max_local_entity)
        curr_dist: (batch_size, max_local_entity)
        instruction: (batch_size, hidden_size)
        #rel_features是relation embedding
        '''
        batch_heads, batch_rels, batch_tails, batch_ids, fact_ids, weight_list = edge_list
        num_fact = len(fact_ids)
        batch_size, max_local_entity = local_entity.size()
        hidden_size = self.in_features
        fact2head = torch.LongTensor([batch_heads, fact_ids]).to(self.device)
        fact2tail = torch.LongTensor([batch_tails, fact_ids]).to(self.device)
        batch_rels = torch.LongTensor(batch_rels).to(self.device)
        batch_ids = torch.LongTensor(batch_ids).to(self.device)
        val_one = torch.ones_like(batch_ids).float().to(self.device)

        # print("Prepare data:{:.4f}".format(time.time() - st))
        # Step 1: Calculate value for every fact with rel and head

        fact_rel = torch.index_select(rel_features, dim=0, index=batch_rels)##获取batch数据的relation embedding
        # fact_val = F.relu(self.kb_self_linear(fact_rel) + self.kb_head_linear(self.linear_drop(fact_ent)))
        ###这里应该是4.2.2节公式（2）的Wt那个参数,论文中是集成relation的时候做的，这里是先把所有relation*Wt，再集成邻居以初始化实体
        fact_val = self.kb_self_linear(fact_rel)##relation embedding进行线性变换
        # fact_val = self.kb_self_linear(fact_rel)#self.kb_head_linear(self.linear_drop(fact_ent))

        # Step 3: Edge Aggregation with Sparse MM
        fact2tail_mat = self._build_sparse_tensor(fact2tail, val_one, (batch_size * max_local_entity, num_fact))#batchsize个数量的所有的实体的head和facts的对应矩阵
        fact2head_mat = self._build_sparse_tensor(fact2head, val_one, (batch_size * max_local_entity, num_fact))

        # neighbor_rep = torch.sparse.mm(fact2tail_mat, self.kb_tail_linear(self.linear_drop(fact_val)))
        #下面这一句的意思是将以当前实体e为head和tail的fact的relation集合起来（注意先前已经对relation做了Wt*r的操作了）
        #只不过是一下子初始化所有的entities，所以有点看不懂

        ###注意！！！这里初始化用到的是出边和入边邻居，而后续forward推理的时候，只用到了入边邻居
        f2e_emb = F.relu(torch.sparse.mm(fact2tail_mat, fact_val) + torch.sparse.mm(fact2head_mat, fact_val))#torch.sparse.mm是矩阵乘法

        assert not torch.isnan(f2e_emb).any()

        f2e_emb = f2e_emb.view(batch_size, max_local_entity, hidden_size)#pytorch里的view方法，用法类似resize

        return f2e_emb

    def _build_sparse_tensor(self, indices, values, size):
        return torch.sparse.FloatTensor(indices, values, size).to(self.device)#在构建一个size大小的稀疏张量，在indices提供的位置设置value

##更新实体表示
##4.4.2公式（3）（4）   但似乎没有公式（5）
class STLayer(nn.Module):
    """
    Sparse version GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, linear_drop, device):
        super(STLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.linear_drop = linear_drop
        # self.kb_head_linear = nn.Linear(in_features, out_features)
        ##这个是公式（3）的WR
        self.kb_self_linear = nn.Linear(in_features, out_features)
        # self.kb_tail_linear = nn.Linear(out_features, out_features)
        self.device = device

    def forward(self, input_vector, edge_list, curr_dist, instruction, rel_features):
        '''
        input_vector: (batch_size, max_local_entity, hidden_size)
        curr_dist: (batch_size, max_local_entity)
        instruction: (batch_size, hidden_size)
        '''
        batch_heads, batch_rels, batch_tails, batch_ids, fact_ids, weight_list = edge_list
        num_fact = len(fact_ids)
        batch_size, max_local_entity, hidden_size = input_vector.size()#(batch_size, max_local_entity, hidden_size)
        # input_vector = input_vector.view(batch_size * max_local_entity, hidden_size)
        # fact2head = torch.LongTensor([batch_heads, fact_ids]).to(self.device)
        fact2tail = torch.LongTensor([batch_tails, fact_ids]).to(self.device)
        head2fact = torch.LongTensor([fact_ids, batch_heads]).to(self.device)
        # batch_heads = torch.LongTensor(batch_heads).to(self.device)
        # batch_tails = torch.LongTensor(batch_tails).to(self.device)
        batch_rels = torch.LongTensor(batch_rels).to(self.device)
        batch_ids = torch.LongTensor(batch_ids).to(self.device)
        val_one = torch.ones_like(batch_ids).float().to(self.device)

        # print("Prepare data:{:.4f}".format(time.time() - st))
        # Step 1: Calculate value for every fact with rel and head
        fact_rel = torch.index_select(rel_features, dim=0, index=batch_rels)#从张量的某个维度的指定位置选取数据。

        #选取batch个query对应的instruction vector
        fact_query = torch.index_select(instruction, dim=0, index=batch_ids)
        # fact_val = F.relu(self.kb_self_linear(fact_rel) + self.kb_head_linear(self.linear_drop(fact_ent)))

    ##这里就是4.2.2的公式（3）呀计算之后，下面公式已经变成 match vector m了
        fact_val = F.relu(self.kb_self_linear(fact_rel) * fact_query)
        # fact_val = self.kb_self_linear(fact_rel)#self.kb_head_linear(self.linear_drop(fact_ent))

        # Step 3: Edge Aggregation with Sparse MM
        head2fact_mat = self._build_sparse_tensor(head2fact, val_one, (num_fact, batch_size * max_local_entity))
        fact2tail_mat = self._build_sparse_tensor(fact2tail, val_one, (batch_size * max_local_entity, num_fact))

        #下面是在计算p（k-1）吧，就是公式（4）要用到的，p（k-1）就是上一跳推理的实体分布，为什么是curr_dist，因为是根据当前的推理下一跳的呀
        #为什么使用head2fact乘以curr_dist呢，因为下一跳一定是以当前的entities作为头的呀，
        #这里相当于得到了batch个fact，他们的head的distribution（越大，下一跳选择的relation越是这个fact的relation呀）
        fact_prior = torch.sparse.mm(head2fact_mat, curr_dist.view(-1, 1))
        # (num_fact, batch_size * max_local_entity) (batch_size * max_local_entity, 1) -> (num_fact, 1)

        # fact_val = fact_val * edge_e
        #这里相当于4.2.2节公式（4）的p(k-1)*m (公式（4）还没聚合邻居的版本)
        fact_val = fact_val * fact_prior
        # neighbor_rep = torch.sparse.mm(fact2tail_mat, self.kb_tail_linear(self.linear_drop(fact_val)))
        ##聚合邻居，得到更新后的下一跳表达
        #以下公式就是 计算batch中entity作为 ‘当前entity'的时候 聚合以当前entiity为tail的所有关系（即入边）
       ##注意！！！ ##为什么这里只聚合入边的relation info，而初始化时聚合了出边和入边relation info？
        ##因为，在推理时，肯定是下一跳的选择肯定取决于上一跳呀
        f2e_emb = torch.sparse.mm(fact2tail_mat, fact_val)
        assert not torch.isnan(f2e_emb).any()

        f2e_emb = f2e_emb.view(batch_size, max_local_entity, hidden_size)

        return f2e_emb

    def _build_sparse_tensor(self, indices, values, size):
        return torch.sparse.FloatTensor(indices, values, size).to(self.device)