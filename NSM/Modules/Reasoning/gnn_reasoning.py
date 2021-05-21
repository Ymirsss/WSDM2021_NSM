import torch
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import time
import numpy as np
from NSM.Modules.Reasoning.base_reasoning import BaseReasoning
VERY_SMALL_NUMBER = 1e-10
VERY_NEG_NUMBER = -100000000000


class GNNReasoning(BaseReasoning):

    def __init__(self, args, num_entity, num_relation):
        super(GNNReasoning, self).__init__(args, num_entity, num_relation)
        self.share_module_def()
        self.private_module_def()

    def private_module_def(self):
        entity_dim = self.entity_dim
        self.softmax_d1 = nn.Softmax(dim=1)
        self.score_func = nn.Linear(in_features=entity_dim, out_features=1)
        for i in range(self.num_step):
            self.add_module('rel_linear' + str(i), nn.Linear(in_features=entity_dim, out_features=entity_dim))
            self.add_module('e2e_linear' + str(i), nn.Linear(in_features=2 * entity_dim, out_features=entity_dim))
            # self.add_module('score_func' + str(i), nn.Linear(in_features=entity_dim, out_features=1))

    def reason_layer(self, curr_dist, instruction, rel_linear):
        batch_size = self.batch_size
        max_local_entity = self.max_local_entity
        # num_relation = self.num_relation
        rel_features = self.rel_features
        fact_rel = torch.index_select(rel_features, dim=0, index=self.batch_rels)
        fact_query = torch.index_select(instruction, dim=0, index=self.batch_ids)
        # fact_val = F.relu(self.kb_self_linear(fact_rel) + self.kb_head_linear(self.linear_drop(fact_ent)))
        ##根据当前跳的instruction集成relation info(这是m（k）)
        fact_val = F.relu(rel_linear(fact_rel) * fact_query)
        ##这就是p（k-1）呗
        fact_prior = torch.sparse.mm(self.head2fact_mat, curr_dist.view(-1, 1))
        ##这是公式未求和的（4）
        fact_val = fact_val * fact_prior
        # neighbor_rep = torch.sparse.mm(fact2tail_mat, self.kb_tail_linear(self.linear_drop(fact_val)))

        #这是求和的公式（4）
        f2e_emb = torch.sparse.mm(self.fact2tail_mat, fact_val)

        possible_tail = torch.sparse.mm(self.fact2tail_mat, fact_prior)
        # (batch_size *max_local_entity, num_fact) (num_fact, 1)
        possible_tail = (possible_tail > VERY_SMALL_NUMBER).float().view(batch_size, max_local_entity)
        assert not torch.isnan(f2e_emb).any()

        neighbor_rep = f2e_emb.view(batch_size, max_local_entity, self.entity_dim)
        return neighbor_rep, possible_tail

    def init_reason(self, local_entity, kb_adj_mat, local_entity_emb, rel_features, query_node_emb=None):
        batch_size, max_local_entity = local_entity.size()
        self.local_entity_mask = (local_entity != self.num_entity).float()
        self.batch_size = batch_size
        self.max_local_entity = max_local_entity
        self.edge_list = kb_adj_mat
        self.rel_features = rel_features
        self.local_entity_emb = local_entity_emb
        self.num_relation = self.rel_features.size(0)
        self.possible_cand = []
        self.build_matrix()

##进行一跳推理，返回推理得出的当前跳的实体分布
    def forward(self, current_dist, relational_ins, step=0, return_score=False):
        rel_linear = getattr(self, 'rel_linear' + str(step))
        e2e_linear = getattr(self, 'e2e_linear' + str(step))
        # score_func = getattr(self, 'score_func' + str(step))
        score_func = self.score_func
        relational_ins = relational_ins.squeeze(1)
        neighbor_rep, possible_tail = self.reason_layer(current_dist, relational_ins, rel_linear)
        ##公式（5）
        next_local_entity_emb = torch.cat((self.local_entity_emb, neighbor_rep), dim=2)
        self.local_entity_emb = F.relu(e2e_linear(self.linear_drop(next_local_entity_emb)))
##score_func是一个线性函数，应该是公式（6）的里面部分 score就是将entity embedding喂进一个线性网络，得到每个entity的score
        score_tp = score_func(self.linear_drop(self.local_entity_emb)).squeeze(dim=2)
        if self.reason_kb:
            answer_mask = self.local_entity_mask * possible_tail
        else:
            answer_mask = self.local_entity_mask
        self.possible_cand.append(answer_mask)
        score_tp = score_tp + (1 - answer_mask) * VERY_NEG_NUMBER

        ##完整的公式（6）
        current_dist = self.softmax_d1(score_tp)
        if return_score:
            return score_tp, current_dist
        return current_dist

    def forward_all(self, curr_dist, instruction_list):
        dist_history = [curr_dist]
        score_list = []
        for i in range(self.num_step):
            score_tp, curr_dist = self.forward(curr_dist, instruction_list[i], step=i, return_score=True)
            score_list.append(score_tp)
            dist_history.append(curr_dist)
        return dist_history, score_list

    # def __repr__(self):
    #     return "GNN based reasoning"
