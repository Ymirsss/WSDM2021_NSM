import torch
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import time
import numpy as np
from NSM.Modules.Instruction.base_instruction import BaseInstruction
VERY_SMALL_NUMBER = 1e-10
VERY_NEG_NUMBER = -100000000000

#返回query 每一个word对应的hidden state 的att和 全部K跳推理的instruction vector
class LSTMInstruction(BaseInstruction):

    def __init__(self, args, word_embedding, num_word):
        super(LSTMInstruction, self).__init__(args)
        self.word_embedding = word_embedding
        self.num_word = num_word
        self.encoder_def()
        entity_dim = self.entity_dim##注意relation的维度和question 编码后的维度都是entity_dim
        self.cq_linear = nn.Linear(in_features=2 * entity_dim, out_features=entity_dim)#公式（1）的第三个公式
        self.ca_linear = nn.Linear(in_features=entity_dim, out_features=1)
        for i in range(self.num_step):
            self.add_module('question_linear' + str(i), nn.Linear(in_features=entity_dim, out_features=entity_dim))

#node encoder
    def encoder_def(self):
        # initialize entity embedding
        word_dim = self.word_dim
        kg_dim = self.kg_dim
        kge_dim = self.kge_dim
        entity_dim = self.entity_dim
        self.node_encoder = nn.LSTM(input_size=word_dim, hidden_size=entity_dim,
                                    batch_first=True, bidirectional=False)
#编码问题
    def encode_question(self, query_text):
        batch_size = query_text.size(0)
        query_word_emb = self.word_embedding(query_text)  # batch_size, max_query_word, word_dim
        query_hidden_emb, (h_n, c_n) = self.node_encoder(self.lstm_drop(query_word_emb),
                                                         self.init_hidden(1, batch_size,#init_hidden参数返回初始化为0的参数张量，维度同输入参数（即LSTM参数）
                                                                          self.entity_dim))  # （LSTM参数：layer,batch_size,输入特征维度）1层, batch_size, entity_dim是输入的embedding维度（这里的entity_dim似乎是单词维度？）
        #query_hidden_emb是question送入lstm最后一个layer的h0-hn（横向） hn是每一层LSTM层最后一个t的hn（纵向）  cn就是每一层lstm里的最后一个t的细胞状态，形状和hn一致。
        self.instruction_hidden = h_n
        self.instruction_mem = c_n
        self.query_node_emb = h_n.squeeze(dim=0).unsqueeze(dim=1)  # batch_size, 1, entity_dim
        #总之 query_node_emb就是每个query经过lstm最后一个t的hidden state ——hn,后面要用到q=hn
        # squeeze如果dim指定的维度的值为1，则将该维度删除，若指定的维度值不为1，则返回原来的tensor，
        #unsqueeze给指定位置加上维度为1的维度
        self.query_hidden_emb = query_hidden_emb
        self.query_mask = (query_text != self.num_word).float()
        return query_hidden_emb, self.query_node_emb

    def init_reason(self, query_text):
        batch_size = query_text.size(0)#返回第一维，即行数
        self.encode_question(query_text)
        self.relational_ins = torch.zeros(batch_size, self.entity_dim).to(self.device)
        self.instructions = []
        self.attn_list = []

#Insructor Component
    def get_instruction(self, relational_ins, step=0, query_node_emb=None):
        query_hidden_emb = self.query_hidden_emb
        query_mask = self.query_mask
        if query_node_emb is None:
            query_node_emb = self.query_node_emb
        relational_ins = relational_ins.unsqueeze(1)#在第二个位置增加一个维度 如（2，3）变为（2，1，3）
        question_linear = getattr(self, 'question_linear' + str(step))#question_linear是nn.linear网络
        q_i = question_linear(self.linear_drop(query_node_emb))#论文中question representation（公式（1）那里），本来是q=hl，但这里输出的可能是多层的hn，所以用一个线性网络
        cq = self.cq_linear(self.linear_drop(torch.cat((relational_ins, q_i), dim=-1)))#对应公式（1）中q(k)的计算（查询向量和instruction vector的att——instruction vector 关注问题的不同部分）
        # batch_size, 1, entity_dim
        ca = self.ca_linear(self.linear_drop(cq * query_hidden_emb))#对应公式（1）中alpha的计算(没加softmax)
        # batch_size, max_local_entity, 1
        # cv = self.softmax_d1(ca + (1 - query_mask.unsqueeze(2)) * VERY_NEG_NUMBER)#加了softmax后完整的alpha
        attn_weight = F.softmax(ca + (1 - query_mask.unsqueeze(2)) * VERY_NEG_NUMBER, dim=1)##？？后面这一坨都是优化吗
        # batch_size, max_local_entity, 1
        relational_ins = torch.sum(attn_weight * query_hidden_emb, dim=1)##最终的instructor
        return relational_ins, attn_weight

#上面的函数是一步的instructor构建网络，但当前instructor是以上一跳instructor作为输入的，这是一个forward过程
    def forward(self, query_text):
        self.init_reason(query_text)
        for i in range(self.num_step):##num_step是推理步骤嘛
            relational_ins, attn_weight = self.get_instruction(self.relational_ins, step=i)#每一个reason step都可以计算一个对应的instructor向量，对应关注q的不同重点
            self.instructions.append(relational_ins)
            self.attn_list.append(attn_weight)
            self.relational_ins = relational_ins
        return self.instructions, self.attn_list#最终所有step的instruction

    # def __repr__(self):
    #     return "LSTM + token-level attention"
