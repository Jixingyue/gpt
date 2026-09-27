# ============================================================
# 模型预测（推理）脚本 —— 只加载训练好的模型.pt，直接和 GPT 对话
# ------------------------------------------------------------
# 与"手写gpt.py"的区别：
#   手写gpt.py = 训练 + 对话（运行时会重新训练模型）
#   predict.py = 对话（加载现成的 model.pt，跳过训练，更快）
#
# 运行：python predict.py   （会自动加载 model.pt 后进入对话模式）
#
# 注意：模型结构必须和训练时完全一致（这里用的是 DecoderLayer 单注意力版，
#       若与 model.pt 训练时结构不一致会加载失败，需改用对应结构）。
# ============================================================

import torch
import torch.utils.data as Data   # 数据加载（本文件实际未使用，可忽略）
from torch import nn, optim       # nn：网络模块；optim：优化器（本文件推理用不到）
import numpy as np                # 生成上三角 mask 矩阵
import random                     # 随机数（随机采样解码用）
import time                       # 计时
from tqdm import tqdm             # 进度条（本文件推理用不到，可忽略）

# 设备：有 GPU 用 GPU，否则用 CPU。放在文件顶部，让下面的函数都能拿到
device = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# 一、Mask（掩码）工具函数（含义同 手写gpt.py）
# ============================================================

def get_attn_pad_mask(seq_q, seq_k):
    """padding 掩码：把 <pad>(id=0) 的位置遮住，不让模型注意没有意义的填充位。"""
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()

    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)   # seq_k 中为 0（<pad>）的位置记为 True
    return pad_attn_mask.expand(batch_size, len_q, len_k)   # 扩展成 [batch, len_q, len_k]


def get_attn_subsequence_mask(seq):
    """因果掩码（上三角矩阵）：预测当前位置时不能偷看后面的字（未来）。"""
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
    subsequence_mask = np.triu(np.ones(attn_shape), k=1)    # 上三角全 1
    subsequence_mask = torch.from_numpy(subsequence_mask).byte()
    subsequence_mask = subsequence_mask.to(device)
    return subsequence_mask


# ============================================================
# 二、缩放点积注意力（单头版）
# ============================================================
class ScaledDotProductAttention(nn.Module):
    """缩放点积注意力：Q 和 K 算相似度 -> softmax 变成权重 -> 对 V 加权求和。"""

    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)  # 相似度得分，除以 sqrt(d_k) 防数值过大
        scores.masked_fill_(attn_mask, -1e9)                          # 被 mask 的位置填极小值，softmax 后约等于 0

        attn = nn.Softmax(dim=-1)(scores)     # 归一化成注意力权重
        context = torch.matmul(attn, V)       # 加权求和，得到每个位置融合后的表示
        return context, attn


# ============================================================
# 三、多头自注意力
# ============================================================
class MultiHeadAttention(nn.Module):
    """多头自注意力：把输入拆成多个"头"分别做注意力，再合并，最后残差+归一化。"""

    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=False)  # Q 投影
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=False)  # K 投影
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=False)  # V 投影
        self.fc = nn.Linear(n_heads * d_v, d_model, bias=False)   # 合并多头输出
        self.layernorm = nn.LayerNorm(d_model)                    # 层归一化

    def forward(self, input_Q, input_K, input_V, attn_mask):
        residual, batch_size = input_Q, input_Q.size(0)   # 残差 = 原始输入

        # 投影并拆成多个头：[batch, len, d_model] -> [batch, n_heads, len, d_k/d_v]
        Q = self.W_Q(input_Q).view(batch_size, -1, n_heads, d_k).transpose(1, 2)
        K = self.W_K(input_K).view(batch_size, -1, n_heads, d_k).transpose(1, 2)
        V = self.W_V(input_V).view(batch_size, -1, n_heads, d_v).transpose(1, 2)

        attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1)  # mask 复制到每个头

        context, attn = ScaledDotProductAttention()(Q, K, V, attn_mask)  # 多头注意力计算
        context = context.transpose(1, 2).reshape(batch_size, -1, n_heads * d_v)  # 把多头拼回去
        output = self.fc(context)                                    # 线性变换
        return self.layernorm(output + residual), attn               # 残差 + 归一化


# ============================================================
# 四、前馈神经网络层
# ============================================================
class PoswiseFeedForwardNet(nn.Module):
    """逐位置前馈网络：对每个字独立做两层全连接 + ReLU，最后残差 + 归一化。"""

    def __init__(self):
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.ReLU(),
            nn.Linear(d_ff, d_model, bias=False)
        )
        self.layernorm = nn.LayerNorm(d_model)

    def forward(self, inputs):
        '''
        inputs: [batch_size, seq_len, d_model]
        '''
        residual = inputs
        output = self.fc(inputs)
        return self.layernorm(output + residual)


# ============================================================
# 五、单个 Decoder 层
# ============================================================
class DecoderLayer(nn.Module):
    """一个 Decoder 层 = 自注意力（带 mask）+ 前馈网络。

    说明：这是"标准版"写法（一个注意力块），和 手写gpt.py 里
    用两个注意力块的"教学演示版"在结构上略有不同，注意区分。
    """

    def __init__(self):
        super(DecoderLayer, self).__init__()
        self.dec_self_attn = MultiHeadAttention()   # 自注意力（Q=K=V=输入自己）
        self.pos_ffn = PoswiseFeedForwardNet()      # 前馈层

    def forward(self, dec_inputs, dec_self_attn_mask):
        # 自注意力：Q/K/V 都是 dec_inputs 自己，并带上 mask
        dec_outputs, dec_self_attn = self.dec_self_attn.forward(dec_inputs, dec_inputs, dec_inputs, dec_self_attn_mask)
        dec_outputs = self.pos_ffn(dec_outputs)     # 前馈层
        return dec_outputs, dec_self_attn


# ============================================================
# 六、Decoder 主体
# ============================================================
class Decoder(nn.Module):
    """词嵌入 + 位置嵌入 + 堆叠 N 个 DecoderLayer。"""

    def __init__(self):
        super(Decoder, self).__init__()
        self.tgt_emb = nn.Embedding(vocab_size, d_model)  # 词嵌入表
        self.pos_emb = nn.Embedding(max_pos, d_model)     # 位置嵌入表
        self.layers = nn.ModuleList([DecoderLayer() for _ in range(n_layers)])

    def forward(self, dec_inputs):
        seq_len = dec_inputs.size(1)
        pos = torch.arange(seq_len, dtype=torch.long, device=device)  # 位置索引 [0,1,2,...]
        pos = pos.unsqueeze(0).expand_as(dec_inputs)

        dec_outputs = self.tgt_emb(dec_inputs) + self.pos_emb(pos)  # 词向量 + 位置向量

        # 合并两个 mask：遮住 <pad> + 遮住未来
        dec_self_attn_pad_mask = get_attn_pad_mask(dec_inputs, dec_inputs)
        dec_self_attn_subsequence_mask = get_attn_subsequence_mask(dec_inputs)
        dec_self_attn_mask = torch.gt((dec_self_attn_pad_mask + dec_self_attn_subsequence_mask), 0)

        dec_self_attns = []   # 收集各层注意力（可用于可视化）
        for layer in self.layers:
            dec_outputs, dec_self_attn = layer(dec_outputs, dec_self_attn_mask)
            dec_self_attns.append(dec_self_attn)

        return dec_outputs, dec_self_attns


# ============================================================
# 七、完整 GPT 模型（仅推理，无损失函数）
# ============================================================
class GPT(nn.Module):
    """GPT = Decoder + 输出映射层。输入一串 id，输出每个位置对每个字的得分。"""

    def __init__(self):
        super(GPT, self).__init__()
        self.decoder = Decoder()
        self.projection = nn.Linear(d_model, vocab_size)  # 隐藏向量 -> 每个字的得分

    def forward(self, dec_inputs):
        dec_outputs, dec_self_attns = self.decoder.forward(dec_inputs)   # [batch, len, d_model]
        dec_logits = self.projection(dec_outputs)                        # [batch, len, vocab_size]
        return dec_logits.view(-1, dec_logits.size(-1)), dec_self_attns

    # ---------- 贪心解码：每步选概率最大的字 ----------
    def greedy_decoder(self, dec_input):
        """从给定句子开始，一个一个字往后预测，直到预测出 <sep> 为止。"""
        terminal = False
        start_dec_len = len(dec_input[0])

        while not terminal:
            # 安全阀：续写超过 100 个字仍未结束则强制补 <sep>
            if len(dec_input[0]) - start_dec_len > 100:
                next_symbol = word2id['<sep>']
                dec_input = torch.cat(
                    [dec_input.detach(), torch.tensor([[next_symbol]], dtype=dec_input.dtype, device=device)], -1)
                break

            dec_outputs, _ = self.decoder(dec_input)
            projected = self.projection(dec_outputs)   # [1, len, vocab_size]

            prob = projected.squeeze(0).max(dim=-1, keepdim=False)[1]  # 每个位置概率最大的字 id
            next_word = prob.data[-1]                  # 取最后一个位置（即下一个要生成的字的 id）
            next_symbol = next_word
            if next_symbol == word2id["<sep>"]:        # 生成 <sep> 表示回答结束
                terminal = True

            dec_input = torch.cat(
                [dec_input.detach(), torch.tensor([[next_symbol]], dtype=dec_input.dtype, device=device)], -1)

        return dec_input    # 原句 + 模型续写内容

    # ---------- 随机采样解码：按概率随机选字，回答更多样 ----------
    def random_decoder(self, dec_input, top_n):
        """和贪心解码类似，但不是每次选最大，而是按概率随机采样（top_n 个候选）。"""
        terminal = False
        start_dec_len = len(dec_input[0])

        while not terminal:
            if len(dec_input[0]) - start_dec_len > 100:   # 安全阀，超过 100 字强制结束
                next_symbol = word2id['<sep>']
                dec_input = torch.cat(
                    [dec_input.detach(), torch.tensor([[next_symbol]], dtype=dec_input.dtype, device=device)], -1)
                break

            # 推理时不需要梯度，用 no_grad 省内存
            with torch.no_grad():
                dec_outputs, _ = self.decoder(dec_input)
                projected = self.projection(dec_outputs)   # [1, len, vocab_size]

            # 按概率采样（轮盘赌）：
            a = projected.to('cpu')            # 移到 CPU 操作
            b = a.squeeze(0)[-1]               # 只取最后一个位置 [vocab_size]
            c, idx1 = torch.sort(b, descending=True)   # 得分从大到小排序，idx1 是对应的字 id
            c = np.array(c[:top_n]) ** 2       # 保留前 top_n 个，平方放大概率差距
            idx1 = np.array(idx1[:top_n])

            sum = 0                            # 前 top_n 个得分之和
            for i in c:
                sum += i

            d = sum * random.uniform(0, 1)     # 随机取 [0, sum] 的数
            for i, j in enumerate(c):          # 依次累减，落入哪个区间就选哪个字
                d -= j
                if d <= 0:
                    next_word = idx1[i]
                    break

            next_symbol = next_word
            if next_symbol == word2id["<sep>"]:   # 生成 <sep> 结束
                terminal = True

            dec_input = torch.cat(
                [dec_input.detach(), torch.tensor([[next_symbol]], dtype=dec_input.dtype, device=device)], -1)

        return dec_input

    # ---------- 对外接口：输入一句话，返回模型生成的回答 ----------
    def answer(self, sentence):
        # 把句子转成 id 序列；\t 表示"提问结束"，转成 <sep>
        dec_input = [word2id.get(word, 1) if word != '\t' else word2id['<sep>'] for word in sentence]
        dec_input = torch.tensor(dec_input, dtype=torch.long, device=device).unsqueeze(0)

        output = self.greedy_decoder(dec_input).squeeze(0)   # 贪心解码生成回答

        out = [id2word[int(id)] for id in output]   # id 还原成字

        # 找 <sep> 的位置：提问结束的 <sep> 和回答结束的 <sep> 之间就是回答内容
        sep_indexs = []
        for i in range(len(out)):
            if out[i] == "<sep>":
                sep_indexs.append(i)

        answer = out[sep_indexs[-2] + 1:-1]
        answer = "".join(answer)
        return answer


# ============================================================
# 八、主程序：加载词表 -> 加载模型 -> 对话
# ============================================================
if __name__ == '__main__':
    # 0. 模型超参数（必须和训练 model.pt 时完全一致，否则权重加载不上）
    #    （数值对应 model.pt 的维度：词表 7548、d_model 768、max_pos 300、d_ff 2048、
    #      d_k=d_v=64、n_heads=8、n_layers=6，可用 torch.load 后查看 weight 形状核对）
    max_pos = 300        # 支持的最大序列长度（位置嵌入表的长度）
    d_model = 768        # 每个字的向量维度（隐藏层大小）
    d_ff = 2048          # 前馈层的中间维度
    d_k = d_v = 64       # 每个注意力头里 Q/K/V 的维度
    n_layers = 6         # Decoder 层数
    n_heads = 8          # 注意力头数

    # 1. 加载词表（必须和训练时用同一个 vocab.txt，否则字和 id 对应不上）
    with open("data/vocab.txt", encoding="utf-8") as f:
        id2word = f.read().split("\n")                       # 索引 -> 字
    word2id = {w: i for i, w in enumerate(id2word)}          # 字 -> 索引
    vocab_size = len(word2id)                                # 词表大小

    # 2. 创建模型并加载训练好的权重（model.pt 必须是训练时保存的）
    model = GPT().to(device)
    model.load_state_dict(torch.load('model.pt'))

    # 3. 进入对话模式
    model.eval()   # 切换为评估模式
    sentence = ''
    while True:
        sentence = ''
        temp_sentence = input("请输入:")
        sentence += (temp_sentence + '\t')      # 用 \t 标记提问结束
        if len(sentence) > 200:                 # 过长则只保留最近一段对话
            t_index = sentence.find('\t')
            sentence = sentence[t_index + 1:]
        print("GPT:", model.answer(sentence))
