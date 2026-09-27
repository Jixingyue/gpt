# ============================================================
# 手写 GPT —— 一个简化版 GPT 语言模型（对话机器人）
# ------------------------------------------------------------
# 整个项目想做的事：
#   用一段"一问一答"的对话文本训练一个模型，
#   让模型学会"给定前面所有的字，预测下一个字"。
#   训练完成后，你输入一句话，模型会继续往下续写，直到写出 <sep> 结束符，
#   这段续写的内容就是模型生成的"回答"。
#
# 本文件包含：
#   1. 数据处理   —— 把对话文本转成模型能计算的数字序列
#   2. 模型搭建   —— 词嵌入 + 位置嵌入 + 多层 Transformer Decoder
#   3. 训练逻辑   —— 让模型学会"预测下一个字"
#   4. 对话交互   —— 用户输入问题，模型自动续写回答
#
# 运行：python 手写gpt.py （会自动先训练，再进入对话模式）
# ============================================================

import torch.utils.data as Data   # PyTorch 数据集与数据加载模块
from torch import nn, optim       # nn：搭建神经网络的工具箱；optim：优化器（Adam）
import numpy as np                # 科学计算库，用于生成上三角 mask 矩阵
import random                     # 随机数，用于"随机采样生成回答"
import time                       # 计时
import torch                      # PyTorch 主库
from tqdm import tqdm             # 训练时显示进度条

# ============================================================
# 一、数据集类：把一段段对话文本变成 (输入, 输出) 训练样本
# ============================================================
class MyDataSet(Data.Dataset):
    """自定义数据集。

    语言模型的训练方式是"自回归"：给模型前 n 个字，让模型预测第 n+1 个字。
    所以每一段对话都会被处理成：
        decoder_input  = 整段对话的字 id 序列去掉最后一个字
        decoder_output = 整段对话的字 id 序列去掉第一个字（即输入右移一位）
    模型看到 decoder_input，努力预测出 decoder_output，这就是"学习"。

    举个具体例子：假设对话"你好"转成 id 后是 [10, 20, 2, 2]（2 是 <sep>，表示"说完了"）：
        decoder_input  = [10, 20, 2]      # 去掉最后一个 2
        decoder_output = [20, 2, 2]      # 整体右移一位
    含义：看到 [10]        -> 应预测出 20（"你"的下一个字是"好"）
          看到 [10, 20]    -> 应预测出 2（"你好"说完了）
          看到 [10, 20, 2] -> 应预测出 2（整段对话结束）
    一次前向传播中，每个位置都在同时做"预测下一个字"，一起算损失、一起学习。
    """

    def __init__(self, datas):
        # datas：对话列表，每个元素是一段对话（多行文本，行与行之间是对话的轮流发言）
        self.datas = datas

    # 根据索引取出一个样本（PyTorch Dataset 要求必须实现）
    def __getitem__(self, item):
        text_data = self.datas[item]        # 取出一段对话
        text_data = text_data.split("\n")   # 按换行切分：每一行就是一句完整的话（一问或一答）

        text_idx = []                       # 用来存放整段对话转成的 token id 序列
        for data in text_data:              # 遍历对话中的每一句话
            # 把句子中的每个字转成 id；词表里没有的字用 1 表示（<unk> 未知字符）
            text_idx.extend([word_2_index.get(i, 1) for i in data])
            text_idx.append(2)              # 每句话结束后加一个 2（<sep> 分隔符），表示这句话说完了
        text_idx.append(2)                  # 整段对话最后再加一个 <sep>，作为"模型回答结束"的标志

        # 关键：构造"预测下一个字"的训练对
        decoder_input = text_idx[:-1]       # 输入 = 去掉最后一个字
        decoder_output = text_idx[1:]       # 输出 = 去掉第一个字（输入右移一位）

        decoder_input_len = len(decoder_input)    # 记录长度，供 padding_batch 使用
        decoder_output_len = len(decoder_output)

        # 返回一个字典（后续 padding_batch 会拿它来做长度补齐）
        return {"decoder_input": decoder_input, "decoder_input_len": decoder_input_len,
                "decoder_output": decoder_output, "decoder_output_len": decoder_output_len}

    def __len__(self):
        # 返回数据集里一共有多少段对话（DataLoader 靠它知道要取几个样本）
        return len(self.datas)

    # collate_fn：DataLoader 在取出一批样本后调用它，
    # 把一批"长度不同"的句子统一填充到相同长度，变成规则的矩阵才能并行计算
    def padding_batch(self, batch):
        decoder_input_lens = [d["decoder_input_len"] for d in batch]    # 每个样本输入的长度
        decoder_output_lens = [d["decoder_output_len"] for d in batch]  # 每个样本输出的长度

        decoder_input_maxlen = max(decoder_input_lens)    # 本批次输入的最大长度
        decoder_output_maxlen = max(decoder_output_lens)

        for d in batch:  # 把长度不足的样本用 <pad>(id=0) 补到和本批次最长的一样
            d["decoder_input"].extend([word_2_index["<pad>"]] * (decoder_input_maxlen - d["decoder_input_len"]))
            d["decoder_output"].extend([word_2_index["<pad>"]] * (decoder_output_maxlen - d["decoder_output_len"]))
        decoder_inputs = torch.tensor([d["decoder_input"] for d in batch], dtype=torch.long)   # 转成 torch.long 张量
        decoder_outputs = torch.tensor([d["decoder_output"] for d in batch], dtype=torch.long)

        # 返回形状 [batch_size, max_len] 的输入和标签
        return decoder_inputs, decoder_outputs


# ============================================================
# 二、两个 Mask（掩码）工具函数
# ------------------------------------------------------------
# Mask 的作用是"遮住不该看的位置"，让注意力机制只能看到允许看的地方。
# ============================================================

def make_pad_mask(seq_q, seq_k):
    """padding 掩码：把 <pad>(id=0) 的位置遮住。

    原因：句子被填充后的那些空白位置没有真实含义，不能让模型"注意"到它们。
    seq_q: 查询序列 [batch, len_q]，seq_k: 键序列 [batch, len_k]
    返回 [batch, len_q, len_k] 的布尔矩阵，seq_k 中是 0（即 <pad>）的位置为 True。

    例子：seq_k = [5, 8, 0, 0]（后两个 0 是填充）且 len_q = 4 时，返回：
        [[F, F, T, T],
         [F, F, T, T],
         [F, F, T, T],
         [F, F, T, T]]     <- 第 3、4 列是 True：所有字都禁止去"注意"填充位
    """
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)   # 找出所有等于 0（<pad>）的位置，升一维方便广播
    return pad_attn_mask.expand(batch_size, len_q, len_k)  # 扩展成 [batch, len_q, len_k]


def make_lookahead_mask(seq):
    """因果掩码（上三角矩阵）：让模型"不能偷看未来的字"。

    因为 GPT 是一个一个往后预测的，预测第 n 个字时，只能看到它前面的字，
    不能看到后面的字（否则就作弊了）。
    返回 [batch, seq_len, seq_len] 的矩阵，上三角（k=1 以上）位置为 1（需要被遮住）。

    例子：长度为 3 的序列返回：
        [[0, 1, 1],
         [0, 0, 1],
         [0, 0, 0]]     <- 第 1 行：预测第 1 个字时，第 2、3 个字（"未来"）被遮住
    """
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]          # [batch, len, len]
    subsequence_mask = np.triu(np.ones(attn_shape), k=1)          # 生成上三角全 1 矩阵
    subsequence_mask = torch.from_numpy(subsequence_mask).byte()  # 转成 torch 的 byte 张量
    subsequence_mask = subsequence_mask.to(device)                # 放到模型所在的设备（GPU/CPU）
    return subsequence_mask


# ============================================================
# 三、模型组件
# ============================================================

class MultiHeadAttention(nn.Module):
    """多头自注意力机制（Transformer 的核心）。

    一句话理解注意力：每个字都去"看"一遍序列里所有的字，
    决定自己在表达时，应该重点参考哪些字。
    多头 = 同时用多组不同的"看法"（多个头），让模型看到更多角度。
    """

    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        # 三个线性层分别把输入映射成 Q（查询）、K（键）、V（值）
        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=False)  # 查询投影
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=False)  # 键投影
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=False)  # 值投影
        self.fc = nn.Linear(n_heads * d_v, d_model, bias=False)   # 把多头结果合并回原维度
        self.layernorm = nn.LayerNorm(d_model)                    # 层归一化（稳定训练）

    def forward(self, input_x, attn_mask):
        residual, batch_size = input_x, input_x.size(0)  # 保存残差（用于后面的残差连接）

        # 把输入分别投影成 Q/K/V，并拆成多个头：
        # 输入 [batch, len, d_model] -> [batch, len, n_heads, d_k] -> [batch, n_heads, len, d_k]
        Q = self.W_Q(input_x).view(batch_size, -1, n_heads, d_k).transpose(1, 2)
        K = self.W_K(input_x).view(batch_size, -1, n_heads, d_k).transpose(1, 2)
        V = self.W_V(input_x).view(batch_size, -1, n_heads, d_v).transpose(1, 2)

        # mask 要作用于每个头，所以复制 n_heads 份： [batch, n_heads, len_q, len_k]
        attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1)

        # 缩放点积注意力得分：Q 和 K 做内积，除以 sqrt(d_k) 防止数值过大
        # 形状变化：Q [b, h, len, 64] x K^T [b, h, 64, len] -> scores [b, h, len_q, len_k]
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)
        scores.masked_fill_(attn_mask, -1e9)   # 被 mask 的位置填一个极小值，softmax 后概率趋近于 0

        attn = nn.Softmax(dim=-1)(scores)      # 对最后一个维度做 softmax -> 注意力权重，每行加起来等于 1
        context = torch.matmul(attn, V)         # 用权重对 V 做加权求和 -> [b, h, len_q, 64]，每个字得到"融合了上下文"的新表示

        # 把多头结果拼回去并过一层线性层，再加残差、做层归一化
        context = context.transpose(1, 2).reshape(batch_size, -1, n_heads * d_v)
        output = self.fc(context)
        return self.layernorm(output + residual), attn   # 返回输出和注意力权重（用于可视化）


class FeedForward(nn.Module):
    """前馈神经网络层（对每个字独立做一次非线性变换）。

    作用：注意力负责"字与字之间交换信息"，前馈层负责"每个字内部做更深的计算"。
    """

    def __init__(self):
        super(FeedForward, self).__init__()
        self.fc = nn.Sequential(   # 一个两层的全连接网络，中间夹 ReLU 激活函数
            nn.Linear(d_model, d_ff, bias=False),
            nn.ReLU(),
            nn.Linear(d_ff, d_model, bias=False)
        )
        self.layernorm = nn.LayerNorm(d_model)

    def forward(self, inputs):
        residual = inputs                             # 残差连接：保留原始输入
        output = self.fc(inputs)                      # 前馈计算
        return self.layernorm(output + residual)      # 残差 + 层归一化


class DecoderBlock(nn.Module):
    """一个 Transformer Decoder 层（由 2 个注意力块 + 1 个前馈层组成）。

    说明：标准 GPT 每层是【带 mask 的注意力 + 前馈】；
    这里为了教学演示，拆成了两个注意力块，第一个带 mask，第二个不带 mask。
    """

    def __init__(self):
        super(DecoderBlock, self).__init__()
        self.attention_block1 = MultiHeadAttention()   # 第一个注意力：带 mask（防止偷看未来、忽略 pad）
        self.attention_block2 = MultiHeadAttention()   # 第二个注意力：不带 mask（全量自注意力）
        self.feed_forward = FeedForward()              # 前馈层

    def forward(self, dec_inputs, dec_self_attn_mask):
        # 第一次注意力：用带 mask 的注意力（遮住 pad 和未来位置）
        dec_inputs, _ = self.attention_block1.forward(dec_inputs, dec_self_attn_mask)

        # 第二次注意力：全 0 的 mask（什么都不遮，让字与字之间充分交流）
        no_mask = torch.zeros_like(dec_self_attn_mask, device=dec_inputs.device, requires_grad=False)
        dec_outputs, dec_self_attn = self.attention_block2.forward(dec_inputs, no_mask)

        dec_outputs = self.feed_forward(dec_outputs)   # 前馈层做深加工
        return dec_outputs, dec_self_attn              # 返回处理后的结果和注意力权重


class EmbeddingLayer(nn.Module):
    """嵌入层：把"字 id"和"位置"都变成向量，并相加。

    两个嵌入：
        tgt_emb —— 词嵌入：把每个字的 id 变成一个 d_model 维的向量（让计算机能"理解"这个字）
        pos_emb —— 位置嵌入：给每个位置一个向量，让模型知道字与字的先后顺序
    最终每个字的表示 = 词向量 + 位置向量。
    """

    def __init__(self):
        super().__init__()
        self.tgt_emb = nn.Embedding(vocab_size, d_model)  # 词嵌入表：大小 [词表大小, 768]
        self.pos_emb = nn.Embedding(max_pos, d_model)     # 位置嵌入表：大小 [最长位置数, 768]

    def forward(self, x):
        seq_len = x.size(1)                                    # 当前序列长度
        pos = torch.arange(seq_len, dtype=torch.long, device=device)   # 位置索引 [0,1,2,...]
        pos = pos.unsqueeze(0).expand_as(x)                    # 扩展到和输入一样的形状
        return self.tgt_emb(x) + self.pos_emb(pos)             # 词向量 + 位置向量


class Decoder(nn.Module):
    """GPT 的解码器主体：嵌入层 + 多个 DecoderBlock 堆叠。"""

    def __init__(self):
        super(Decoder, self).__init__()
        self.embedding = EmbeddingLayer()                              # 嵌入层
        self.layers = nn.ModuleList([DecoderBlock() for _ in range(n_layers)])  # 堆叠 n_layers 层

    def forward(self, dec_inputs):
        dec_outputs = self.embedding(dec_inputs)                       # 先做嵌入，得到 [batch, len, d_model]

        # 合并两个 mask，得到最终注意力掩码：
        dec_self_attn_pad_mask = make_pad_mask(dec_inputs, dec_inputs)             # 遮住 <pad> 的位置
        dec_self_attn_subsequence_mask = make_lookahead_mask(dec_inputs)           # 遮住"未来"的位置
        dec_self_attn_mask = torch.gt((dec_self_attn_pad_mask + dec_self_attn_subsequence_mask), 0)  # 只要任一为真就遮住

        dec_self_attns = []   # 收集每层的注意力权重（可用于可视化，本代码暂未使用）
        for layer in self.layers:
            # dec_outputs: [batch, len, d_model]，dec_self_attn: [batch, n_heads, len, len]
            dec_outputs, dec_self_attn = layer(dec_outputs, dec_self_attn_mask)
            dec_self_attns.append(dec_self_attn)

        return dec_outputs, dec_self_attns


class GPT_Model(nn.Module):
    """完整的 GPT 模型：Decoder + 输出映射层 + 损失函数。"""

    def __init__(self):
        super(GPT_Model, self).__init__()
        self.decoder = Decoder()                                  # 解码器主体
        self.projection = nn.Linear(d_model, vocab_size)          # 把隐藏向量映射成"每个字的得分"
        self.loss_fun = nn.CrossEntropyLoss(ignore_index=0)       # 交叉熵损失；ignore_index=0 表示 <pad> 不参与计算损失

    def forward(self, dec_inputs, label=None):
        # 输入 [batch, len] -> Decoder -> [batch, len, d_model] -> 投影 -> [batch, len, vocab_size]
        dec_outputs, dec_self_attns = self.decoder.forward(dec_inputs)
        dec_logits = self.projection(dec_outputs)                 # 每个位置对每个字的得分（logits）

        if label is not None:
            # 训练模式：把预测和真实标签展平成一行，计算损失
            loss = self.loss_fun(dec_logits.reshape(-1, dec_logits.shape[-1]), label.reshape(-1))
            return loss
        else:
            # 推理模式：直接返回每个字的得分和注意力权重
            return dec_logits.view(-1, dec_logits.size(-1)), dec_self_attns

    # ---------- 生成回答的方式一：贪心解码（每步选概率最大的字） ----------
    def greedy_decoder(self, dec_input):
        """输入一个句子（转成 id），一个一个字地往后预测，直到预测出 <sep> 结束。

        原理：把"当前的整句话"喂给模型，模型对最后一个位置的预测就是"下一个字"；
              把这个字拼到句尾，再喂回去……如此循环，直到吐出 <sep> 或超过 100 字。
        """
        terminal = False
        start_dec_len = len(dec_input[0])   # 记录初始长度，防止无限循环

        while not terminal:
            # 安全阀：如果已经续写了超过 100 个字还没结束，就强制加 <sep> 结束
            if len(dec_input[0]) - start_dec_len > 100:
                next_symbol = word_2_index['<sep>']
                dec_input = torch.cat(
                    [dec_input.detach(), torch.tensor([[next_symbol]], dtype=dec_input.dtype, device=device)], -1)
                break

            dec_outputs, _ = self.decoder(dec_input)        # 前向计算
            projected = self.projection(dec_outputs)        # 得到每个字位置的得分 [1, len, vocab_size]

            prob = projected.squeeze(0).max(dim=-1, keepdim=False)[1]  # 每个位置取概率最大的字的 id
            next_word = prob.data[-1]                       # 只看最后一个位置，即"下一个要生成的字"
            next_symbol = next_word
            if next_symbol == word_2_index["<sep>"]:        # 如果生成的是 <sep>，对话结束
                terminal = True

            # 把新预测出的字拼到序列末尾，继续下一轮预测
            # detach()：断开与计算图的连接（推理阶段不需要梯度）；cat：在序列末尾拼接新字
            dec_input = torch.cat([dec_input.detach(), torch.tensor([[next_symbol]], dtype=dec_input.dtype, device=device)], -1)

        return dec_input    # 返回完整序列（原句 + 模型续写的部分）

    # ---------- 生成回答的方式二：随机采样解码（带一点随机性，回答更多样） ----------
    def random_decoder(self, dec_input, top_n):
        """与贪心解码类似，但不是"每步选最大"，而是按概率随机抽样，使回答更丰富不呆板。"""
        terminal = False
        start_dec_len = len(dec_input[0])
        # 一直预测下一个字，直到预测到 <sep> 结束；超过 100 字强制结束并补 <sep>
        while not terminal:
            if len(dec_input[0]) - start_dec_len > 100:
                next_symbol = word_2_index['<sep>']
                dec_input = torch.cat(
                    [dec_input.detach(), torch.tensor([[next_symbol]], dtype=dec_input.dtype, device=device)], -1)
                break

            # 前向计算；推理阶段不需要梯度，用 no_grad 省内存
            with torch.no_grad():
                dec_outputs, _ = self.decoder(dec_input)
                projected = self.projection(dec_outputs)   # [1, len, vocab_size]

            # 下面做"按概率采样"（轮盘赌法）：
            # 直观理解：把候选字的得分当成轮盘上的格子大小，得分越高格子越大；
            # 随机扔一次骰子，落在哪个格子就选哪个字 —— 既保留随机性，又偏向高分字
            a = projected.to('cpu')             # [1, len, vocab_size] 移到 CPU 上操作
            b = a.squeeze(0)[-1]                # 只取最后一个位置： [vocab_size] 每个字的得分
            c, idx1 = torch.sort(b, descending=True)   # 从大到小排序：c=得分，idx1=对应的字 id
            c = np.array(c[:top_n]) ** 2        # 只保留概率最高的前 top_n 个，并平方放大差距（高分字更容易被抽中）
            idx1 = np.array(idx1[:top_n])       # 对应的字 id

            sum = 0                             # 计算前 top_n 个得分的总和
            for i in c:
                sum += i

            d = sum * random.uniform(0, 1)      # 随机取 [0, sum] 之间的一个数
            for i, j in enumerate(c):           # 依次累减，落入哪个区间就选哪个字（类似"轮盘赌"）
                d -= j
                if d <= 0:
                    next_word = idx1[i]         # 选中的字 id
                    break

            next_symbol = next_word
            if next_symbol == word_2_index["<sep>"]:   # 预测到 <sep> 则结束
                terminal = True

            dec_input = torch.cat(
                [dec_input.detach(), torch.tensor([[next_symbol]], dtype=dec_input.dtype, device=device)], -1)

        return dec_input    # [1, 原长度 + 续写长度]

    # ---------- 对外接口：输入一句话，返回模型生成的回答 ----------
    def answer(self, sentence):
        # 把用户输入的句子转成 id 序列；\t 是"提问结束"的标志，转成 <sep>
        dec_input = [word_2_index.get(word, 1) if word != '\t' else word_2_index['<sep>'] for word in sentence]
        dec_input = torch.tensor(dec_input, dtype=torch.long, device=device).unsqueeze(0)   # 变成 [1, len]

        output = self.greedy_decoder(dec_input).squeeze(0)   # 贪心解码生成回答（也可以换成 random_decoder）
        # output = self.random_decoder(dec_input, top_n=3).squeeze(0)  # 想用随机采样就换成这行

        out = [index_2_word[int(id)] for id in output]       # 把 id 还原成字

        # 找出所有 <sep> 的位置：
        # 第一个 <sep> 是提问结束，最后一个 <sep> 是回答结束，
        # 两者之间就是模型生成的回答内容
        # 例：out = [你,好,<sep>,我,很,好,<sep>] -> 提取出的 answer = "我很好"
        sep_indexs = []
        for i in range(len(out)):
            if out[i] == "<sep>":
                sep_indexs.append(i)

        answer = out[sep_indexs[-2] + 1:-1]   # 取倒数第二个 <sep> 之后、最后一个 <sep> 之前的内容
        answer = "".join(answer)
        return answer


# ============================================================
# 四、读取训练数据
# ============================================================
def read_data(path, num=None):
    """读取训练数据。数据格式：每两段对话之间用空行分隔，一段对话里每行是一句话。"""
    with open(path, encoding="utf-8") as f:
        all_data = f.read().split("\n\n")   # 按空行切分，得到一段段对话

    if num:                                 # num 指定只取前 num 段对话（小规模快速训练用）
        return all_data[:-1][:num]
    else:
        return all_data[:-1]                # 去掉最后一个空块，返回全部对话


# ============================================================
# 五、主程序：配置 -> 数据 -> 模型 -> 训练 -> 对话
# ============================================================
if __name__ == "__main__":
    # ----- 1. 读取训练数据（这里只取 30 段对话，方便快速跑通；想用全部数据就把 30 删掉） -----
    t_data = read_data("data/train.txt", 30)

    # ----- 2. 读取词表（vocab.txt 每行一个字，第 0 行是 <pad>，第 1 行是 <unk>，第 2 行是 <sep>） -----
    with open("data/vocab.txt", encoding="utf-8") as f:
        index_2_word = f.read().split("\n")          # 索引 -> 字
    word_2_index = {w: i for i, w in enumerate(index_2_word)}   # 字 -> 索引（查表用）
    vocab_size = len(word_2_index)                   # 词表大小

    # ----- 3. 模型超参数（模型的"设计尺寸"，决定了模型的大小和能力） -----
    max_pos = 300        # 支持的最大序列长度（位置嵌入表 300 行；对话总长超过 300 会越界报错）
    d_model = 768        # 每个字的向量维度（隐藏层大小，与 GPT-2 small 相同）
    d_ff = 2048          # 前馈层的中间维度（先把 768 放大到 2048 再缩回来，通常约为 d_model 的 2~4 倍）
    d_k = d_v = 64       # 每个注意力头里 Q/K/V 的维度；768 = 8 个头 x 64，多头拼起来正好回到 768
    n_layers = 6         # Decoder 层数（模型有多"深"）
    n_heads = 8          # 注意力头数（8 组不同"视角"并行看序列）
    CLIP = 1             # 梯度裁剪上限（梯度总长超过 1 就等比缩小，防止梯度爆炸导致训练发散）
    batch_size = 1       # 每批 1 段对话（对话长短不一，取 1 最简单稳妥）
    epochs = 10          # 训练轮数（把 30 段对话完整过 10 遍）
    device = "cuda" if torch.cuda.is_available() else "cpu"   # 有 GPU 用 GPU，否则用 CPU

    # ----- 4. 构建数据集和加载器 -----
    dataset = MyDataSet(t_data)
    data_loader = Data.DataLoader(dataset, batch_size=batch_size, collate_fn=dataset.padding_batch)

    # ----- 5. 创建模型和优化器 -----
    model = GPT_Model().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)   # Adam 优化器，学习率 1e-4

    # ----- 6. 训练循环 -----
    for epoch in range(epochs):
        epoch_loss = 0
        for i, (batch_text, batch_label) in enumerate(tqdm(data_loader)):
            batch_text = batch_text.to(device)      # 输入序列
            batch_label = batch_label.to(device)    # 目标序列（每个位置应预测的字）

            loss = model.forward(batch_text, batch_label)   # 前向计算损失

            epoch_loss += loss.item()
            loss.backward()                                 # 反向传播计算梯度
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)  # 梯度裁剪
            optimizer.step()                                # 更新参数
            optimizer.zero_grad()                           # 清空梯度，准备下一批
        end_time = time.time()   # 原代码遗留的计时语句（本代码中没有使用）

        # 每轮结束打印一次平均损失（越小说明模型学得越好）
        train_loss = epoch_loss
        print(f'\tTrain Loss: {train_loss:.3f}')

    # 保存训练好的模型（取消注释即可保存）。
    # 注意 1：把文件名改成 'model.pt' 才能被 predict.py 直接加载。
    # 注意 2：本文件的模型结构（每层"双注意力"）与 predict.py（每层"单注意力"）不同，
    #         所以本文件训练出的权重无法直接给 predict.py 用；想用 predict.py 推理，
    #         需要按 predict.py 的结构训练出对应的 model.pt（详见 README.md"常见坑"）。
    # torch.save(model.state_dict(), r'model1.pt')

    # ----- 7. 进入对话模式：输入一句话，模型自动续写回答 -----
    model.eval()           # 切换为评估模式（关闭训练相关的行为）
    sentence = ''

    while True:
        sentence = ''
        temp_sentence = input("请输入:")
        sentence += (temp_sentence + '\t')      # 用 \t 标记"提问到此结束"
        if len(sentence) > 200:                 # 输入超 200 字时只保留第一个 \t 之后的部分（防止超出位置嵌入 300 的上限）
            t_index = sentence.find('\t')
            sentence = sentence[t_index + 1:]
        print("GPT_Model:", model.answer(sentence))
