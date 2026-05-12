# Calib-OPD Idea 审阅问题详细回答

本文档回答对 `/data/zhaoenyue/OPD/calib_opd_idea.md` 前七节审阅时提出的六个关键问题。整体结论是：原始 idea 需要从“普通 OPD 一定有问题”收敛为更严谨的研究问题：

> 普通 OPD 在可验证 RLVR 场景中可以通过 verifier 或 reward 抑制错误答案；但在 open-domain QA、RAG、长答案生成、claim-level factuality、不可回答问题等弱验证或不可验证场景中，普通 distribution matching 式 OPD 缺少显式的拒答、置信度和证据校准机制。Calib-OPD 的价值不是否定 OPD，而是研究并实现 calibrated imitation：什么时候该蒸馏 teacher，什么时候该拒答，什么时候该跳过或惩罚 student 的 hallucinated trajectory。

---

## 问题 1：普通 OPD 不能抑制错误答案吗？在 RLVR 场景中 student 回答是可验证的，Calib-OPD 是不是主要用于泛化到其他领域？

### 简短回答

普通 OPD 不是完全不能抑制错误答案。尤其在 RLVR 场景中，如果 student 的回答可以被 verifier 判断对错，那么可以通过以下方式避免学习错误答案：

```text
只在 correct rollout 上做 OPD；
wrong rollout 跳过、给负奖励，或不进入 distillation loss。
```

因此不能把论文主张写成：

```text
普通 OPD 无法抑制错误答案。
```

更准确的主张应该是：

```text
普通 OPD 在 verifiable tasks 中可以借助 verifier 过滤错误；
但在 weakly-verifiable / unverifiable / long-form factual generation 中，
普通 OPD 缺少校准知识边界、拒答和证据支持的机制。
```

### 分场景看

#### 1. 可验证 RLVR 场景

例如数学、代码、形式化推理、选择题：

- answer correctness 可以由程序、规则、单元测试或标准答案验证；
- student rollout 可以被判定 correct / wrong；
- wrong rollout 可以被丢弃或给负 reward；
- OPD 只在 correct rollout 上使用 teacher token-level supervision。

这种情况下，普通 OPD + verifier filtering 已经可以抑制很多错误答案。

所以 Calib-OPD 不能把贡献点放在“RLVR 中 OPD 完全不能处理错误答案”。这个说法会被质疑。

#### 2. 弱验证或不可验证场景

例如：

- open-domain factual QA；
- RAG with noisy retrieval；
- long-form answer；
- biography generation；
- medical / legal explanation；
- multi-hop factual reasoning；
- claim-level factuality；
- future questions / false-premise questions / unanswerable questions。

这些任务里，错误不是总能便宜、稳定地验证：

- 有些答案没有唯一标准答案；
- 有些 claim 是局部错误；
- 有些问题本身不该回答；
- RAG 文档可能支持部分 claim，但不支持完整回答；
- teacher 和 verifier 也可能不确定。

这正是 Calib-OPD 更适合切入的地方。

### 对 Calib-OPD 定位的修正

原 proposal 应该从：

> 普通 OPD 会放大 hallucination。

改成：

> 普通 OPD 在可验证任务上有效，但在不确定、不可回答或部分可验证的 factual generation 场景下，blind imitation 不足以保证 truthfulness。Calib-OPD 将 OPD 泛化为 calibration-aware imitation，使 student 同时学习答案、拒答、置信度和证据边界。

---

## 问题 2：为什么说普通 OPD 有问题？普通 OPD 的损失函数都是 KL 散度吗？如果换成其他 loss 还会有这个问题吗？

### 普通 OPD 不一定全是 KL，但多数是 distribution matching

OPD 的核心是：student 在自己的 on-policy trajectory 上接受 teacher 的 token-level supervision。

常见形式包括：

```text
Reverse KL:
D_KL(pi_s || pi_t)

Forward KL:
D_KL(pi_t || pi_s)

Symmetric KL / JS divergence

Teacher-token cross entropy

Top-k KL

Sampled-token OPD

Logit matching / MSE

Top-1 teacher target matching
```

2604.13016v2 主要分析的是 reverse KL、sampled-token OPD、top-k OPD，以及 overlap-token / non-overlap-token 的作用。2604.03128v1 在 OPSD 理论分析中强调，问题来自 information-asymmetric distribution matching，即 teacher conditioned on privileged information，而 student 不能看到这些信息。

因此，不能简单说“KL 是原罪”。

### 问题不在 KL 本身，而在 blind imitation

更准确的问题是：

```text
如果 loss 只要求 student match teacher distribution，
但没有显式判断当前 trajectory 是否正确、是否有证据、是否应该拒答，
那么这种 imitation objective 在 hallucination 场景中可能不够安全。
```

也就是说，风险来自：

1. student 的 on-policy trajectory 可能已经进入错误上下文；
2. teacher 在错误 prefix 下的 continuation 不一定能恢复全局正确性；
3. distribution matching 只优化局部 token 分布接近；
4. loss 本身不区分 correct answer、reasonable abstention、hallucination；
5. 如果 teacher 也不确定或错误，student 可能继承 teacher 的错误倾向。

### 换成其他 loss 是否仍有问题？

取决于 loss 是否仍然是 imitation-only。

如果换成：

```text
KL -> cross entropy
KL -> logit MSE
KL -> top-1 teacher imitation
KL -> sampled-token teacher reward
```

但仍然没有 correctness / evidence / confidence / abstention gating，那么核心问题仍可能存在。因为这些 loss 仍然在做 teacher imitation。

如果换成：

```text
verifier-filtered OPD
reward-weighted OPD
confidence-gated OPD
unlikelihood for wrong claims
abstention distillation
evidence-aware distillation
```

那就已经不是普通 OPD，而是 selective / calibrated / reward-aware OPD。

### 推荐论文表述

不要写：

```text
KL loss causes hallucination.
```

应该写：

```text
The issue is not KL divergence per se, but uncalibrated distribution matching. When teacher imitation is applied indiscriminately to uncertain or unsupported student trajectories, the objective lacks a mechanism to decide when not to imitate.
```

中文就是：

> 问题不是 KL 本身，而是未校准的模仿目标不知道什么时候不该模仿。

---

## 问题 3：如何证明普通 OPD 确实有问题？是不是需要分析型实验或对比实验？能否使用 teacher 和 student 都不能正确回答的 prompt？

### 简短回答

是的，必须做分析型实验。不能直接把“普通 OPD 强化过度自信和错误回答”作为前提。它应该是待验证假设。

最合适的实验不是只看最终 benchmark，而是构造分桶分析：

```text
A. teacher correct, student correct
B. teacher correct, student wrong
C. teacher uncertain/wrong, student uncertain
D. teacher uncertain/wrong, student overconfident wrong
```

其中 D 类最关键，因为它对应 hallucination 的核心风险：student 不会但很自信。

### 可以使用 teacher 和 student 都不能正确回答的 prompt

可以，而且很有价值。尤其是以下类型：

1. unanswerable questions；
2. future questions；
3. false-premise questions；
4. obscure long-tail facts；
5. retrieval documents 不支持答案的问题；
6. adversarial factual QA；
7. teacher 和 student 都低置信或都答错的问题。

但要注意细分：

#### 情况 1：teacher 和 student 都输出拒答

这种情况下普通 OPD 未必有问题。teacher 的行为本身是合理的，student 模仿拒答反而是好的。

#### 情况 2：teacher 和 student 都自信答错

这是最能暴露 blind imitation 风险的样本。普通 OPD 可能会强化错误答案或错误表达。

#### 情况 3：student 答错，teacher 能识别问题不可回答

这适合展示 Calib-OPD 的优势：不继续模仿错误 trajectory，而是蒸馏拒答或降低置信。

#### 情况 4：student 进入错误 prefix 后，teacher continuation 变得 plausible but still wrong

这适合证明 on-policy wrong-context 中 teacher token-level signal 可能局部合理但全局不可靠。

### 推荐实验 1：Unanswerable Prompt Analysis

构造或选取一批不可回答问题：

```text
future event questions
false premise questions
missing evidence questions
unknown entity questions
ambiguous questions
```

比较：

```text
Base model
Vanilla OPD
Correct-only OPD
Verifier-filtered OPD
Calib-OPD
```

指标：

```text
Answer Rate
Refusal Rate
Hallucination Rate
High-confidence Wrong Rate
ECE / Brier
Average Confidence on Wrong Answers
```

如果 vanilla OPD 后：

- answer rate 上升；
- refusal rate 下降；
- high-confidence wrong rate 上升；
- wrong answer confidence 上升；

就可以支持“普通 OPD 可能强化过度自信错误回答”。

### 推荐实验 2：Teacher-Student Knowledge Boundary Bucketing

先让 teacher 和 student 都生成答案和 confidence，然后分桶：

```text
T_correct / S_correct
T_correct / S_wrong
T_wrong / S_correct
T_wrong / S_wrong
T_uncertain / S_overconfident
```

重点观察：

```text
T_wrong / S_wrong
T_uncertain / S_overconfident
```

训练前后比较：

```text
P(wrong answer)
P(refusal)
student entropy
student confidence
teacher-student top-k overlap
wrong entity/date/number probability
```

### 推荐实验 3：Wrong-Prefix Continuation

流程：

1. student 生成错误 reasoning prefix；
2. 从不同 prefix depth 截断；
3. 让 teacher continuation；
4. 评估 continuation 是否能恢复正确答案。

这与 2604.13016v2 的 long-horizon reward degradation 分析类似。那篇论文发现 teacher continuation 的 advantage 会随着 student prefix 深度增加而下降。

如果在 hallucination 数据上也观察到：

```text
prefix 越错误/越深，teacher continuation 越难恢复正确；
但 vanilla OPD 仍在这些 prefix 上做 token-level matching；
```

就能说明普通 OPD 的 token-level signal 在错误轨迹上可能不可靠。

### 推荐实验 4：Vanilla OPD 是否增加 overconfidence

训练前后比较校准曲线：

```text
confidence bin -> empirical accuracy
```

重点看：

```text
confidence > 0.8 的错误样本比例
```

如果 vanilla OPD 让模型输出更集中、更低 entropy，但错误样本 confidence 也升高，就能支撑 H1。

### 推荐实验结论应该如何写

不要预设结论，而是写：

> We empirically test whether indiscriminate OPD amplifies overconfident errors under uncertain or unanswerable prompts.

也就是：

> 我们实证检验普通 OPD 是否会在不确定或不可回答问题上放大过度自信错误。

---

## 问题 4：H1 从何得来？有没有 related work 可以证明？

### H1 不能写成已被完全证明的事实

原 proposal 中的 H1：

```text
Blind imitation amplifies hallucination.
```

现在应该改成：

```text
Blind imitation may amplify hallucination under uncertain, unverifiable, or unsupported trajectories.
```

它应当是由相关工作启发的研究假设，而不是直接声明为已知事实。

### 可支撑 H1 的 related work

#### 1. Why Language Models Hallucinate / Kalai et al.

相关观点：

- hallucination 与训练和评测目标有关；
- accuracy-based grading 会鼓励模型猜；
- 如果不知道时没有部分奖励或拒答机制，模型倾向于给出答案。

对 H1 的支持：

```text
如果 OPD 的目标也是“尽量输出 teacher-like answer”，但没有 abstention/correctness calibration，
它可能继承类似 guessing incentive。
```

注意：这不是直接证明 OPD 会放大 hallucination，而是提供理论动机。

#### 2. Behaviorally Calibrated RL / 2512.19920v3

相关观点：

- 标准 RLVR binary reward 让模型像 good test-taker，而不是 honest communicator；
- 模型应该根据风险阈值决定回答或拒答；
- calibration 可以作为独立于 accuracy 的能力训练出来。

对 H1 的支持：

```text
如果 binary reward 都会鼓励 guessing，那么纯 imitation objective 更需要显式 calibration。
```

#### 3. TruthRL / 2509.25760v1

相关观点：

- binary reward 把 wrong answer 和 abstention 混在一起；
- ternary reward 区分 correct / uncertain / incorrect；
- ternary reward 明显降低 hallucination。

对 H1 的支持：

```text
普通 OPD 没有内置 ternary distinction，因此在 truthfulness 任务上可能不如 calibrated objective。
```

#### 4. R-Tuning

相关观点：

- 需要显式训练模型说 “I don't know”；
- 否则模型不一定自然学会拒答；
- 但过度拒答会损害 coverage。

对 H1 的支持：

```text
拒答是需要训练的行为；普通 OPD 如果只模仿答案分布，可能不足以学习知识边界。
```

#### 5. 2604.13016v2：Rethinking OPD

相关发现：

- OPD 成功依赖 thinking-pattern consistency 和 new knowledge；
- benchmark 分数更高的 teacher 不一定带来更好 OPD；
- OPD 主要通过 student-teacher high-probability overlap tokens 起作用；
- teacher reward quality 会随着 student trajectory depth 增加而退化；
- globally informative reward 不保证 locally exploitable signal。

对 H1 的支持：

```text
OPD 的 teacher signal 有适用条件。若在 hallucinated / unsupported / long-horizon wrong trajectory 上盲目使用，信号可能不可靠。
```

这篇最适合支撑“普通 OPD 有条件限制”，但不直接证明 hallucination amplification。

#### 6. 2604.03128v1：Self-Distilled RLVR

相关发现：

- OPSD 中 teacher conditioned on privileged information，而 student 看不到；
- distribution matching 在 information asymmetry 下会导致 privileged information leakage；
- RLSD 不再让 teacher 决定 update direction，而只让 teacher 调节 token-level magnitude；
- environment reward 决定更新方向。

对 H1 的支持：

```text
当 teacher 和 student 信息不对称时，直接 distribution matching 会产生结构性问题。
Calib-OPD 可以借鉴 RLSD：direction 应由可靠 verifier/correctness/evidence 决定，teacher 只提供局部 token-level guidance。
```

### 推荐 H1 新表述

```text
H1: In uncertain or unverifiable factual generation, indiscriminate OPD may reinforce overconfident incorrect behavior because its imitation objective is not explicitly anchored to correctness, evidence support, or abstention calibration.
```

中文：

> 在不确定或不可验证的事实生成中，未加筛选的 OPD 可能强化过度自信错误行为，因为它的模仿目标没有显式锚定正确性、证据支持或拒答校准。

---

## 问题 5：Teacher 评估会不会算力太大？如果不用 teacher 用 verifier，那和 RLVR 有什么区别？

### Teacher 评估确实可能成本很高

如果 Calib-OPD 每一步都需要：

```text
student rollout
teacher full-vocab logits
teacher confidence
teacher claim-level annotation
evidence verification
LLM-as-judge
```

那么算力和工程成本会非常高，甚至比普通 OPD 和 RLVR 都复杂。

所以方法必须明确区分：

```text
teacher logits / distribution: 用于 token-level distillation
verifier / judge: 用于 gating 或 reward
```

### 如果完全不用 teacher，只用 verifier，那就是 RLVR

RLVR 的形式是：

```text
student rollout y ~ pi_s(. | x)
verifier gives reward R(x, y)
policy gradient update
```

如果 Calib-OPD 只做：

```text
correct -> reinforce
wrong -> penalize
unknown -> abstain
```

而没有 teacher distribution 或 teacher response supervision，那么它就是 TruthRL / calibrated RLVR，不是 OPD。

### Calib-OPD 和 RLVR 的区别

核心区别应该写成：

```text
RLVR:
  verifier provides scalar outcome reward;
  training signal is sparse;
  update is policy-gradient based.

Calib-OPD:
  verifier provides gate / label / correctness anchor;
  teacher provides token-level distribution or response-level supervision;
  training signal is dense but selectively applied.
```

也就是：

```text
Verifier decides whether to learn.
Teacher tells the student how to learn.
```

中文：

> verifier 判断该不该学，teacher 提供怎么学的 dense signal。

### 降低算力成本的策略

#### 1. 只对 gated-positive 样本计算 teacher logits

先用便宜 verifier 判断：

```text
correct / supported / answerable
```

只有通过 gate 的样本才调用 teacher logits。

```text
if gate == positive:
    query teacher logits
else:
    use abstention or skip
```

#### 2. 使用 top-k logits

不需要 full vocabulary teacher distribution，可以只取：

```text
teacher top-k
student top-k
overlap top-k
```

2604.13016v2 显示 top-k / sampled-token OPD 已经能保留很多效果。

#### 3. 离线 teacher annotation

对于固定训练集，可以提前离线生成：

```text
teacher responses
teacher confidence
teacher answerability label
evidence support label
```

训练时只在线计算必要的 token-level logits。

#### 4. 分层训练

第一阶段：

```text
response-level Calib-OPD
```

第二阶段：

```text
claim-level Calib-OPD on smaller subset
```

不要一开始就全量 claim-level。

#### 5. 用 verifier 处理 direction，用 teacher 处理 magnitude

借鉴 2604.03128v1 的 RLSD：

```text
direction: verifier reward / correctness / evidence
magnitude: teacher token-level log-ratio or KL
```

这样可以减少 teacher 对训练方向的控制，降低 teacher 错误带来的风险。

### 推荐方法边界

Calib-OPD 必须保留 OPD 特征：

```text
student on-policy rollout
teacher token-level or response-level dense supervision
calibration/verifier gate
```

否则会退化成普通 RLVR。

---

## 问题 6：损失函数是不是太复杂？其他 RL 后训练论文也会设置这么复杂吗？

### 当前 proposal 中的 loss 确实偏复杂

原始总损失写成：

```text
L = alpha * g_pos * L_opd
  + beta  * g_abs * L_abs
  + gamma * g_neg * L_ul
  + delta * L_conf
  + eta   * L_rl
```

这个设计覆盖面很广，但作为主方法可能太复杂。reviewer 会问：

- 到底哪个组件起作用？
- 是否只是堆 loss？
- hyperparameter 是否难调？
- 是否比 TruthRL / RLSD / GRPO 工程复杂太多？

### 其他 RL 后训练论文确实有复合目标，但通常主线很清楚

例如：

#### PPO / RLHF

包含：

```text
policy loss
value loss
KL penalty
entropy bonus
```

但核心主线是 policy optimization。

#### GRPO

去掉 critic，用 group-relative advantage，形式更简单。

#### TruthRL

核心主线是 ternary reward：

```text
correct: +1
uncertain: 0
wrong: -1
```

虽然实验有 reward variants，但主方法很简单。

#### Behaviorally Calibrated RL

核心是 proper scoring rule / confidence calibration。虽然有 Explicit Thresholding、Brier、CE、Critic Value 多个策略，但每个策略的目标相对独立。

#### RLSD

核心很简洁：

```text
environment reward determines update direction;
privileged teacher ratio modulates token-level magnitude.
```

虽然公式涉及 clipping、lambda schedule、teacher logprob ratio，但主线不是多个 loss 堆叠，而是修改 advantage。

### Calib-OPD 应该简化主方法

推荐把主方法改成 Calib-OPD-lite：

```text
if correct/support and teacher_conf > tau:
    L = L_OPD
elif should_abstain:
    L = L_abs
else:
    skip
```

也就是只保留两个核心组件：

```text
1. verifier-gated OPD
2. abstention distillation
```

这个版本已经足够表达创新点：

> selective imitation + abstention transfer

### 复杂组件作为扩展或消融

把其他 loss 放到 ablation：

```text
+ confidence calibration loss
+ unlikelihood loss
+ claim-level gate
+ evidence support gate
+ RLVR joint loss
```

主论文可以这样安排：

#### Main Method

```text
Calib-OPD:
  gated OPD on supported/correct trajectories
  abstention distillation on unanswerable/unsupported trajectories
```

#### Extensions

```text
Calib-OPD + Conf
Calib-OPD + UL
Calib-OPD + Claim
Calib-OPD + RL
```

#### Ablation

```text
which component matters?
```

### 推荐最终 loss

第一版主方法：

```text
L = g_pos * L_OPD + lambda_abs * g_abs * L_abs
```

其中：

```text
g_pos = I[correct or supported] * I[teacher_conf > tau]
g_abs = I[should_abstain or unsupported]
```

如果要加入 confidence，也建议作为第二版：

```text
L = g_pos * L_OPD + lambda_abs * g_abs * L_abs + lambda_conf * L_conf
```

unlikelihood 不建议作为主方法第一版，因为它容易带来训练不稳定和过度拒答。

---

## 对原 proposal 的建议修改

### 1. 修改核心 claim

原表述：

```text
普通 OPD 可能放大 hallucination。
```

建议改为：

```text
普通 OPD 在可验证任务中有效，但在不确定、不可回答和弱验证事实生成中，未校准的 imitation objective 缺少 deciding when not to imitate 的机制。
```

### 2. 修改 H1

原表述：

```text
Blind imitation amplifies hallucination.
```

建议改为：

```text
H1: Indiscriminate OPD may amplify overconfident errors on uncertain or unsupported prompts.
```

### 3. 修改方法主线

原方法过于复杂。建议主方法：

```text
Calib-OPD-lite:
  supported/correct -> OPD
  unanswerable/unsupported -> abstention distillation
  hallucinated/uncertain -> skip
```

扩展方法：

```text
Calib-OPD-conf:
  add confidence calibration

Calib-OPD-ul:
  add unlikelihood for hallucinated claims

Calib-OPD-claim:
  claim-level gating
```

### 4. 修改实验主线

实验应该先证明问题存在，再证明方法有效：

```text
Part 1: Does vanilla OPD amplify overconfident errors?
Part 2: Does gating fix it?
Part 3: Does abstention distillation improve unknown/unanswerable prompts?
Part 4: Does Calib-OPD generalize beyond RLVR to factual QA/RAG?
```

### 5. 最清晰的论文定位

最终论文定位建议：

> We do not claim OPD fails in verifiable RLVR. Instead, we study OPD under uncertainty and weak verifiability, where correctness, evidence support, and answerability are not trivially available. We show that calibrated gating is necessary to transfer not only knowledge but also knowledge boundaries.

中文：

> 我们不是说 OPD 在 RLVR 中失败，而是研究 OPD 在不确定和弱验证事实生成中的适用边界。Calib-OPD 的目标是蒸馏知识，同时蒸馏知识边界。

---

## 最终建议

现在这个 idea 最稳妥的版本不是“替代 OPD”，而是：

```text
Calib-OPD = OPD for weakly-verifiable hallucination settings
```

它的核心问题应该写成：

```text
When should a student imitate a teacher under uncertainty?
```

而不是：

```text
Why is OPD bad?
```

这样的定位更容易成立，也更容易和 2604.13016v2、2604.03128v1、TruthRL、Behaviorally Calibrated RL 形成清楚的 related work 关系。
