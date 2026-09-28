# 微调平台 QA 数据侧 — QA 数据生成、蒸馏及知识过滤技术设计稿

## 文档概述

### 背景

大语言模型（LLM）的微调效果高度依赖训练数据的质量、多样性与知识覆盖度。在企业级微调平台中，QA（Question\-Answer）对数据是 SFT（Supervised Fine\-Tuning）与 DPO（Direct Preference Optimization）阶段的核心资产。然而，纯人工标注成本高昂、周期长、规模有限；完全依赖公开数据集又面临领域不匹配、知识过时、格式不一致等问题。

因此，构建一条**自动化、可审计、可迭代**的 QA 数据生成与过滤管线，成为微调平台数据侧的关键基础设施。该管线需要解决三个核心问题：

1. **Question 从哪里来**：如何从企业内部文档（技术文档、产品手册、知识库等）自动生成高质量、可回答的问题？

2. **Answer 谁来写**：如何利用强教师模型蒸馏出事实准确、推理完整的答案？

3. **质量如何保证**：如何过滤幻觉、去重、确保知识增益，避免"垃圾进垃圾出"？

### 目标

本设计稿的目标是：

- **构建端到端 QA 数据生成管线**：从原始文本块输入，到结构化 QA 对入库，全流程自动化。

- **保证数据质量**：通过多级过滤与质检机制，确保入库数据的事实一致性、可回答性与多样性。

- **控制成本与效率**：在数据质量与生成成本之间取得平衡，支持日级百万级 QA 对的吞吐。

- **可扩展与可演进**：模块化设计，支持后续接入新教师模型、新过滤策略、新评估维度。

### 范围

|维度|包含|不包含|
|---|---|---|
|数据类型|单轮 QA 对、多轮对话片段、CoT 推理数据|纯指令跟随数据（无知识依赖）、代码数据|
|数据来源|企业内部文档（Chunk 化后）、公开 QA 数据集清洗|实时用户日志、人工标注平台|
|下游用途|SFT 训练集、DPO 偏好对构建|RLHF 奖励模型训练、预训练语料|
|模型范围|教师模型 API 调用、开源模型自托管|学生模型训练本身、训练框架选型|

### 术语表

|术语|英文|定义|
|---|---|---|
|Chunk|Text Chunk|经过切分的文本块，作为 QA 生成的最小知识单元|
|锚点|Anchor|从 Chunk 中抽取的关键实体、术语或句子，用于驱动 Question 生成|
|教师模型|Teacher Model|用于蒸馏答案的强模型，如 GPT、Claude 等|
|响应蒸馏|Response Distillation|仅使用教师模型的输出文本进行 SFT，不访问 logits|
|CoT|Chain\-of\-Thought|思维链，模型在给出答案前展示推理过程|
|NLI|Natural Language Inference|自然语言推理，判断前提是否蕴含假设|
|知识增益|Knowledge Gain|问题是否必须依赖给定上下文才能正确回答|
|Evol\-Instruct|Evolutionary Instruction|通过深度/广度进化将简单指令复杂化的技术|
|Round\-trip|Round\-trip Validation|生成问题后，用教师模型基于原文回答，验证可回答性|
|LLM\-as\-Judge|LLM\-as\-Judge|用强 LLM 作为评分者对数据质量进行多维打分|

---

## 调研综述

本部分基于三方向调研（Part1: Question 生成、Part2: 教师蒸馏、Part3: 知识过滤），对主流技术路线进行横向对比分析。

---

### Question生成技术路线对比

本节对比 Question 生成的经典技术路线（核心做法、优劣势与代表工作），并将 2025–2026 年的最新演进融入对应路线。

#### Self\-Instruct 范式

**核心做法**：少量人工种子指令 → LLM 自我生成新指令 → 启发式过滤 → 迭代扩充。优点是冷启动成本极低、可完全复现、无需标注；缺点是指令偏任务型、同质化严重、质量天花板受教师限制。

**代表工作**：

- [Self\-Instruct](https://arxiv.org/abs/2212.10560)（ACL 2023）：从 175 条人工种子指令出发，迭代生成 52K 指令，用 ROUGE\-L 去重和格式过滤控制质量。

- [Stanford Alpaca](https://github.com/tatsu-lab/stanford_alpaca)（[博客](https://crfm.stanford.edu/2023/03/13/alpaca.html)）：基于 Self\-Instruct 流程，用 text\-davinci\-003 生成 52K 指令跟随数据微调 LLaMA\-7B，总成本不到 600 美元。

**针对"完全依赖模型内部知识、领域知识过时或不全"短板的演进方向**：

*检索增强 / 文档锚定*——让合成数据锚定真实人类文档，而非模型内部参数记忆：

- [SearchInstruct](https://arxiv.org/abs/2509.10708)：通过 query 扩展 → 文档检索 → 答案生成的迭代 pipeline，针对欠覆盖的子领域和题型定向补数据。

- [CRAFT](https://aclanthology.org/2025.tacl-1.76/)（TACL 2025）：用 8–32 条人工示例 \+ 大规模语料相似度检索 \+ LLM 增强改写，让合成数据锚定真实人类文档，可扩展到 25K 条。

- RAG\-grounded QA 流水线：从领域知识库/知识图谱检索片段生成有依据的 QA 对，再用 [RAGAS](https://arxiv.org/abs/2309.15217)（[GitHub](https://github.com/explodinggradients/ragas)）类评分（faithfulness、answer relevancy 等）过滤。

*技能元认知 / 原则驱动*——不直接让大模型出题，而是先提炼"怎么出题"的元规则：

- [instruct\-skillmix](https://arxiv.org/abs/2408.14774)：先用 LLM 从种子中提取核心"技能"（元认知），再随机组合技能对生成 \(instruction, response\) 数据。

- [PSI（Principle\-based Self\-Instruction）](https://arxiv.org/abs/2507.05991)：由大模型从少量任务数据总结多级任务原则，小模型依据这些原则批量生成数据，不依赖大模型直接出题，兼顾数据隐私和成本。

---

#### Evol\-Instruct 进化路线

**核心做法**：对已有指令做深度进化（加约束/深化/推理）和广度进化（长尾新任务），由 Eliminator 淘汰无法回答的样本。优点是显著提升指令复杂度与多样性、覆盖长尾；缺点是进化可能引入 chunk 外信息、破坏可回答性、过度复杂化。

**代表工作**：

- [WizardLM](https://arxiv.org/abs/2304.12244)：提出 In\-depth Evol（加约束、加深、具体化、推理）和 In\-breadth Evol（新任务方向），配合 Eliminator 过滤不可回答样本。

- [WizardCoder](https://arxiv.org/abs/2306.08568)（ICLR 2024）：将 Evol\-Instruct 适配到代码域，用 Code Evol\-Instruct 对 StarCoder 做 SFT，HumanEval pass@1 达 57\.3%。

- [WizardMath](https://arxiv.org/abs/2308.09583)：在 Evol\-Instruct 基础上引入 Reinforced Evol\-Instruct（RLEIF），70B 模型 GSM8K pass@1 达 81\.6%。

**在 Evol\-Instruct 基础上改进效率与多样性的新变体**：

- [Tag\-Evol](https://arxiv.org/abs/2505.24165)（ACL 2025）：用多样化的知识标签注入原始指令实现"受控进化"，无需多轮迭代即可生成不同难度的高质量进化数据，比传统迭代进化更高效、更多样。

- [Infinite\-Instruct](https://arxiv.org/abs/2505.23177)：面向代码域做双向合成（代码片段 → prompt 与 prompt → 代码）\+ 静态验证，突破 Evol\-Instruct 只从已有指令出发的局限。

- [CoT\-Self\-Instruct](https://arxiv.org/abs/2507.23751)：分两阶段，先以 CoT 方式创建合成指令、再做合成指令筛选，在数学任务上超过现有训练集，开放任务上配合 RIP 改善 DPO。

---

#### Answer\-aware QG（学术路线）

**核心做法**：给定 context \+ 答案 span → 反向生成问题（SQuAD 范式）。优点是可 round\-trip 验证、答案可控、工程成熟；缺点是依赖预标注答案 span、问题类型受限。

**代表工作**：

- [SQuAD](https://arxiv.org/abs/1606.05250)（EMNLP 2016）：10 万\+ 维基百科阅读理解问答对，答案为原文片段，奠定 extractive QA 与反向 QG 的基础范式。

- [MixQG](https://arxiv.org/abs/2110.08175)（TACL 2022）：用多数据集混合答案类型训练神经 QG 模型，在 SQuAD 和 NQ 上超越同规模直接微调模型。

- [FLAN](https://arxiv.org/abs/2109.01652)（ICLR 2022）：在 60\+ NLP 数据集上以自然语言指令模板进行多任务指令微调，验证了指令描述 → 问题 → 答案这一范式对零样本泛化的价值。

---

#### 锚点驱动 LLM 反向提问

**核心做法**：NER \+ TF\-IDF/RAKE 抽取锚点实体 → LLM 围绕锚点生成问题 → JSON 结构化输出。优点是答案天然来自 chunk、可定位 evidence span、领域适配好；缺点是锚点抽取质量影响大、需设计好 few\-shot prompt。

**代表工作**：

- 工业界管线：广泛用于 RAG 系统和知识库问答的批量 QA 生成，以实体/术语为锚点保证答案可溯源。

- [Self\-Alignment（Humpback）](https://arxiv.org/abs/2308.06259)：从人类网页文本出发做 instruction backtranslation——先让模型为人类写的文档反向生成指令（self\-augmentation），再用 reward model 筛选高质量样本（self\-curation），本质是"答案文档 → 问题"的反向提问范式。

---

#### 多跳 QG 与知识图谱/知识树驱动

**核心做法**：跨多句/多实体生成需要组合推理的问题。优点是提升推理难度、贴近真实复杂需求；缺点是生成难度高、易幻觉、验证成本大。

**经典代表工作**：

- [HotpotQA](https://arxiv.org/abs/1809.09600)（EMNLP 2018）：11\.3 万维基百科多跳问答对，提供句子级 supporting facts，涵盖 bridge（桥接实体）和 comparison（对比）两类多跳问题。

- [MixQG](https://arxiv.org/abs/2110.08175)：同时覆盖单跳与多跳答案类型的 QG 模型。

**用结构化知识保证覆盖度和多跳推理质量的新方向**：

- [Condor](https://arxiv.org/abs/2501.12273)：两阶段框架——World Knowledge Tree（世界知识树，8400\+ 标签层次化规划知识点）→ 生成 → Self\-Reflection Refinement 自我精炼，仅 20K 样本即可超过大量基线，支持到 72B 模型迭代自改进。

- [GraphGen](https://arxiv.org/abs/2505.20416)（[GitHub](https://github.com/open-sciencelab/GraphGen)）：从源语料构建细粒度知识图谱，分别生成原子 QA（基础知识）、聚合 QA（综合知识）、多跳 QA（k\-hop 推理）三类，并用 expected calibration error 识别模型知识盲区、优先生成长尾知识的 QA。

- [LinkQA / LinkSyn](https://arxiv.org/abs/2508.01317)（ACL 2026）：从种子 QA 抽取知识点（KP）构建 KP 图，通过图游走生成跨知识点关联的多样 QA，可灵活控制学科与难度分布，平衡 KP 覆盖率与流行度。

---

#### 难度感知 / 课程式生成

**核心思想**：生成模型"恰好不会"的题（self\-consistency ≈ 0\.5 时学习信号最大），而非盲目生成大量题目。

**代表工作**：

- [QueST](https://arxiv.org/abs/2510.17715)：难度感知图采样 \+ 难度感知拒绝微调，直接优化"出题器"去生成难题，训练后的出题器在生成难题上超过 GPT\-4o。

- [LLM\-adaptive 难度分级](https://arxiv.org/abs/2504.11919)：先用基础模型在评测集上的表现估计每题难度，按难度分布采样关键题，再用 DeepSeek\-R1 生成 CoT 数据。

- [MathMixup](https://arxiv.org/abs/2601.17006)：通过混合（hybrid）和分解（decomposed）策略生成难度可控的数学题，构建难度梯度并配合课程学习训练。

- [TTCS](https://arxiv.org/abs/2601.22628)：由 Synthesizer 和 Solver 共同迭代，生成器根据 solver 的不确定性（约 0\.5）奖励出题，形成自演进课程，1\.5B 模型在数学基准上提升 24\+ 分。

- 大规模分等级合成：用多等级合成器（覆盖教育各学段）\+ 高难度合成器（提升 H4/H5 样本比例）\+ 答案精炼模块，构建系统的难度梯度。

---

#### 主动学习 / 选择性生成

**核心做法**：不再盲目大规模生成，而是挑选最有价值的点生成——从当前未标注池中选信息量最大的点，加入教师模型的生成 prompt，迭代式主动合成。

**代表工作**：

- [Towards Active Synthetic Data Generation](https://arxiv.org/abs/2512.00884)：迭代闭环框架，学生模型在种子数据上评估 → 按评分指标选点 → 教师模型针对薄弱点生成新样本 → 训练 → 再评估，避免生成冗余或不相关数据。

---

#### 选型建议

- **领域适配 / 有私有文档时**：优先走检索锚定（[CRAFT](https://aclanthology.org/2025.tacl-1.76/) / [SearchInstruct](https://arxiv.org/abs/2509.10708)）或知识图谱驱动（[GraphGen](https://arxiv.org/abs/2505.20416) / [Condor](https://arxiv.org/abs/2501.12273)），避免幻觉。

- **通用指令跟随场景**：[Tag\-Evol](https://arxiv.org/abs/2505.24165) / [CoT\-Self\-Instruct](https://arxiv.org/abs/2507.23751) 比纯 Evol\-Instruct 更高效。

- **算力有限、想省标注时**：[PSI](https://arxiv.org/abs/2507.05991) 原则驱动或[主动合成](https://arxiv.org/abs/2512.00884)可以用小模型批量生成、大模型只做原则提炼。

#### 对比分析结论

1. **Self\-Instruct** 适合冷启动但同质化问题在领域知识 QA 中尤其突出——同一 chunk 反复生成"请解释 XX"类问题。

2. **Evol\-Instruct** 是复杂度提升的有效手段，但必须与锚点约束结合，否则进化出的问题会超出 chunk 范围，失去知识 grounding。

3. **Answer\-aware QG** 的 round\-trip 验证思想值得借鉴，但预标注答案 span 的成本在大规模管线中不可接受。

4. **锚点驱动 \+ LLM 反向提问**是工程上最均衡的选择：锚点保证可定位性，LLM 保证多样性，JSON 结构化输出保证工程可处理。

> Self\-Instruct 范式与 Part2 的 Alpaca 式响应蒸馏同属"种子→指令→响应→过滤"管线（Answer 侧见 2\.2），本 Part 仅从 Question 侧描述；round\-trip 可回答性验证与 Evol Eliminator 均复用 Part2 的教师回答操作，作为 Q\&A 联合方法统一说明于 5\.3\.3。
> 
> 

所有链接已检索完毕。以下是重新整理后的 Part2 内容——去掉"近期演进"标签，将每条技术路线的经典工作与新进展自然融合，所有论文/项目均附上原文链接。

---

### 教师蒸馏技术路线对比

本节对比 Answer 侧（教师蒸馏）的经典技术路线（核心做法、优劣势与代表工作），并将 2025–2026 年的最新演进融入对应路线。

#### Alpaca 式响应蒸馏（单轮）

**核心做法**：与 Part1 的 Self\-Instruct 范式同一条"种子→指令→响应→过滤"管线（指令生成侧见 2\.1）：种子 → 指令生成 → 教师生成响应 → 过滤去重 → SFT。本条目聚焦 Answer 侧的教师响应蒸馏。优点是极简可复现、API 成本低（约 $500/52K 条）；缺点是教师天花板、单轮为主、多样性依赖 seed。

**代表工作**：

- [Stanford Alpaca](https://github.com/tatsu-lab/stanford_alpaca)（[博客](https://crfm.stanford.edu/2023/03/13/alpaca.html)）：175 条种子指令 \+ text\-davinci\-003 生成 52K 单轮指令跟随数据，微调 LLaMA\-7B。

- GPT\-3\.5 蒸馏：工业界普遍采用的"闭源大模型 API → 批量生成 → SFT"范式，是当前单轮指令数据的默认来源。

---

#### 多轮对话与工具调用数据合成

**核心做法**：单轮 QA 之外，展开多轮对话和工具调用轨迹。经典路线是 Meta\-topic 设计 → 首轮 Query → 教师交替扮演用户/助手迭代对话；优点是多轮覆盖广（平均 3\.8 轮）、对话场景丰富；缺点是自说自话倾向、可能产生怪癖对话、单轮质量不高。

**经典代表工作**：

- [UltraChat](https://arxiv.org/abs/2305.14233)：用 30 个 meta\-topic \+ 20 种材料种子，让两个 GPT API 交替扮演用户和助手，迭代生成 150 万条多轮对话，平均 3\.85 轮。

- UltraLM：基于 UltraChat 数据训练的聊天模型，在通用对话质量上超越此前开源模型。

**多轮对话连贯性改进**：

- [ConsistentChat](https://arxiv.org/abs/2506.03558)（EMNLP 2025）：用骨架引导——先建模 9 种意图轨迹的全局结构（Intent Modeling），再生成对齐的用户查询序列（Skeleton Generation），保证多轮连贯性；构建约 1\.5 万段多轮对话，聊天一致性提升 20–30%。

- [Review\-Instruct](https://arxiv.org/abs/2505.11010)（Findings of ACL 2025）：采用"提问\-回答\-评审"多智能体循环——Candidate 生成指令、多个 Reviewer 评审、Chairman 迭代精炼，增强对话多样性和难度。

- [DialogueForge](https://arxiv.org/abs/2507.15752)：从真实人类\-chatbot 首轮对话出发，推断任务目标后交替生成 Inquirer（人类）/ Responder（机器人）轮次，用真实首轮对话锚定对话风格。

**工具调用 / Agent 轨迹合成**：

- [APIGen\-MT](https://arxiv.org/abs/2504.03601)（[项目页](https://apigen-mt.github.io/)，NeurIPS 2025）：分两阶段，先产出带 ground\-truth 动作的任务蓝图（LLM 委员会评审），再模拟人\-智能体交互展开成完整轨迹；训练了 xLAM\-2\-fc\-r 系列（1B–70B）。

- [ToolACE\-MT](https://arxiv.org/abs/2508.12685)：用非自回归迭代生成（粗骨架初始化 → mask\-and\-fill 精炼 → 离线验证）构建多轮工具对话轨迹。

- [Magnet](https://arxiv.org/abs/2503.07826)（ACL 2025）：基于函数签名的图翻译——将函数组织为依赖图，采样签名路径（Function Signature Path），再迭代翻译为多轮用户查询和可执行函数调用，配合 context distillation 生成正负样本。

---

#### CoT 蒸馏与可验证奖励

**核心做法**：教师生成带思维链的推理轨迹，学生学习 cross\-entropy。优点是推理能力迁移效果显著、R1 范式验证成功；缺点是 CoT 长度不可控、成本高、易引入错误推理链。

**经典代表工作**：

- [DeepSeek\-R1](https://arxiv.org/abs/2501.12948)：通过 RL（GRPO）激发推理能力，同时验证了 R1 推理轨迹蒸馏到小模型（Qwen2\.5\-32B 等）的有效性，是当前 CoT 蒸馏路线的标杆。

- OpenAI o1 蒸馏路线：闭源推理模型通过 API 输出推理轨迹供学生模型 SFT，工业界广泛采用。

**可验证奖励 \+ 拒绝采样微调（RFT/STaR 系）——2025 年推理模型数据合成的主流范式**：让教师模型对每题采样多条 CoT，用确定性验证器（数值答案、单元测试、代码可执行性）过滤，只保留正确轨迹做 SFT。

- [STaR](https://arxiv.org/abs/2203.14465)（NeurIPS 2022）：Self\-Taught Reasoner 的原始框架——生成 rationale → 验证答案正确性 → 正确轨迹加入 SFT → 错误轨迹给答案反向补 rationale（rationalization）→ 迭代。

- [RAFT](https://arxiv.org/abs/2504.11343)：RFT 的简化变体，只用正样本（正确答案的推理轨迹），实验显示早期收敛比混合正负样本的 GRPO 更快；进一步引入 importance sampling 和 clipping 得到 RAFT\+\+。

- [RIFT](https://arxiv.org/abs/2601.09253)：改进 RFT 丢弃负样本的问题，把低于阈值的"失败样本"通过奖励信息复用（reweighting loss），让模型学会区分对错而非简单过滤。

- [Enigmata](https://arxiv.org/abs/2505.19914)（[项目页](https://enigmata.ai/)，NeurIPS 2025）：合成 36 类可验证逻辑谜题（每个有 generator \+ rule\-based verifier），配合 VC\-PPO 做全自动 RL pipeline 缩放推理能力。

---

#### 多教师集成

**核心做法**：2–3 个教师分别生成答案 → LLM\-as\-judge 选最优。优点是质量上限提升、降低单教师偏见；缺点是成本翻倍、judge 本身有偏差。

**代表工作**：工业界管线实践——在边界问题、难题上同时调用多个教师模型（如 GPT\-4o \+ Claude \+ DeepSeek\-R1），用 LLM\-as\-judge 或规则投票选择最优回答；简单问题仍走单教师以控制成本。

---

#### Constitutional AI（CAI）

**核心做法**：SL\-CAI 自我批评改写 \+ RL\-CAI AI 偏好判断 → PPO。优点是对齐良好、可迭代改进；缺点是需要 RL 训练、不直接适用于 QA 数据生成。

**代表工作**：

- [Constitutional AI](https://arxiv.org/abs/2212.08073)（Anthropic, 2022）：让模型依据一组"宪法"原则自行批评和修订不安全回答（SL\-CAI），再用 AI 对修订前后回答做偏好排序训练 reward model，最后 PPO 优化（RL\-CAI）。

---

#### 选型建议

- **数学/代码等可验证域**：RFT \+ 难度感知出题（[QueST](https://arxiv.org/abs/2510.17715) / [TTCS](https://arxiv.org/abs/2601.22628)）是当前性价比最高的路线，验证器自动保证正确性。

- **Agent / 工具调用场景**：直接用 [APIGen\-MT](https://arxiv.org/abs/2504.03601) / [ToolACE\-MT](https://arxiv.org/abs/2508.12685) 这类多轮骨架 \+ 验证的 pipeline。

#### 对比分析结论

1. **响应蒸馏是默认选择**——闭源 API 场景下唯一可行方案，成本可控。

2. **CoT 蒸馏是增量增强**——在数学、推理类问题上叠加 CoT，通用 QA 保持简洁答案。

3. **多教师集成分层使用**——简单问题单教师即可，难题/边界问题才走多教师 \+ judge。

---

### 知识过滤技术路线对比

本节对比针对**合成 QA 训练对**（Question \+ Context \+ Answer）进入 SFT 训练集之前的过滤技术路线——核心问题是：海量合成 QA 对中，哪些应该保留、哪些应该丢弃？过滤维度包括：事实是否锚定原文、问题是否真正可回答、是否真正需要原文、教师是否给出一致答案、整体质量是否达标、是否存在冗余。

#### 事实蕴含过滤（NLI / Faithfulness）

**核心做法**：逐句判断 Answer 是否被 Context 蕴含（Context ⊨ Answer），不蕴含则丢弃。这是训练数据的第一道事实闸——确保答案不是模型编造的。

**代表工作**：

- [DeBERTa\-v3\-large\-mnli](https://huggingface.co/microsoft/deberta-v3-large-mnli)（[DeBERTa v3 论文](https://arxiv.org/abs/2301.10226)）：工业界最常用的事实蕴含判断模型，逐句比对 context 与 answer 的 entailment 概率，低于阈值即过滤。优点是速度快、可规模化；缺点是 NLI 模型本身有误差、长文本效果下降。

- [RAGAS](https://arxiv.org/abs/2309.15217)（[GitHub](https://github.com/explodinggradients/ragas)）：其 Faithfulness 指标将 answer 拆成 claim，逐条验证 claim 是否可从 context 推断，不可推断的 claim 比例高则丢弃该 QA 对；同时提供 Answer Relevancy 指标衡量问题与答案的匹配度，可作为过滤信号。

---

#### 可回答性验证（Round\-trip / Answerability）

**核心做法**：把 Context \+ Question 喂给模型，看模型能否复现合成时给定的 Answer；多次尝试仍不对则说明该 QA 对存在问题（问题歧义、答案不可推导、chunk 不完整），直接丢弃。

**代表工作**：

- [Source2Synth](https://arxiv.org/abs/2409.08239)：从真实数据源（表格、多跳文档）合成 QA 后，用模型对每条样本做 k=3 次推理验证——若模型无法复现标准答案则整条丢弃；在多跳 QA 上过滤掉了约 13% 的低质量样本。

- [ARES](https://arxiv.org/abs/2311.09476)（NAACL 2024）：生成合成 QA 后，用语义检索做 round\-trip 一致性验证——只有当问题能通过向量检索正确映射回源 chunk 时才保留，确保 QA 对与原文证据可双向追溯。

- [Aligning Extraction and Generation](https://arxiv.org/abs/2503.04789)：从两个维度过滤——**Answer Validity**（答案是否可从相关 chunk 推导）和 **Chunk Validity**（噪声 chunk 是否不包含答案，防止模型从干扰文档中"猜对"），双维度同时通过才保留。

---

#### 知识增益验证（无上下文消融）

**核心做法**：遮住 Context，只给 Question 让模型回答；如果模型不看原文也能答对，说明这道题对 RAG 微调毫无价值——它考的是模型内部已有知识，而非从原文检索信息，必须过滤。

**代表工作**：

- [QuALITY](https://arxiv.org/abs/2212.06084)（ACL 2023）：长文本阅读理解基准，问题设计上要求阅读大量原文才能回答——天然排除了"不看原文也能答对"的题，是知识增益过滤的参照标准。

- [InstructRetro](https://arxiv.org/abs/2310.07713)（ICLR 2024）：在构建检索增强指令微调数据时，系统对比了"有检索 vs 无检索"的性能差异——只有当有检索时性能显著提升的 QA 对才保留，这直接量化了知识增益。

---

#### 教师一致性检测（SelfCheckGPT）

**核心做法**：对同一问题 \+ Context 采样 N=5–10 个答案，计算答案间一致性；低一致性说明教师对该问题不确定或存在幻觉，该 QA 对的标签本身不可靠。

**代表工作**：

- [SelfCheckGPT](https://arxiv.org/abs/2303.08896)（EMNLP 2023）：采样一致性检测框架——模型"知道"的事实，不同采样应给出相似答案；幻觉事实则不同采样互相矛盾。在训练数据过滤场景中，低一致性的 QA 对说明答案标签不稳定，应丢弃或重新生成。优点是无需外部模型；缺点是 N 倍调用成本高、教师自信错误时（模型对错误知识高度一致）会漏检。

---

#### 多维打分（Reward Model / LLM\-as\-Judge）

**核心做法**：用专门打分模型或强 LLM 对 QA 对的相关性、正确性、完整性、语言质量多维评分，按分数排序选 Top\-K。质量是连续的，S/A/B 分层比 pass/fail 更有信息量。

**代表工作**：

- [AlpaGasus](https://arxiv.org/abs/2307.08701)（ICML 2024）：用 GPT\-3\.5 对 Alpaca 52K 数据逐条打分（helpfulness \+ accuracy），只保留 9K 高质量样本，训练效果反而超越完整 52K。

- [Deita](https://arxiv.org/abs/2312.15685)：从复杂度、质量、多样性三维度系统研究数据选择，提出 evol score \+ Repr Filter，用 1/10 训练数据达到 SOTA。

- [InsTag](https://arxiv.org/abs/2308.07074)（NeurIPS 2023）：用 LLM 为每条指令生成 6\.6K 知识标签，通过标签数量衡量复杂度、标签分布衡量多样性，据此筛选。

- NVIDIA Nemotron Reward Model：从 Helpfulness、Correctness、Coherence、Complexity、Verbosity 五个维度打分，设阈值过滤低质量合成数据。

---

#### 干扰文档增强（RAFT 路线）

**核心做法**：不完全是"过滤"而是"增强式构造"——黄金文档 \+ 4 篇干扰文档拼在一起做训练，其中 20% 的样本完全不含黄金文档（纯干扰），教会模型"无依据时拒答"而非胡编。这种构造方式本身就是一种数据筛选策略：只保留能被干扰文档包围后仍正确回答的 QA 对。

**代表工作**：

- [RAFT](https://arxiv.org/abs/2403.10131)（Meta, 2024）：Adapting Language Model to Domain Specific RAG——训练时将黄金文档与 4 篇同领域干扰文档拼接，20% 样本纯干扰，答案采用 CoT 格式并要求直接引用原文。实验证明这种构造方式显著提升模型在噪声检索下的拒答能力和事实遵循。

---

#### 三层去重

**核心做法**：n\-gram MinHash / LSH（精确去重）→ SemDeDup 语义去重（embedding 空间近邻删除）→ 聚类多样性采样。合成 QA 对极易产生大量语义重复的问题（同一 chunk 反复生成"请解释 XX"），去重是规模管线的必要步骤。

**代表工作**：

- [SemDeDup](https://arxiv.org/abs/2303.09540)（ICML 2023）：用预训练模型 embedding 识别语义重复数据对，在 LAION 上删除 50% 数据后性能几乎不降，OOD 性能反而提升。

- [Data Selection Survey](https://arxiv.org/abs/2402.05123)：系统综述指令微调数据选择方法，将去重、质量过滤、多样性采样统一到 quality / diversity / complexity 三维框架。

---

#### 对比分析结论

1. **NLI / Faithfulness 是第一道粗筛**——成本低、速度快、可规模化；但不能单独使用，需 LLM 复核或 round\-trip 验证。

2. **可回答性验证是第二道闸**——模型复现不了标准答案的 QA 对，要么问题有歧义、要么答案不可推导，直接丢弃（Source2Synth 过滤掉约 13%）。

3. **无上下文消融是知识增益的核心指标**——"不看原文也能答对"的问题对 RAG 微调毫无价值，必须过滤。

4. **多维打分替代纯二分类**——质量是连续的，S/A/B 分层比 pass/fail 更有信息量。

5. **RAFT 思路值得借鉴**——20% 的"纯干扰"样本教会模型拒答，这属于数据增强式筛选而非纯过滤。

6. **去重在规模管线中不可省略**——SemDeDup 可删 \~50% 语义重复而不掉点。

---

## 推荐技术路线

### **智训平台现状 QA 方案（As\-Is）**

本节描述智训微调平台当前已上线的数据构造能力（实现位于 `data_governance/backend`，需求对应 FR\-2\.x / FR\-4）。该方案是设计稿第 3 节「推荐技术路线」的对照基线；本仓库 `qa_generate` 以 Recipe `baseline_platform`（`configs/recipes/baseline_platform.yaml`）对其行为做可复现封装，用于消融实验 E1\_baseline。

#### **3\.4\.1 定位与能力边界**

|维度|现状|
|---|---|
|产品入口|处理任务（`Task`）\+ 能力目标 `goal`（默认 `qa` 知识问答，另支持 `description` / `extract`）|
|教师模型|单一 DeepSeek 兼容 API（配方字段 `teacher: DeepSeek 教师模型`，不可切换）|
|生成范式|一次调用同时产出 Q \+ A \+ evidence，无独立的 Question 生成 / Answer 蒸馏两阶段|
|自动化过滤|规则级证据子串校验 \+ 教师 `quality_check`（答案是否被证据支持）\+ 任务内 Q/A 哈希去重|
|质量分层|无 S/A/B 自动分层；样本带 `issue` 字段，人工审核（`pending` → `approved`）后方可发布|
|下游训练|发布为不可变 `DatasetVersion` 快照（JSONL：`messages` \+ `metadata`），LoRA SFT 由 `training_worker` 消费|

与设计稿「锚点反向提问 → 响应蒸馏 → 多级过滤 → S/A/B 分层」相比，现状方案更轻、更依赖人工终审，但在证据可定位、来源追溯、分区防泄漏方面已与平台治理模型深度绑定。

#### **3\.4\.2 端到端数据流**

```Plain Text
资料上传(File) → 解析(Block) → 任务内片段(TaskBlock) → 分批构造(GenerationBatch)
      → 候选样本(Sample, issue?) → 人工审核 → 发布数据集(DatasetVersion) → 微调(TrainRun)
```

阶段说明（`app/services/pipeline.py` 编排）：

1. 解析（FR\-2\.1）：PDF/Office/图片走 MinerU 转 Markdown；txt/md/csv/json 直读。产出工作空间级 `Block`（含 `loc` 页码/章节、`kind`：text/table/image\_caption）。

2. 任务内预处理（FR\-2\.2 \~ FR\-2\.3）：按任务配方 `recipe` 对来源文本再切分为 `TaskBlock`（与解析默认一致：**章节边界优先，章节内字符窗 ****`split=600`****、****`overlap=80`**）。可选开启：文本规范化（`governance.normalize`）、PII 脱敏（`privacy`）、片段级去重（`dedup`，基于内容哈希）。

3. 分批构造（FR\-2\.4）：按 `batch_chars`（默认 16000）将多个片段拼成一批，单次教师调用生成 `count` 条样本（默认 6），受 `max_calls` / `max_samples` 预算约束，失败批次可重试 `retries` 次。

4. 入库与状态：每条写入 `Sample`，继承来源资料的 `train`/`validation`/`test` 分区；`dedup_key` 为 Q/A 规范化哈希，任务内去重。

5. 发布（FR\-4）：仅 `status=approved` 且通过 `sample_errors`（含证据子串、来源族分区一致性、有效人工复核 digest）的样本进入快照；版本号 `v1.{n}` 自增。

#### **3\.4\.3 Question / Answer 生成（****`generation.generate_samples`****）**

对 `goal=qa`，系统 Prompt 要求：

- 问题贴近真实用户提问，答案完全由资料支持、简洁准确；

- 每条必须标注片段编号 `block` 与 证据原文 `evidence`（须为片段内连续原文、不得改写）；

- `kind` 从 \{事实问答, 步骤说明, 条件问答, 比较问答\} 选取。

用户侧可附加 `recipe.requirements`（默认：「仅根据资料生成样本，保留来源与页码；证据不足时不编造答案。」）。

与设计稿差异：无锚点抽取、无问题类型配额、无 Evol\-Instruct、无 round\-trip 可回答性预校验、无按 `q_type` 的 CoT/教师路由；答案与问题在同一次 JSON 结构中生成，不经过独立蒸馏模块。

#### **3\.4\.4 自动化质检门**

|顺序|门控|实现|未通过时|
|---|---|---|---|
|1|片段与证据定位|生成后立即校验：`block` 合法且 `normalized(evidence) ⊆ normalized(block.content)`|写入 `issue`，如「完整证据未能在来源片段中定位」|
|2|教师交叉核对|`generation.quality_check`：判断「回答」是否被「证据」支持（JSON `supported`）|`issue` 记录理由或调用失败提示|
|3|任务内去重|Q/A 内容哈希 `dedup_key`|重复候选丢弃，不入库|
|4|发布前硬校验|`quality.sample_errors`：非空 Q/A、block/file 有效、证据子串、分区与来源族一致、人工 `approve` digest 匹配|阻止发布，返回最多 100 条错误|

未纳入现状的能力（见设计稿第 5\.4 节）：NLI 蕴含闸、MinHash/SemDeDup 语义去重、LLM\-as\-Judge 多维打分、无上下文知识增益消融、自动 S/A/B 分层与多样性配额采样。

#### **3\.4\.5 默认任务配方（****`DEFAULT_RECIPE`****）**

|参数|默认值|含义|
|---|---|---|
|`split` / `overlap`|600 / 80|字符窗切分与重叠（须 `overlap < split`）|
|`count`|6|每批教师调用目标样本数|
|`batch_chars`|16000|单批拼接片段字符上限（硬上限 24000 字符 corpus）|
|`max_calls`|200|教师调用预算（含质检占用）|
|`max_samples`|10000|任务候选样本上限|
|`retries`|1|批次失败重试次数|
|`clean` / `privacy` / `dedup`|true|规范化、脱敏、片段去重|

#### **3\.4\.6 发布数据形态（训练集 JSONL）**

```JSON
# 发布快照每行结构（`routes.publish_dataset`）：
{
  "id": "<sample_id>",
  "split": "train",
  "messages": [
    {"role": "user", "content": "<问题>"},
    {"role": "assistant", "content": "<回答>"}
  ],
  "metadata": {
    "dataset": "<数据集名称>",
    "version": "v1.0",
    "source": "<文件名>",
    "source_file_id": "...",
    "source_group": "...",
    "source_hash": "...",
    "block_id": "...",
    "location": "第 3 页 · 2.1 启动前检查",
    "evidence": "<chunk 内连续原文>",
    "goal": "qa",
    "review_status": "approved",
    "revision": 1,
    "review_history": [...]
  }
}
# qa_generate 导出智训 JSONL（`adapters/zhixun.py`）在 `metadata` 中额外保留管线审计字段（`generation_trace`、`filter_trace`、`grade`、`nli_score` 等），便于与设计稿推荐栈产物对比或手工导入。
```

#### **3\.4\.7 现状架构示意**

```Plain Text
┌────────────┐   ┌─────────────────┐   ┌──────────────────┐   ┌─────────────┐
│ MinerU/直读 │──▶│ 章节+字符窗切分  │──▶│ 教师一次生成      │──▶│ 证据子串门   │
│  → Block   │   │ + 治理→TaskBlock │   │ Q+A+evidence+kind │   │ + LLM supported│
└────────────┘   └─────────────────┘   └──────────────────┘   └──────┬──────┘
                                                                      │
                    ┌─────────────────────────────────────────────────┘
                    ▼
            ┌───────────────┐   ┌────────────────┐   ┌──────────────────┐
            │ 人工审核       │──▶│ DatasetVersion │──▶│ LoRA SFT         │
            │ approve+digest │   │ JSONL 快照      │   │ (training_worker) │
            └───────────────┘   └────────────────┘   └──────────────────┘
```

---

### 我们的选择

综合三方向调研结论，推荐采用以下组合路线：

**Question 生成侧**：以**锚点驱动 LLM 反向提问**为基础范式，叠加**知识标签受控进化（Tag-Evol 思路）**提升复杂度与多样性，辅以**知识图谱/知识树覆盖度规划**（GraphGen / Condor 思路）保证知识点不遗漏。具体流程为：Chunk 预处理 → 锚点抽取（NER + 关键短语）→ 按知识点配额生成多类型问题（事实/步骤/条件/比较/多跳）→ 受控进化提升难度 → Round-trip 可回答性预筛。

**Answer 蒸馏侧**：以**单轮响应蒸馏**为默认路径，对推理类/多跳类问题自动切换**CoT 蒸馏**；对边界难题启用**多教师集成 + LLM-as-Judge 选优**；对数学/代码等可验证域叠加**可验证奖励过滤（RFT 思路）**。教师模型采用分级路由：简单问题走低成本模型，难题走强模型。

**知识过滤侧**：采用**六级流水线过滤**——① NLI/Faithfulness 事实蕴含粗筛 → ② Round-trip 可回答性验证 → ③ 无上下文知识增益消融 → ④ 教师一致性检测（SelfCheckGPT，仅高价值样本）→ ⑤ LLM-as-Judge 多维打分 → ⑥ 三层去重（精确 → 语义 → 多样性采样）。最终输出 **S/A/B 三级质量分层**数据集。

> 整体路线可概括为：**锚点反向提问 + 受控进化 → 分级响应蒸馏 → 六级过滤 + S/A/B 分层**。

### 选择理由

1. **锚点驱动而非纯 Self-Instruct**：企业微调场景的核心诉求是知识可溯源、答案可定位。锚点（实体/术语/关键句）保证 Question 天然锚定 Chunk 内容，evidence span 可自动定位，从源头降低幻觉。Self-Instruct 依赖模型内部知识，在领域文档场景下知识过时与不可控问题突出。

2. **受控进化而非纯 Evol-Instruct**：Evol-Instruct 的深度/广度进化能有效提升复杂度，但无约束进化会引入 Chunk 外信息、破坏可回答性。借鉴 Tag-Evol 的"知识标签注入"思路，将进化约束在 Chunk 锚点范围内，既提升难度又保持 grounding。

3. **响应蒸馏为默认 + CoT 按需叠加**：闭源 API 场景下响应蒸馏是唯一可行方案，成本可控（约 $10/万条）。CoT 蒸馏对推理类问题增益显著，但对事实类简洁问答反而增加噪声与成本，因此按问题类型自动路由而非全量 CoT。

4. **六级过滤而非单一 LLM 打分**：单一 LLM-as-Judge 存在 judge 偏差且成本高。六级流水线按成本递增排列——前三级（NLI、round-trip、知识增益）成本低、可规模化，先过滤掉明显劣质样本；后三级（一致性、多维打分、去重）仅对通过粗筛的样本执行，控制整体成本。S/A/B 分层比 pass/fail 更有信息量，可按下游任务灵活配比。

5. **与现状方案的平滑过渡**：现状方案已实现证据子串校验、教师 quality_check、任务内去重和人工审核。推荐路线将其升级为自动化六级过滤 + 自动分层，人工审核从"全量终审"降级为"S 级抽检 + B 级复核"，在提升吞吐的同时保留治理兜底。【待实验验证：E1 基线对比——推荐路线在同等数据量下下游 SFT 效果是否显著优于现状方案】

### 不选择的路线及原因

|路线|不选择原因|
|---|---|
|纯 Self-Instruct / Alpaca 范式|依赖模型内部知识，领域文档场景下知识过时、不可溯源；同质化严重，同一 Chunk 反复生成"请解释 XX"类问题。|
|纯 Evol-Instruct（无锚点约束）|无约束进化会引入 Chunk 外信息，破坏可回答性与知识 grounding；Eliminator 过滤后有效率低，成本浪费。|
|Answer-aware QG（SQuAD 范式）|依赖预标注答案 span，大规模管线中标注成本不可接受；问题类型受限（偏 extractive），难以覆盖步骤说明、条件问答等类型。|
|纯知识图谱驱动（GraphGen / Condor）|需要预先构建高质量领域知识图谱，冷启动成本高；对非结构化文档（如产品手册、操作指南）的图谱构建误差大。作为覆盖度规划的辅助手段而非主路线。|
|难度感知主动生成（QueST / TTCS）|需要学生模型参与闭环评估，迭代周期长，适合数学/代码等可验证域；通用领域 QA 的难度评估缺乏确定性验证器，主动学习收益不确定。【待实验验证：E2 消融——在通用文档场景下，难度感知采样是否比随机采样带来额外增益】|
|Constitutional AI（CAI）|需要 RL 训练（PPO），不直接适用于 QA 数据生成管线；对齐目标与知识准确性目标不同，CAI 主要解决安全性而非事实性。|
|纯多教师集成|成本翻倍（2–3 倍 API 调用），且 judge 本身有偏差。仅对边界难题启用，简单问题走单教师即可。【待实验验证：E3 消融——多教师集成在难题子集上的质量增益是否覆盖额外成本】|

---

## 整体架构

### 端到端架构概览

```Plain Text
┌─────────────────────────────────────────────────────────────────────────────┐
│                           输入层 (Input Layer)                               │
│  企业文档(PDF/Office/MD) → MinerU解析 → Block → 章节+字符窗切分 → TaskChunk  │
│                              │                                              │
│                              ▼                                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                      Question 生成层 (QG Layer)                              │
│  ┌──────────────┐   ┌───────────────┐   ┌────────────────┐   ┌───────────┐  │
│  │ 锚点抽取器    │──▶│ 问题类型配额   │──▶│ LLM反向提问     │──▶│ 受控进化   │  │
│  │ NER+关键短语  │   │ 事实/步骤/    │   │ 锚点+Chunk→Q    │   │ Tag-Evol  │  │
│  │              │   │ 条件/比较/多跳│   │ JSON结构化输出  │   │ 难度提升   │  │
│  └──────────────┘   └───────────────┘   └────────────────┘   └─────┬─────┘  │
│                                                                    │        │
│                              ┌─────────────────────────────────────┘        │
│                              ▼                                              │
│                     Round-trip 可回答性预筛                                  │
│                     (Context+Q → 教师回答 → 与候选A比对)                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                      Answer 蒸馏层 (Distill Layer)                           │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  分级路由：简单问题→低成本教师 │ 推理类→CoT教师 │ 难题→多教师+Judge选优 │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                              │                                              │
│                              ▼                                              │
│              可验证域过滤(数学/代码)：确定性验证器 → 仅保留正确轨迹           │
├─────────────────────────────────────────────────────────────────────────────┤
│                      知识过滤层 (Filter Layer)                               │
│  ┌────────┐  ┌──────────┐  ┌──────────┐  ┌────────┐  ┌────────┐  ┌──────┐ │
│  │①NLI蕴含│─▶│②Round-trip│─▶│③知识增益 │─▶│④教师一致│─▶│⑤多维打分│─▶│⑥去重 │ │
│  │ 粗筛   │  │ 可回答性  │  │ 无上下文 │  │ 性检测  │  │LLM-Judge│  │三层  │ │
│  └────────┘  └──────────┘  └──────────┘  └────────┘  └────────┘  └──────┘ │
│                              │                                              │
│                              ▼                                              │
│                     S/A/B 质量分层 + 多样性配额采样                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                      输出层 (Output Layer)                                   │
│  DatasetVersion 快照(JSONL: messages+metadata) → 人工抽检(S级) → 发布 → SFT │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 模块间数据流说明

**阶段一：输入预处理**

```
原始文档 → MinerU解析 → Block(含loc/kind) → 章节边界优先切分(split=600, overlap=80)
         → 文本规范化 + PII脱敏 + 片段级去重 → TaskChunk 队列
```

每个 TaskChunk 携带：`chunk_id`、`source_file`、`location`（页码/章节）、`content`、`kind`（text/table/image_caption）、`source_group`（分区防泄漏）。

**阶段二：Question 生成**

```
TaskChunk → 锚点抽取器(NER+TF-IDF/RAKE关键短语) → anchors[]
         → 按问题类型配额分配(事实40%/步骤20%/条件15%/比较15%/多跳10%)
         → LLM反向提问(Chunk+anchors+q_type → JSON: question, evidence_span, q_type)
         → 受控进化(Tag-Evol: 注入知识标签做深度/广度进化，约束在Chunk范围内)
         → Round-trip预筛(Chunk+Q → 教师回答 → 语义相似度 > 阈值则保留)
         → 候选 Question 池(含 chunk_id, evidence_span, q_type, evol_level)
```

**阶段三：Answer 蒸馏**

```
候选Question + Chunk → 分级路由器
  ├─ 简单问题(事实类, evol_level≤1) → 低成本教师(如 DeepSeek-Chat) → 简洁答案
  ├─ 推理类(步骤/条件/多跳, evol_level≥2) → 强教师(如 GPT-4o) + CoT 格式
  └─ 边界难题(round-trip预筛低置信) → 多教师(2-3个)并行生成 → LLM-as-Judge选优
         → 可验证域(数学/代码) → 确定性验证器过滤 → 仅保留正确轨迹
         → 候选 QA 对(question, answer, evidence, q_type, teacher, cot_flag)
```

**阶段四：六级过滤**

```
候选QA对 → ①NLI蕴含闸(DeBERTa逐句 entailment > 0.7)
         → ②Round-trip可回答性(k=3次复现, 至少1次语义匹配)
         → ③知识增益消融(无上下文回答正确率 < 0.3, 即必须依赖原文)
         → ④教师一致性检测(SelfCheckGPT, N=5采样, 一致性 > 0.6)【仅S级候选】
         → ⑤LLM-as-Judge多维打分(事实性/相关性/完整性/语言质量, 1-5分)
         → ⑥三层去重(精确哈希 → SemDeDup语义 → 聚类多样性采样)
         → S/A/B分层(S: 总分≥4.5且各维≥4; A: 总分≥3.5; B: 总分≥2.5)
         → 多样性配额采样(按q_type/知识点/难度分层采样)
```

**阶段五：输出与发布**

```
S/A/B分层数据 → 写入 Sample 表(含 filter_trace 审计字段)
             → S级自动通过 + 人工抽检(10%)
             → A级人工复核(30%)
             → B级标记为低质量, 仅用于扩充时按需启用
             → DatasetVersion 快照(JSONL) → 发布 → training_worker 消费
```

全链路审计：每条 QA 数据可追溯到 `chunk_id` → `source_file` → `generation_trace`（教师模型、prompt版本、进化参数）→ `filter_trace`（每级过滤的分数与决策）。



### 设计原则

1. **模块化**：每个模块独立可替换，接口标准化（JSON Schema 约束）。

2. **可观测**：全链路审计日志，每条 QA 数据可追溯到原始 chunk、生成模型、过滤决策。

3. **幂等重试**：失败任务自动重试，支持断点续跑。

4. **成本可控**：分级路由 \+ 批量处理 \+ 缓存命中，降低 API 调用成本。

5. **质量前置**：能在生成阶段约束的，不放到过滤阶段补救。

---

## 模块详细设计

### 5.1 输入预处理模块

**职责**：将原始文档转化为标准化 TaskChunk，作为 QA 生成的最小知识单元。

**输入**：原始文件（PDF/Office/MD/txt/csv/json）、任务配方 `recipe`

**输出**：`TaskChunk` 列表

**核心逻辑**：

1. **解析**：PDF/Office/图片走 MinerU 转 Markdown；txt/md/csv/json 直读。产出工作空间级 `Block`，含 `loc`（页码/章节）、`kind`（text/table/image_caption）。
2. **切分**：章节边界优先，章节内字符窗 `split=600`、`overlap=80`。表格类 Block 保持完整不切分。
3. **治理**：可选开启文本规范化（Unicode 归一化、全半角统一）、PII 脱敏（手机号/身份证/邮箱正则替换）、片段级去重（内容哈希）。
4. **分区**：继承来源资料的 `train`/`validation`/`test` 分区标记，`source_group` 用于防泄漏。

**关键参数**：

|参数|默认值|说明|
|---|---|---|
|`split`|600|字符窗切分大小|
|`overlap`|80|字符窗重叠（须 `overlap < split`）|
|`normalize`|true|文本规范化|
|`privacy`|true|PII 脱敏|
|`dedup`|true|片段级去重|

**异常处理**：解析失败的文件记录错误日志并跳过，不阻断整体任务；切分后为空的 Block 过滤。

---

### 5.2 锚点抽取模块

**职责**：从 TaskChunk 中抽取关键实体、术语和句子，作为 Question 生成的驱动锚点。

**输入**：`TaskChunk`（content, kind）

**输出**：`Anchor[]`，每个 Anchor 含 `text`、`type`（entity/term/key_sentence）、`span`（在 chunk 中的字符偏移）、`importance`（0-1）

**核心逻辑**：

1. **实体抽取**：使用领域 NER 模型（或通用 NER + 领域词典匹配）抽取命名实体（人名、机构、产品、版本号、参数名等）。
2. **术语抽取**：TF-IDF 关键词 + RAKE 关键短语，结合领域术语词典（可配置），识别技术术语。
3. **关键句抽取**：基于位置权重（首句/尾句加权）+ 信息熵评分，抽取 1-2 句核心句。
4. **重要性排序**：综合 IDF 值、出现频率、位置权重计算 `importance`，取 Top-K（默认 K=5）。
5. **去重**：语义相似的锚点合并（embedding 余弦相似度 > 0.85）。

**关键参数**：

|参数|默认值|说明|
|---|---|---|
|`max_anchors`|5|每个 chunk 抽取锚点上限|
|`ner_model`|领域适配|NER 模型，可切换|
|`similarity_threshold`|0.85|锚点去重阈值|

**设计考量**：锚点质量直接决定 Question 生成质量。若 NER 模型在特定领域表现差，可退化为纯 TF-IDF + 关键句模式。【待实验验证：E4 消融——锚点抽取质量对最终 QA 数据质量的影响程度】

---

### 5.3 Question 生成模块

**职责**：基于锚点和 Chunk 内容，按类型配额生成多样化、可回答的问题。

**输入**：`TaskChunk`、`Anchor[]`、问题类型配额配置

**输出**：候选 `Question[]`，含 `question`、`q_type`、`evidence_span`、`evol_level`、`generation_trace`

**核心逻辑**：

#### 5.3.1 问题类型与配额

|q_type|说明|默认配额|示例|
|---|---|---|---|
|`factual`|事实问答：是什么/有哪些|40%|"系统支持哪些文件格式？"|
|`procedural`|步骤说明：怎么做/操作流程|20%|"如何配置批量导入模板？"|
|`conditional`|条件问答：在XX情况下如何|15%|"当证据子串校验失败时应如何处理？"|
|`comparative`|比较问答：A与B的区别|15%|"NLI过滤与round-trip验证有何不同？"|
|`multihop`|多跳推理：需组合多句信息|10%|"结合切分策略和过滤机制，说明为什么overlap必须小于split？"|

#### 5.3.2 LLM 反向提问

对每个锚点，构造 Prompt：

```
你是一个领域专家。基于以下文档片段，围绕锚点「{anchor.text}」生成一个{q_type}类型的问题。

要求：
1. 问题必须能仅基于本文档片段回答
2. 问题应贴近真实用户提问，避免过于宽泛
3. 标注答案在原文中的证据片段（连续原文，不得改写）
4. 输出JSON格式：{{"question": "...", "evidence": "...", "q_type": "..."}}

文档片段：
{chunk.content}
```

JSON 结构化输出，解析失败时重试 1 次，仍失败则跳过该锚点。

#### 5.3.3 受控进化（Tag-Evol 思路）

对生成的基础问题，按 `evol_level` 决定是否进化：

- **evol_level=0**：基础问题，不进化
- **evol_level=1（深度进化）**：添加约束条件（如"在XX配置下"）、具体化（从一般到特定场景）、增加推理要求
- **evol_level=2（广度进化）**：转换问题角度、跨知识点关联（仍约束在同一 chunk 内）

进化 Prompt 中明确注入知识标签（锚点文本 + q_type），要求进化后的问题仍必须能基于原 chunk 回答。进化后重新做 round-trip 预筛。

**关键参数**：

|参数|默认值|说明|
|---|---|---|
|`questions_per_chunk`|3-6|每个 chunk 生成问题数（随 chunk 长度动态调整）|
|`evol_ratio`|0.3|进行进化的问题比例|
|`max_evol_level`|2|最大进化等级|
|`retries`|1|JSON 解析失败重试次数|

---

### 5.4 Round-trip 可回答性预筛模块

**职责**：在 Answer 蒸馏之前，快速验证 Question 是否可基于 Chunk 回答，过滤不可回答的问题。

**输入**：`Question`、`TaskChunk`

**输出**：通过/拒绝 + 置信度分数

**核心逻辑**：

1. 将 `Chunk + Question` 喂给低成本教师模型，要求基于 chunk 回答。
2. 将教师回答与 Question 生成时标注的 `evidence_span` 做语义相似度计算（embedding 余弦相似度）。
3. 相似度 > 阈值（默认 0.7）则通过，否则拒绝。
4. 同时检查教师回答是否包含"无法回答""根据提供的信息"等拒答信号，若有则拒绝。

**设计考量**：此模块在 Answer 蒸馏之前执行，用低成本模型快速过滤，避免对不可回答问题浪费强教师蒸馏成本。

---

### 5.5 Answer 蒸馏模块

**职责**：为通过预筛的 Question 生成高质量 Answer，按问题类型分级路由教师模型。

**输入**：`Question`、`TaskChunk`、`q_type`、`evol_level`

**输出**：`QA` 对，含 `answer`、`teacher`、`cot_flag`、`generation_trace`

**核心逻辑**：

#### 5.5.1 分级路由

|问题特征|路由策略|教师模型|输出格式|
|---|---|---|---|
|事实类 + evol_level≤1|低成本单教师|DeepSeek-Chat / 同级别|简洁答案（2-5句）|
|步骤/条件/比较类|强教师单教师|GPT-4o / Claude / 同级别|结构化答案（分点）|
|多跳类 + evol_level≥2|强教师 + CoT|GPT-4o / R1 类推理模型|CoT 推理链 + 最终答案|
|round-trip 低置信|多教师集成|2-3个教师并行|LLM-as-Judge 选优|

#### 5.5.2 CoT 蒸馏（推理类问题）

Prompt 要求教师按以下格式输出：

```
## 推理过程
（逐步分析，引用原文证据）

## 最终答案
（简洁明确的答案）
```

训练时可选择仅用"最终答案"做 SFT，或用完整 CoT 做 SFT（由配方 `cot_in_training` 控制）。

#### 5.5.3 多教师集成 + Judge 选优

对边界难题，同时调用 2-3 个教师生成答案，构造 Judge Prompt：

```
请从以下{num_teachers}个回答中选出最优答案，从事实准确性、完整性、语言质量三个维度评分（1-5分）。

问题：{question}
文档片段：{chunk.content}

回答A：{answer_a}
回答B：{answer_b}
...

输出JSON：{{"best": "A/B/...", "scores": {{"A": {{"accuracy":..., "completeness":..., "quality":...}}, ...}}}}
```

#### 5.5.4 可验证域过滤（数学/代码）

若问题属于数学/代码域（由 q_type 或领域分类器判定），对教师输出执行确定性验证：
- 数学：提取最终数值/表达式，与标准答案（若有）或符号计算引擎比对
- 代码：在沙箱中执行单元测试，pass 则保留

**关键参数**：

|参数|默认值|说明|
|---|---|---|
|`cheap_teacher`|DeepSeek-Chat|低成本教师模型|
|`strong_teacher`|GPT-4o|强教师模型|
|`cot_teacher`|GPT-4o / R1|CoT 推理教师|
|`multi_teacher_count`|2|多教师集成数量|
|`cot_in_training`|false|训练时是否包含CoT|

---

### 5.6 六级过滤模块

#### 5.6.1 第一级：NLI 事实蕴含粗筛

**职责**：逐句判断 Answer 是否被 Chunk 蕴含，过滤明显幻觉。

**实现**：DeBERTa-v3-large-mnli 模型，将 Answer 拆分为句子，逐句与 Chunk 做 entailment 判断。

**通过条件**：所有句子的 entailment 概率 > 0.7；若存在 contradiction 概率 > 0.5 的句子则直接拒绝。

**成本**：低（开源模型本地推理，<10ms/句）

#### 5.6.2 第二级：Round-trip 可回答性验证

**职责**：验证模型能否基于 Chunk 复现 Answer，过滤问题歧义或答案不可推导的样本。

**实现**：将 `Chunk + Question` 喂给教师模型，k=3 次采样回答，与蒸馏 Answer 做语义相似度比对。

**通过条件**：至少 1 次回答与蒸馏 Answer 的语义相似度 > 0.75；或 3 次回答的平均相似度 > 0.6。

**参考**：Source2Synth 在多跳 QA 上以此过滤约 13% 低质量样本。

#### 5.6.3 第三级：知识增益消融（无上下文验证）

**职责**：验证问题是否必须依赖 Chunk 才能回答，过滤模型内部知识即可回答的"无增益"问题。

**实现**：遮住 Chunk，仅给 Question 让模型回答（k=2 次），判断回答正确率。

**通过条件**：无上下文回答的正确率 < 0.3（即模型不看原文基本答不对）。若模型不看原文也能答对，说明该问题对领域知识微调无增益。

**参考**：InstructRetro 系统对比"有检索 vs 无检索"性能差异，仅保留有检索显著提升的样本。

#### 5.6.4 第四级：教师一致性检测（SelfCheckGPT）

**职责**：对高价值候选样本，检测教师答案的稳定性，过滤教师不确定或存在幻觉的样本。

**实现**：对同一 `Chunk + Question` 采样 N=5 个答案，计算答案间语义一致性（平均两两相似度）。

**通过条件**：一致性 > 0.6。低一致性说明教师对该问题不确定，答案标签不可靠。

**成本控制**：仅对通过前三级且目标为 S 级的样本执行，不做全量检测。

#### 5.6.5 第五级：LLM-as-Judge 多维打分

**职责**：对 QA 对做综合质量评分，为 S/A/B 分层提供依据。

**实现**：强 LLM 按四个维度打分（1-5 分）：

|维度|说明|
|---|---|
|事实性|答案是否完全被原文支持，无幻觉|
|相关性|问题与答案是否匹配，是否答非所问|
|完整性|答案是否覆盖了问题所需的全部信息|
|语言质量|表达是否清晰、简洁、专业|

**输出**：`{"accuracy": x, "relevancy": x, "completeness": x, "quality": x, "total": x}`

**参考**：AlpaGasus 用 GPT-3.5 打分筛选 9K/52K 样本，训练效果超越全量；NVIDIA Nemotron RM 五维打分。

#### 5.6.6 第六级：三层去重

**职责**：去除语义重复的 QA 对，保证数据集多样性。

**实现**：

1. **精确去重**：Question + Answer 规范化哈希，完全相同则去重。
2. **语义去重（SemDeDup）**：Question embedding 聚类，类内余弦相似度 > 0.9 的保留质量分最高的 1 条。
3. **多样性采样**：按 q_type / 知识点 / 难度分层采样，控制各类型比例，避免某类过载。

**参考**：SemDeDup 在 LAION 上删除 50% 数据后性能几乎不降，OOD 性能反而提升。

---

### 5.7 质量分层与多样性采样模块

**职责**：根据过滤分数将 QA 对分为 S/A/B 三级，并按多样性配额采样输出。

**分层规则**：

|等级|条件|用途|人工审核|
|---|---|---|---|
|S|多维打分总分 ≥ 4.5 且各维 ≥ 4，通过全部六级过滤|核心训练集，优先使用|抽检 10%|
|A|多维打分总分 ≥ 3.5，通过前五级过滤|补充训练集|复核 30%|
|B|多维打分总分 ≥ 2.5，通过前四级过滤|低质量池，仅在数据量不足时按需启用|不自动发布|

**多样性配额**：

- q_type 比例：事实 ≤ 50%、步骤 ≥ 15%、条件 ≥ 10%、比较 ≥ 10%、多跳 ≥ 5%
- 难度分布：evol_level=0/1/2 比例约 4:4:2
- 知识点覆盖：每个知识点至少 1 条，热门知识点上限 20 条（防止过拟合）

---

### 5.8 数据入库与版本管理模块

**职责**：将过滤后的 QA 对写入数据库，支持版本化发布和审计追溯。

**数据模型**：

```
Sample {
  id: string
  task_id: string
  chunk_id: string
  question: string
  answer: string
  evidence: string
  q_type: string
  evol_level: int
  teacher: string
  cot_flag: bool
  grade: "S" | "A" | "B"
  filter_trace: {
    nli_score: float,
    roundtrip_score: float,
    knowledge_gain_score: float,
    consistency_score: float,
    judge_scores: {accuracy, relevancy, completeness, quality},
    dedup_group_id: string
  }
  generation_trace: {
    anchor: string,
    prompt_version: string,
    evol_params: object,
    teacher_model: string
  }
  metadata: {
    source_file, source_file_id, source_group, source_hash,
    block_id, location, split, goal
  }
  status: "pending" | "approved" | "rejected"
  review_history: Review[]
}
```

**发布流程**：S/A 级样本经人工审核后，发布为不可变 `DatasetVersion` 快照（JSONL 格式），版本号 `v1.{n}` 自增。发布时执行硬校验：非空 Q/A、block/file 有效、证据子串、分区与来源族一致。

## 关键技术选型表

|模块|技术选型|备选方案|选型理由|
|---|---|---|---|
|文档解析|MinerU|Unstructured / PyPDF2|MinerU 对中文 PDF/表格/公式解析效果优，与现状方案一致|
|Chunk 切分|章节边界优先 + 字符窗(split=600, overlap=80)|语义切分 / 递归切分|章节边界保证语义完整，字符窗保证粒度均匀；与现状兼容|
|锚点抽取|NER + TF-IDF/RAKE + 关键句|纯 LLM 抽取|规则+模型混合速度快、可解释；LLM 抽取成本高且不稳定|
|Question 生成|锚点驱动 LLM 反向提问 + Tag-Evol 受控进化|纯 Self-Instruct / Answer-aware QG|锚点保证可溯源，受控进化保证复杂度且不破坏 grounding|
|低成本教师|DeepSeek-Chat（或同级开源/API）|Qwen / Llama|中文效果好、API 成本低，适合批量生成和 round-trip 预筛|
|强教师|GPT-4o（或同级）|Claude / Gemini|推理能力强，适合难题和 CoT 蒸馏；多教师场景可混用|
|CoT 教师|GPT-4o / DeepSeek-R1 类|o1 / Claude Opus|推理轨迹质量高，R1 范式已验证蒸馏有效性|
|NLI 蕴含|DeBERTa-v3-large-mnli|RoBERTa-mnli / BERT-mnli|工业界最常用，中文/英文效果均衡，推理速度快|
|语义相似度|Sentence-BERT (bge-large-zh)|OpenAI embedding / E5|中文语义匹配效果好，可本地部署降低成本|
|Round-trip 验证|低成本教师 k=3 采样 + 语义相似度|ARES 语义检索 round-trip|实现简单、与蒸馏模块复用教师调用；ARES 需额外索引|
|知识增益消融|无上下文 k=2 采样 + 正确率评估|QuALITY 人工标注标准|自动化可规模化；人工标注成本高仅适合评测集|
|教师一致性|SelfCheckGPT N=5 采样|外部 NLI 交叉验证|无需外部模型，仅对 S 级候选执行控制成本|
|多维打分|LLM-as-Judge (强教师四维度)|Reward Model (Nemotron RM)|LLM Judge 灵活可配置维度；RM 需额外训练但推理成本低|
|精确去重|内容哈希（MurmurHash3）|MD5 / SHA-1|速度快，碰撞率可接受|
|语义去重|SemDeDup (embedding 聚类)|MinHash LSH|SemDeDup 语义去重效果优于 n-gram 方法，可删 ~50% 不掉点|
|数据存储|PostgreSQL (Sample 表) + 对象存储(JSONL快照)|MongoDB / 纯文件|关系型适合审计查询和过滤追溯，JSONL 快照兼容训练框架|
|任务编排|Python 异步 + 批量队列|Airflow / Celery|轻量可控，与现有 `data_governance/backend` 技术栈一致|

---

## 评估指标体系

### 数据质量指标

|指标|定义|计算方式|目标值|
|---|---|---|---|
|事实蕴含率|Answer 被 Chunk 蕴含的比例|NLI entailment > 0.7 的样本占比|≥ 95%|
|可回答率|Round-trip 验证通过的比例|k=3 次中至少 1 次语义匹配 > 0.75|≥ 90%|
|知识增益率|必须依赖原文才能回答的比例|无上下文回答正确率 < 0.3 的样本占比|≥ 70%|
|教师一致性|S 级样本的 SelfCheckGPT 一致性|N=5 采样平均两两相似度|≥ 0.6|
|多维打分均值|LLM-as-Judge 四维度平均分|(accuracy+relevancy+completeness+quality)/4|S级 ≥ 4.5，A级 ≥ 3.5|
|证据定位率|evidence 为 chunk 内连续原文的比例|子串匹配校验通过率|≥ 98%|
|问题类型分布|各 q_type 占比|统计 factual/procedural/conditional/comparative/multihop|符合配额（事实≤50%，多跳≥5%）|
|语义重复率|去重后剩余重复比例|SemDeDup 类内相似度 > 0.9 的对数 / 总数|≤ 5%|
|幻觉率|人工抽检中发现幻觉的比例|S 级抽检 10%，标注幻觉样本占比|≤ 2%【待实验验证：E5 人工评测】|

### 管线效率指标

|指标|定义|计算方式|目标值|
|---|---|---|---|
|端到端吞吐|每小时产出合格 QA 对数|合格入库数 / 耗时|≥ 5000 条/小时（单 worker）|
|教师调用成本|每万条合格 QA 的 API 费用|总 API 费用 / 合格数 × 10000|≤ $50/万条【待实验验证：E6 成本分析】|
|各级过滤通过率|每级过滤的保留比例|通过数 / 输入数|NLI ≥ 80%，Round-trip ≥ 85%，知识增益 ≥ 60%|
|整体合格率|最终入库数 / 初始候选数|入库数 / 生成候选数|≥ 30%|
|平均延迟|单条 QA 从生成到入库的耗时|P50 / P95 延迟|P50 ≤ 30s，P95 ≤ 120s|
|失败重试率|需要重试的批次比例|重试次数 / 总批次数|≤ 5%|

### 下游效果指标

|指标|定义|计算方式|目标值|
|---|---|---|---|
|SFT 后领域问答准确率|在领域评测集上的准确率|学生模型微调后在 held-out 评测集上的正确率|相比基线提升 ≥ 5 个百分点【待实验验证：E1 基线对比】|
|SFT 后拒答率|面对无依据问题时的拒答比例|构造干扰文档测试集，模型正确拒答的比例|≥ 80%（RAFT 思路验证）【待实验验证：E7 拒答能力】|
|SFT 后事实遵循率|答案被检索文档支持的比例|RAGAS faithfulness 评分|≥ 0.9【待实验验证：E1】|
|OOD 泛化|在未见过文档类型上的表现|跨领域评测集准确率|相比基线不下降【待实验验证：E1】|
|训练效率|达到目标性能所需的训练数据量|不同数据量下的学习曲线|S/A 级数据用 50% 量达到全量基线性能【待实验验证：E8 数据效率】|

## 风险与应对

|风险|影响|概率|应对措施|
|---|---|---|---|
|教师模型幻觉传播|Answer 中的幻觉被当作训练标签，学生模型学到错误知识|中|① NLI 蕴含粗筛 + round-trip 验证双重过滤；② 教师一致性检测过滤不稳定样本；③ S 级人工抽检；④ 可验证域用确定性验证器|
|教师模型 API 不稳定/限流|管线中断、批次失败、延迟增加|中|① 多教师模型 fallback 配置；② 指数退避重试（retries=3）；③ 批量队列削峰；④ 关键教师响应缓存（相同 Chunk+Q 命中缓存）|
|过滤过严导致数据量不足|合格样本过少，无法满足训练需求|中|① 分级输出 S/A/B，B 级可按需启用；② 过滤阈值可配置，初期放宽后逐步收紧；③ 增加生成量补偿过滤损失；④ 多样性采样保证覆盖而非纯数量|
|过滤过松导致低质量数据入库|训练效果下降，甚至比基线更差|中|① 六级过滤逐级收紧，前三级粗筛 + 后三级精筛；② S/A/B 分层，S 级高门槛；③ 人工抽检兜底；④ 定期用下游 SFT 效果反哺阈值调优|
|锚点抽取质量差|Question 偏离核心知识点，生成大量低价值问题|中|① 混合 NER + TF-IDF + 关键句多策略；② 领域术语词典可配置；③ 锚点重要性排序取 Top-K；④ 【待实验验证：E4 消融锚点策略对质量的影响】|
|Evol 进化破坏 grounding|进化后的问题超出 Chunk 范围，无法回答|中|① 受控进化注入知识标签约束在 Chunk 内；② 进化后重新 round-trip 预筛；③ evol_level 上限为 2，避免过度复杂化|
|LLM-as-Judge 偏差|打分模型偏好特定风格/长度，导致评分不公|低|① 四维度分开打分而非单一总分；② Judge Prompt 含明确评分标准和 few-shot 示例；③ 定期用人工标注校准 Judge 相关性；④ 多 Judge 投票（高价值样本）|
|数据泄漏（train/test 同源）|评测结果虚高，泛化能力误判|低|① `source_group` 分区防泄漏，同一来源文档不跨分区；② 发布前硬校验分区一致性；③ 语义去重跨分区执行|
|成本超预算|六级过滤 + 多教师导致 API 成本过高|中|① 分级路由：简单问题低成本教师；② 四级一致性检测仅对 S 级候选执行；③ 批量调用 + 缓存；④ 【待实验验证：E6 成本分析，验证每万条 ≤ $50 目标】|
|中文 NLI 模型效果不足|DeBERTa-mnli 主要训练英文，中文蕴含判断误差|低|① 选用中文适配版本（bge-reranker 或中文 NLI 微调版）；② NLI 仅作粗筛，后续 LLM Judge 复核；③ 关键领域可微调 NLI 模型|

---

## 实验设计

本节定义验证推荐技术路线有效性的实验方案。所有实验基于统一的评测集和训练配置，确保可比性。需要实验证明的设计结论已在正文中标注对应实验编号。

### 实验环境与基线

**数据集**：
- 训练语料：3 个领域的企业内部文档（技术文档、产品手册、操作指南），共约 500 个 Chunk
- 评测集：人工标注的领域 QA 评测集，每领域 100 题，含事实/步骤/条件/比较/多跳五类
- 干扰测试集：构造含干扰文档的 RAG 测试集，用于拒答能力评估

**学生模型**：统一使用 7B 开源模型（如 Qwen2.5-7B），LoRA SFT，相同训练超参。

**基线（Baseline）**：智训平台现状方案（`baseline_platform` Recipe），即一次调用生成 Q+A+evidence + 规则级过滤 + 人工审核。

### E1：端到端基线对比

**目标**：验证推荐路线在同等数据量下，下游 SFT 效果是否显著优于现状方案。

**实验设置**：

|组别|数据构造方式|数据量|
|---|---|---|
|Baseline|现状方案（一次生成 + 规则过滤 + 人工审核）|10K 条|
|Ours-Full|推荐路线全量（锚点+进化+蒸馏+六级过滤+S/A/B分层）|10K 条（S+A级）|
|Ours-S-only|推荐路线仅 S 级|约 3-5K 条|

**评估指标**：
- 领域评测集准确率（分 q_type 统计）
- RAGAS faithfulness（事实遵循率）
- 拒答率（干扰测试集）
- OOD 泛化（跨领域文档测试）

**预期结论**：Ours-Full 准确率比 Baseline 高 ≥ 5 个百分点；Ours-S-only 用更少数据达到或超过 Baseline。【验证正文：选择理由第5点、下游效果指标】

### E2：Question 生成策略消融

**目标**：验证各 Question 生成组件的增量贡献。

**实验设置**：

|组别|Question 生成策略|
|---|---|
|A1|纯 Self-Instruct（无锚点，无进化）|
|A2|锚点驱动反向提问（无进化）|
|A3|锚点 + 受控进化（Tag-Evol）|
|A4|锚点 + 纯 Evol-Instruct（无约束）|
|A5|锚点 + 进化 + 难度感知采样（QueST 思路）|

**评估指标**：
- 可回答率（round-trip 通过率）
- 问题类型多样性（熵值）
- 下游 SFT 准确率
- 单条生成成本

**预期结论**：A3 > A2 > A1；A4 可回答率显著低于 A3（验证受控进化的必要性）；A5 在通用文档场景下增益不确定。【验证正文：不选择纯Evol-Instruct、难度感知主动生成的原因】

### E3：多教师集成消融

**目标**：验证多教师集成在难题子集上的质量增益是否覆盖额外成本。

**实验设置**：

|组别|Answer 蒸馏策略|适用范围|
|---|---|---|
|B1|单教师（低成本）|全量|
|B2|单教师（强模型）|全量|
|B3|分级路由（简单→低成本，难题→强教师）|全量|
|B4|多教师集成 + Judge 选优|仅难题子集（约 20%）|

**评估指标**：
- 难题子集（多跳/高 evol_level）的 Answer 质量分
- 全量平均成本
- 下游 SFT 准确率（难题子集表现）

**预期结论**：B4 在难题子集上质量分最高，但全量成本增幅可控（仅 20% 样本走多教师）；B3 是性价比最优解。【验证正文：不选择纯多教师集成的原因】

### E4：锚点抽取策略消融

**目标**：验证锚点抽取质量对最终 QA 数据质量的影响。

**实验设置**：

|组别|锚点抽取策略|
|---|---|
|C1|无锚点（纯 LLM 自由生成）|
|C2|仅 TF-IDF 关键词|
|C3|NER + TF-IDF（默认方案）|
|C4|NER + TF-IDF + 关键句|
|C5|纯 LLM 抽取锚点|

**评估指标**：
- 证据定位率（evidence 子串匹配通过率）
- NLI 蕴含通过率
- 人工评估的问题相关性
- 下游 SFT 准确率

**预期结论**：C3/C4 显著优于 C1/C2；C5 质量略高但成本显著增加，C3 为性价比最优。【验证正文：锚点抽取模块设计考量】

### E5：人工质量评测

**目标**：通过人工标注评估最终入库数据的真实质量，校准自动化指标。

**实验设置**：
- 从 S 级数据中随机抽检 200 条，A 级抽检 100 条
- 3 名标注者独立标注，维度：事实正确性、问题可回答性、答案完整性、语言质量
- 计算标注者间一致性（Cohen's Kappa）

**评估指标**：
- 幻觉率（事实错误样本占比）
- 各维度人工评分均值
- 自动化指标与人工评分的相关性（Pearson r）

**预期结论**：S 级幻觉率 ≤ 2%；LLM-as-Judge 评分与人工评分相关性 r ≥ 0.7。【验证正文：数据质量指标-幻觉率】

### E6：成本与效率分析

**目标**：量化管线的吞吐和成本，验证成本目标。

**实验设置**：
- 全量运行 10K 候选 QA 的生成+过滤管线
- 记录各阶段耗时、API 调用次数、token 消耗、费用
- 按教师模型、过滤级别分别统计成本占比

**评估指标**：
- 每万条合格 QA 的 API 费用
- 端到端吞吐（条/小时）
- 各阶段成本占比饼图
- 各级过滤通过率（漏斗图）

**预期结论**：每万条合格 QA 成本 ≤ $50；低成本教师占总调用 ≥ 60%；六级过滤中前三级（低成本）过滤掉 ≥ 50% 候选。【验证正文：管线效率指标、风险-成本超预算】

### E7：拒答能力验证

**目标**：验证数据管线是否教会模型在无依据时拒答而非胡编。

**实验设置**：
- 构造干扰测试集：黄金文档 + 4 篇同领域干扰文档，其中 20% 样本完全不含黄金文档
- 对比 Baseline 和 Ours-Full 训练的模型在干扰测试集上的表现

**评估指标**：
- 纯干扰样本的拒答率（正确识别"无法回答"）
- 含黄金文档样本的准确率（不应过度拒答）
- 整体 F1（拒答精确率 + 召回率平衡）

**预期结论**：Ours-Full 拒答率 ≥ 80%，显著高于 Baseline；含黄金文档样本准确率不下降。【验证正文：下游效果指标-拒答率、RAFT 思路借鉴】

### E8：数据效率验证

**目标**：验证质量分层后，少量高质量数据能否达到全量基线性能。

**实验设置**：

|组别|训练数据|数据量|
|---|---|---|
|D1|Baseline 全量|10K|
|D2|Ours S 级|约 3-5K|
|D3|Ours S+A 级|约 6-8K|
|D4|Ours S+A+B 级|10K|

**评估指标**：
- 学习曲线（不同训练步数下的验证准确率）
- 最终准确率
- 训练成本（GPU 小时）

**预期结论**：D2（S 级，约 50% 数据量）达到 D1 全量性能；D3 超过 D1。【验证正文：下游效果指标-训练效率、AlpaGasus 结论复现】

### 实验执行优先级

|优先级|实验|目的|
|---|---|---|
|P0（必做）|E1 端到端基线对比|验证整体路线有效性|
|P0（必做）|E6 成本效率分析|验证工程可行性|
|P1（重要）|E2 Question 生成消融|验证核心组件贡献|
|P1（重要）|E5 人工质量评测|校准自动化指标|
|P2（补充）|E3 多教师消融|优化成本配置|
|P2（补充）|E7 拒答能力验证|验证 RAFT 思路效果|
|P3（可选）|E4 锚点策略消融|精细调优|
|P3（可选）|E8 数据效率验证|验证分层价值|