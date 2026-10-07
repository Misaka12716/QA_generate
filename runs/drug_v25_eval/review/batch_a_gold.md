# batch_a_gold 金标材料

每题给出完整 system 与 user，供对照 CSV。模型身份不在这里出现。
ai_note 只是准备阶段的提示，不是审核意见。

## qa_3cfb9b13f6e4 / original

- content_hash: `f5506359b24bfe46452219cb9d0171ba3b1791d6b96f3321d0aadada972f4af3`
- core_question_id: `qfam_4c3584eb34de5a4fd850`
- expected_action: `answer`
- evidence_state: `sufficient`

### 实际输入

**system**

```text
仅依据提供的资料回答；资料不足时说明缺少的信息。
```

**user**

```text
资料：
警示语：严重皮肤反应和HLA－B*1502等位基因在卡马西平治疗期间有报告发生严重且有时是致命的皮肤反应，包括中毒性表皮坏死松懈症（TEN）和Stevens－Johnson综合征（SJS）。在主要是高加索人群的国家中每10000名新用药者估计发生1～6例。但是这一风险在一些亚洲国家估计约比上述国家高10倍。对华裔患者的研究发现SJS/TEN的发生风险与患者体内携带人白细胞抗原HLA－B*1502等位基因之间存在很强的相关性，HLA－B*1502等位基因是HLA－B基因的遗传性等位基因变异体。H－LA－B*1502几乎仅在祖籍亚洲广泛地区的患者人群中发现。在开始卡马西平治疗前可对遗传风险人群患者进行HLA－B*1502筛查。此等位基因阳性患者不得使用卡马西平治疗，除非明确显示治疗效益大于风险（参见【注意事项】）。再生障碍性贫血和粒细胞缺乏症据报告，再生障碍性贫血和粒细胞缺乏症与使用卡马西平有关。来自基于人群的病例对照研究显示用药患者群体发生这些反应的风险比普通人群要高5～8倍，未接受治疗的普通人群这些反应的总体风险比较低，粒细胞缺乏症发生率平均每年每百万人中为6例，再生障碍性贫血平均每年每百万人中为2例。尽管使用卡马西平经常报告发生血小板或白细胞计数一过性或持续减少，但尚无数据来准确估计这些情况的发生率或结局。

问题：资料指出，HLA-B*1502等位基因阳性患者使用卡马西平治疗的限制条件是什么？
```

### 当前金标

- answer: 不得使用卡马西平治疗，除非明确显示治疗效益大于风险
- required_points: ["不得使用卡马西平治疗，除非明确显示治疗效益大于风险"]
- unavailable_points: []
- answerable_part: 
- insufficient_part: 

## qa_3cfb9b13f6e4__rephrase_1 / rephrase_1

- content_hash: `349cdf8681bd2e825c349cbf60c055c4bbfdd86a46d4fc640495964535b56c18`
- core_question_id: `qfam_4c3584eb34de5a4fd850`
- expected_action: `answer`
- evidence_state: `sufficient`

### 实际输入

**system**

```text
仅依据提供的资料回答；资料不足时说明缺少的信息。
```

**user**

```text
资料：
警示语：严重皮肤反应和HLA－B*1502等位基因在卡马西平治疗期间有报告发生严重且有时是致命的皮肤反应，包括中毒性表皮坏死松懈症（TEN）和Stevens－Johnson综合征（SJS）。在主要是高加索人群的国家中每10000名新用药者估计发生1～6例。但是这一风险在一些亚洲国家估计约比上述国家高10倍。对华裔患者的研究发现SJS/TEN的发生风险与患者体内携带人白细胞抗原HLA－B*1502等位基因之间存在很强的相关性，HLA－B*1502等位基因是HLA－B基因的遗传性等位基因变异体。H－LA－B*1502几乎仅在祖籍亚洲广泛地区的患者人群中发现。在开始卡马西平治疗前可对遗传风险人群患者进行HLA－B*1502筛查。此等位基因阳性患者不得使用卡马西平治疗，除非明确显示治疗效益大于风险（参见【注意事项】）。再生障碍性贫血和粒细胞缺乏症据报告，再生障碍性贫血和粒细胞缺乏症与使用卡马西平有关。来自基于人群的病例对照研究显示用药患者群体发生这些反应的风险比普通人群要高5～8倍，未接受治疗的普通人群这些反应的总体风险比较低，粒细胞缺乏症发生率平均每年每百万人中为6例，再生障碍性贫血平均每年每百万人中为2例。尽管使用卡马西平经常报告发生血小板或白细胞计数一过性或持续减少，但尚无数据来准确估计这些情况的发生率或结局。

问题：HLA-B*1502等位基因阳性的患者，使用卡马西平治疗时有什么限制？
```

### 当前金标

- answer: 不得使用卡马西平治疗，除非明确显示治疗效益大于风险
- required_points: ["不得使用卡马西平治疗，除非明确显示治疗效益大于风险"]
- unavailable_points: []
- answerable_part: 
- insufficient_part: 

## qa_3cfb9b13f6e4__rephrase_2 / rephrase_2

- content_hash: `fa6a27b86670bfe871d239d5b297106cfaae886efc2c1fd4eb0fb8d0fcf40455`
- core_question_id: `qfam_4c3584eb34de5a4fd850`
- expected_action: `answer`
- evidence_state: `sufficient`

### 实际输入

**system**

```text
仅依据提供的资料回答；资料不足时说明缺少的信息。
```

**user**

```text
资料：
警示语：严重皮肤反应和HLA－B*1502等位基因在卡马西平治疗期间有报告发生严重且有时是致命的皮肤反应，包括中毒性表皮坏死松懈症（TEN）和Stevens－Johnson综合征（SJS）。在主要是高加索人群的国家中每10000名新用药者估计发生1～6例。但是这一风险在一些亚洲国家估计约比上述国家高10倍。对华裔患者的研究发现SJS/TEN的发生风险与患者体内携带人白细胞抗原HLA－B*1502等位基因之间存在很强的相关性，HLA－B*1502等位基因是HLA－B基因的遗传性等位基因变异体。H－LA－B*1502几乎仅在祖籍亚洲广泛地区的患者人群中发现。在开始卡马西平治疗前可对遗传风险人群患者进行HLA－B*1502筛查。此等位基因阳性患者不得使用卡马西平治疗，除非明确显示治疗效益大于风险（参见【注意事项】）。再生障碍性贫血和粒细胞缺乏症据报告，再生障碍性贫血和粒细胞缺乏症与使用卡马西平有关。来自基于人群的病例对照研究显示用药患者群体发生这些反应的风险比普通人群要高5～8倍，未接受治疗的普通人群这些反应的总体风险比较低，粒细胞缺乏症发生率平均每年每百万人中为6例，再生障碍性贫血平均每年每百万人中为2例。尽管使用卡马西平经常报告发生血小板或白细胞计数一过性或持续减少，但尚无数据来准确估计这些情况的发生率或结局。

问题：对HLA-B*1502等位基因阳性患者，卡马西平治疗受到哪些限制？
```

### 当前金标

- answer: 不得使用卡马西平治疗，除非明确显示治疗效益大于风险
- required_points: ["不得使用卡马西平治疗，除非明确显示治疗效益大于风险"]
- unavailable_points: []
- answerable_part: 
- insufficient_part: 

## qa_086693f5f4ad / original

- content_hash: `b4e538c016e48ec47da0a82cbeeec6bcbf8cf134d231a5862ecb62265a90de32`
- core_question_id: `qfam_af536f77618a221e6c5c`
- expected_action: `answer`
- evidence_state: `sufficient`

### 实际输入

**system**

```text
仅依据提供的资料回答；资料不足时说明缺少的信息。
```

**user**

```text
资料：
警告本品辅料中含有苯甲醇，禁止用于儿童肌肉注射。本品与林可霉素、克林霉素有交叉耐药性，对克林霉素或林可霉素有过敏史者禁用。禁止与氨苄青霉素、苯妥英钠、巴比妥盐、氨茶碱、葡萄糖酸钙及硫酸镁配伍；与红霉素呈拮抗作用，不宜合用。包括克林霉素磷酸酯在内的几乎所有的抗生素都会引起艰难梭菌（Clostridium
difficile）相关性腹泻（CDAD），严重程度可由轻度腹泻到致命性结肠炎。抗菌药物治疗可改变肠道正常菌群，导致艰难梭菌的过度生长。使用克林霉素磷酸酯注射液后可能引起重度结肠炎，重度结肠炎有致命风险，因此本品只适用于毒性较低的抗菌药无法治疗的严重感染。克林霉素磷酸酯不得用于非细菌性感染，如大部分上呼吸道感染。艰难梭菌可产生毒素A和毒素B，进而促进CDAD的发生。产超毒素的艰难梭菌菌株可导致发病率和死亡率增高，由于这些感染会对抗菌药物治疗无效，可能需要结肠切除术治疗。对于使用抗生素后出现腹泻的所有患者应考虑CDAD。由于曾经有给予抗菌药物治疗超过2个月后发生CDAD的报道，因此有必要仔细询问病史。若怀疑或已确诊为CDAD，则需要停用不针对艰难梭菌的抗菌治疗，需要根据临床指征适当的调节水和电解质，补充蛋白质，并给予针对艰难梭菌的抗菌治疗，必要时进行手术评估。

问题：根据资料，克林霉素磷酸酯禁止用于哪些人群的肌肉注射？
```

### 当前金标

- answer: 禁止用于儿童肌肉注射。
- required_points: ["禁止用于儿童肌肉注射。"]
- unavailable_points: []
- answerable_part: 
- insufficient_part: 

## qa_086693f5f4ad__rephrase_1 / rephrase_1

- content_hash: `89462860c79e8f4c09c077cd938e5a9a74785b955c829aef98f5a97d57c130ec`
- core_question_id: `qfam_af536f77618a221e6c5c`
- expected_action: `answer`
- evidence_state: `sufficient`

### 实际输入

**system**

```text
仅依据提供的资料回答；资料不足时说明缺少的信息。
```

**user**

```text
资料：
警告本品辅料中含有苯甲醇，禁止用于儿童肌肉注射。本品与林可霉素、克林霉素有交叉耐药性，对克林霉素或林可霉素有过敏史者禁用。禁止与氨苄青霉素、苯妥英钠、巴比妥盐、氨茶碱、葡萄糖酸钙及硫酸镁配伍；与红霉素呈拮抗作用，不宜合用。包括克林霉素磷酸酯在内的几乎所有的抗生素都会引起艰难梭菌（Clostridium
difficile）相关性腹泻（CDAD），严重程度可由轻度腹泻到致命性结肠炎。抗菌药物治疗可改变肠道正常菌群，导致艰难梭菌的过度生长。使用克林霉素磷酸酯注射液后可能引起重度结肠炎，重度结肠炎有致命风险，因此本品只适用于毒性较低的抗菌药无法治疗的严重感染。克林霉素磷酸酯不得用于非细菌性感染，如大部分上呼吸道感染。艰难梭菌可产生毒素A和毒素B，进而促进CDAD的发生。产超毒素的艰难梭菌菌株可导致发病率和死亡率增高，由于这些感染会对抗菌药物治疗无效，可能需要结肠切除术治疗。对于使用抗生素后出现腹泻的所有患者应考虑CDAD。由于曾经有给予抗菌药物治疗超过2个月后发生CDAD的报道，因此有必要仔细询问病史。若怀疑或已确诊为CDAD，则需要停用不针对艰难梭菌的抗菌治疗，需要根据临床指征适当的调节水和电解质，补充蛋白质，并给予针对艰难梭菌的抗菌治疗，必要时进行手术评估。

问题：克林霉素磷酸酯的肌肉注射禁止用于哪些人群？
```

### 当前金标

- answer: 禁止用于儿童肌肉注射。
- required_points: ["禁止用于儿童肌肉注射。"]
- unavailable_points: []
- answerable_part: 
- insufficient_part: 

## qa_086693f5f4ad__rephrase_2 / rephrase_2

- content_hash: `c9fc1b521b124d4de5081f235be4742a67071cb8a4f092ea4de5eb1e1c86c249`
- core_question_id: `qfam_af536f77618a221e6c5c`
- expected_action: `answer`
- evidence_state: `sufficient`

### 实际输入

**system**

```text
仅依据提供的资料回答；资料不足时说明缺少的信息。
```

**user**

```text
资料：
警告本品辅料中含有苯甲醇，禁止用于儿童肌肉注射。本品与林可霉素、克林霉素有交叉耐药性，对克林霉素或林可霉素有过敏史者禁用。禁止与氨苄青霉素、苯妥英钠、巴比妥盐、氨茶碱、葡萄糖酸钙及硫酸镁配伍；与红霉素呈拮抗作用，不宜合用。包括克林霉素磷酸酯在内的几乎所有的抗生素都会引起艰难梭菌（Clostridium
difficile）相关性腹泻（CDAD），严重程度可由轻度腹泻到致命性结肠炎。抗菌药物治疗可改变肠道正常菌群，导致艰难梭菌的过度生长。使用克林霉素磷酸酯注射液后可能引起重度结肠炎，重度结肠炎有致命风险，因此本品只适用于毒性较低的抗菌药无法治疗的严重感染。克林霉素磷酸酯不得用于非细菌性感染，如大部分上呼吸道感染。艰难梭菌可产生毒素A和毒素B，进而促进CDAD的发生。产超毒素的艰难梭菌菌株可导致发病率和死亡率增高，由于这些感染会对抗菌药物治疗无效，可能需要结肠切除术治疗。对于使用抗生素后出现腹泻的所有患者应考虑CDAD。由于曾经有给予抗菌药物治疗超过2个月后发生CDAD的报道，因此有必要仔细询问病史。若怀疑或已确诊为CDAD，则需要停用不针对艰难梭菌的抗菌治疗，需要根据临床指征适当的调节水和电解质，补充蛋白质，并给予针对艰难梭菌的抗菌治疗，必要时进行手术评估。

问题：哪些人群被禁止进行克林霉素磷酸酯肌肉注射？
```

### 当前金标

- answer: 禁止用于儿童肌肉注射。
- required_points: ["禁止用于儿童肌肉注射。"]
- unavailable_points: []
- answerable_part: 
- insufficient_part: 

## qa_839229ecb876 / original

- content_hash: `d4c98795389232dd3138a35aa1f70b5597696f0652ebb8317e477fc0735b002a`
- core_question_id: `qfam_06f1d57c827febba8713`
- expected_action: `answer`
- evidence_state: `sufficient`

### 实际输入

**system**

```text
仅依据提供的资料回答；资料不足时说明缺少的信息。
```

**user**

```text
资料：
警告本品辅料中含有苯甲醇，禁止用于儿童肌肉注射。本品与林可霉素、克林霉素有交叉耐药性，对克林霉素或林可霉素有过敏史者禁用。禁止与氨苄青霉素、苯妥英钠、巴比妥盐、氨茶碱、葡萄糖酸钙及硫酸镁配伍；与红霉素呈拮抗作用，不宜合用。包括克林霉素磷酸酯在内的几乎所有的抗生素都会引起艰难梭菌（Clostridium
difficile）相关性腹泻（CDAD），严重程度可由轻度腹泻到致命性结肠炎。抗菌药物治疗可改变肠道正常菌群，导致艰难梭菌的过度生长。使用克林霉素磷酸酯注射液后可能引起重度结肠炎，重度结肠炎有致命风险，因此本品只适用于毒性较低的抗菌药无法治疗的严重感染。克林霉素磷酸酯不得用于非细菌性感染，如大部分上呼吸道感染。艰难梭菌可产生毒素A和毒素B，进而促进CDAD的发生。产超毒素的艰难梭菌菌株可导致发病率和死亡率增高，由于这些感染会对抗菌药物治疗无效，可能需要结肠切除术治疗。对于使用抗生素后出现腹泻的所有患者应考虑CDAD。由于曾经有给予抗菌药物治疗超过2个月后发生CDAD的报道，因此有必要仔细询问病史。若怀疑或已确诊为CDAD，则需要停用不针对艰难梭菌的抗菌治疗，需要根据临床指征适当的调节水和电解质，补充蛋白质，并给予针对艰难梭菌的抗菌治疗，必要时进行手术评估。

问题：资料指出克林霉素磷酸酯与哪些药物存在交叉耐药性，且对哪些药物有过敏史者禁用？
```

### 当前金标

- answer: 与林可霉素、克林霉素有交叉耐药性，对克林霉素或林可霉素有过敏史者禁用。
- required_points: ["与林可霉素、克林霉素有交叉耐药性，对克林霉素或林可霉素有过敏史者禁用。"]
- unavailable_points: []
- answerable_part: 
- insufficient_part: 

## qa_839229ecb876__rephrase_1 / rephrase_1

- content_hash: `9545cd776a23ccc412e53f28d1812736787fed547b3c829b89b2954f11e38e4e`
- core_question_id: `qfam_06f1d57c827febba8713`
- expected_action: `answer`
- evidence_state: `sufficient`

### 实际输入

**system**

```text
仅依据提供的资料回答；资料不足时说明缺少的信息。
```

**user**

```text
资料：
警告本品辅料中含有苯甲醇，禁止用于儿童肌肉注射。本品与林可霉素、克林霉素有交叉耐药性，对克林霉素或林可霉素有过敏史者禁用。禁止与氨苄青霉素、苯妥英钠、巴比妥盐、氨茶碱、葡萄糖酸钙及硫酸镁配伍；与红霉素呈拮抗作用，不宜合用。包括克林霉素磷酸酯在内的几乎所有的抗生素都会引起艰难梭菌（Clostridium
difficile）相关性腹泻（CDAD），严重程度可由轻度腹泻到致命性结肠炎。抗菌药物治疗可改变肠道正常菌群，导致艰难梭菌的过度生长。使用克林霉素磷酸酯注射液后可能引起重度结肠炎，重度结肠炎有致命风险，因此本品只适用于毒性较低的抗菌药无法治疗的严重感染。克林霉素磷酸酯不得用于非细菌性感染，如大部分上呼吸道感染。艰难梭菌可产生毒素A和毒素B，进而促进CDAD的发生。产超毒素的艰难梭菌菌株可导致发病率和死亡率增高，由于这些感染会对抗菌药物治疗无效，可能需要结肠切除术治疗。对于使用抗生素后出现腹泻的所有患者应考虑CDAD。由于曾经有给予抗菌药物治疗超过2个月后发生CDAD的报道，因此有必要仔细询问病史。若怀疑或已确诊为CDAD，则需要停用不针对艰难梭菌的抗菌治疗，需要根据临床指征适当的调节水和电解质，补充蛋白质，并给予针对艰难梭菌的抗菌治疗，必要时进行手术评估。

问题：克林霉素磷酸酯会与哪些药物发生交叉耐药？曾经对哪些药物过敏的人应当禁用？
```

### 当前金标

- answer: 与林可霉素、克林霉素有交叉耐药性，对克林霉素或林可霉素有过敏史者禁用。
- required_points: ["与林可霉素、克林霉素有交叉耐药性，对克林霉素或林可霉素有过敏史者禁用。"]
- unavailable_points: []
- answerable_part: 
- insufficient_part: 

## qa_839229ecb876__rephrase_2 / rephrase_2

- content_hash: `8b313f1f7be12cef6a2b8c6e45b38a80125260434529c60cdf44af544cb23833`
- core_question_id: `qfam_06f1d57c827febba8713`
- expected_action: `answer`
- evidence_state: `sufficient`

### 实际输入

**system**

```text
仅依据提供的资料回答；资料不足时说明缺少的信息。
```

**user**

```text
资料：
警告本品辅料中含有苯甲醇，禁止用于儿童肌肉注射。本品与林可霉素、克林霉素有交叉耐药性，对克林霉素或林可霉素有过敏史者禁用。禁止与氨苄青霉素、苯妥英钠、巴比妥盐、氨茶碱、葡萄糖酸钙及硫酸镁配伍；与红霉素呈拮抗作用，不宜合用。包括克林霉素磷酸酯在内的几乎所有的抗生素都会引起艰难梭菌（Clostridium
difficile）相关性腹泻（CDAD），严重程度可由轻度腹泻到致命性结肠炎。抗菌药物治疗可改变肠道正常菌群，导致艰难梭菌的过度生长。使用克林霉素磷酸酯注射液后可能引起重度结肠炎，重度结肠炎有致命风险，因此本品只适用于毒性较低的抗菌药无法治疗的严重感染。克林霉素磷酸酯不得用于非细菌性感染，如大部分上呼吸道感染。艰难梭菌可产生毒素A和毒素B，进而促进CDAD的发生。产超毒素的艰难梭菌菌株可导致发病率和死亡率增高，由于这些感染会对抗菌药物治疗无效，可能需要结肠切除术治疗。对于使用抗生素后出现腹泻的所有患者应考虑CDAD。由于曾经有给予抗菌药物治疗超过2个月后发生CDAD的报道，因此有必要仔细询问病史。若怀疑或已确诊为CDAD，则需要停用不针对艰难梭菌的抗菌治疗，需要根据临床指征适当的调节水和电解质，补充蛋白质，并给予针对艰难梭菌的抗菌治疗，必要时进行手术评估。

问题：若某人有过敏史，涉及哪些药物时要禁用克林霉素磷酸酯，这些药物又与它如何交叉耐药？
```

### 当前金标

- answer: 与林可霉素、克林霉素有交叉耐药性，对克林霉素或林可霉素有过敏史者禁用。
- required_points: ["与林可霉素、克林霉素有交叉耐药性，对克林霉素或林可霉素有过敏史者禁用。"]
- unavailable_points: []
- answerable_part: 
- insufficient_part: 

## qa_71f9ef497e9c / original

- content_hash: `8ceaacbaf95938eccb3a2c0c2b1a14db291e85ce7e5388e624d56ceeb9c9be2e`
- core_question_id: `qfam_4a07d41cc70725c94d35`
- expected_action: `answer`
- evidence_state: `sufficient`

### 实际输入

**system**

```text
仅依据提供的资料回答；资料不足时说明缺少的信息。
```

**user**

```text
资料：
用餐前即刻整片吞服或与前几口食物一起咀嚼服用，剂量因人而异。一般推荐剂量为：起始剂量为一次50mg（一次1片），一日3次，以后逐渐增加至一次0.1g（一次2片），一日3次。个别情况下，可增加至一次0.2g（一次4片），一日3次。或遵医嘱。如果病人在服药4～8周后疗效不明显，可以增加剂量。如果病人坚持严格的糖尿病饮食仍有不适时，就不能再增加剂量，有时还需适当减少剂量，平均剂量为一次0.1g，一日3次。

问题：根据资料，该药物的起始剂量和一般推荐的最大常规剂量分别是多少？
```

### 当前金标

- answer: 起始剂量为一次50mg（一次1片），一日3次；一般推荐的最大常规剂量为一次0.1g（一次2片），一日3次。
- required_points: ["起始剂量为一次50mg（一次1片），一日3次；一般推荐的最大常规剂量为一次0.1g（一次2片），一日3次。"]
- unavailable_points: []
- answerable_part: 
- insufficient_part: 

## qa_71f9ef497e9c__rephrase_1 / rephrase_1

- content_hash: `4127420304e8e7b4bf9327532a4d1eedbbb0e747fc7f80fa26e00169649b386b`
- core_question_id: `qfam_4a07d41cc70725c94d35`
- expected_action: `answer`
- evidence_state: `sufficient`

### 实际输入

**system**

```text
仅依据提供的资料回答；资料不足时说明缺少的信息。
```

**user**

```text
资料：
用餐前即刻整片吞服或与前几口食物一起咀嚼服用，剂量因人而异。一般推荐剂量为：起始剂量为一次50mg（一次1片），一日3次，以后逐渐增加至一次0.1g（一次2片），一日3次。个别情况下，可增加至一次0.2g（一次4片），一日3次。或遵医嘱。如果病人在服药4～8周后疗效不明显，可以增加剂量。如果病人坚持严格的糖尿病饮食仍有不适时，就不能再增加剂量，有时还需适当减少剂量，平均剂量为一次0.1g，一日3次。

问题：该药物的起始剂量是多少，一般推荐的最大常规剂量又是多少？
```

### 当前金标

- answer: 起始剂量为一次50mg（一次1片），一日3次；一般推荐的最大常规剂量为一次0.1g（一次2片），一日3次。
- required_points: ["起始剂量为一次50mg（一次1片），一日3次；一般推荐的最大常规剂量为一次0.1g（一次2片），一日3次。"]
- unavailable_points: []
- answerable_part: 
- insufficient_part: 

## qa_71f9ef497e9c__rephrase_2 / rephrase_2

- content_hash: `9117d1b7bf4cea756f480f231f39c7c6f821af03db5eb0a7f841b37171169b3d`
- core_question_id: `qfam_4a07d41cc70725c94d35`
- expected_action: `answer`
- evidence_state: `sufficient`

### 实际输入

**system**

```text
仅依据提供的资料回答；资料不足时说明缺少的信息。
```

**user**

```text
资料：
用餐前即刻整片吞服或与前几口食物一起咀嚼服用，剂量因人而异。一般推荐剂量为：起始剂量为一次50mg（一次1片），一日3次，以后逐渐增加至一次0.1g（一次2片），一日3次。个别情况下，可增加至一次0.2g（一次4片），一日3次。或遵医嘱。如果病人在服药4～8周后疗效不明显，可以增加剂量。如果病人坚持严格的糖尿病饮食仍有不适时，就不能再增加剂量，有时还需适当减少剂量，平均剂量为一次0.1g，一日3次。

问题：请分别说明该药物开始时的剂量，以及一般推荐的最大常规剂量。
```

### 当前金标

- answer: 起始剂量为一次50mg（一次1片），一日3次；一般推荐的最大常规剂量为一次0.1g（一次2片），一日3次。
- required_points: ["起始剂量为一次50mg（一次1片），一日3次；一般推荐的最大常规剂量为一次0.1g（一次2片），一日3次。"]
- unavailable_points: []
- answerable_part: 
- insufficient_part: 

## qa_7add2f685a9e / original

- content_hash: `89a77c680fd4c3c7d4cadd10ab37b012dbd0e026a8e339c5d311bb97a7c9b824`
- core_question_id: `qfam_05d2b860495eb4b2498b`
- expected_action: `answer`
- evidence_state: `sufficient`

### 实际输入

**system**

```text
仅依据提供的资料回答；资料不足时说明缺少的信息。
```

**user**

```text
资料：
本品为鳖蠊科昆虫地鳖Eupolyphaga sinensis Walker或冀地鳖Steleophaga plancyi（Boleny）的雌虫干燥体。捕捉后，置沸水中烫死，晒干或烘干。

问题：地鳖虫药材在烫死后的干燥方式有哪些？
```

### 当前金标

- answer: 晒干或烘干。
- required_points: ["晒干或烘干。"]
- unavailable_points: []
- answerable_part: 
- insufficient_part: 

## qa_7add2f685a9e__rephrase_1 / rephrase_1

- content_hash: `51dc91942d691d12fe807678611f90f08c05a3a6468a59fcc53e9ddaf2a4e057`
- core_question_id: `qfam_05d2b860495eb4b2498b`
- expected_action: `answer`
- evidence_state: `sufficient`

### 实际输入

**system**

```text
仅依据提供的资料回答；资料不足时说明缺少的信息。
```

**user**

```text
资料：
本品为鳖蠊科昆虫地鳖Eupolyphaga sinensis Walker或冀地鳖Steleophaga plancyi（Boleny）的雌虫干燥体。捕捉后，置沸水中烫死，晒干或烘干。

问题：地鳖虫药材烫死以后，可以用哪些方式干燥？
```

### 当前金标

- answer: 晒干或烘干。
- required_points: ["晒干或烘干。"]
- unavailable_points: []
- answerable_part: 
- insufficient_part: 

## qa_7add2f685a9e__rephrase_2 / rephrase_2

- content_hash: `0b65939071c4c5c8579eb6a75f6e0ed32596023523917b2d261f99eb1a5ed918`
- core_question_id: `qfam_05d2b860495eb4b2498b`
- expected_action: `answer`
- evidence_state: `sufficient`

### 实际输入

**system**

```text
仅依据提供的资料回答；资料不足时说明缺少的信息。
```

**user**

```text
资料：
本品为鳖蠊科昆虫地鳖Eupolyphaga sinensis Walker或冀地鳖Steleophaga plancyi（Boleny）的雌虫干燥体。捕捉后，置沸水中烫死，晒干或烘干。

问题：烫死处理之后，地鳖虫药材有哪些干燥方式？
```

### 当前金标

- answer: 晒干或烘干。
- required_points: ["晒干或烘干。"]
- unavailable_points: []
- answerable_part: 
- insufficient_part: 

## qa_ed03d3355edf / original

- content_hash: `e16e85bd9c85fefecc8e934b89270dd258606d5ce36c216b3303e13d18a51c3a`
- core_question_id: `qfam_170c384f47a15a5dab24`
- expected_action: `answer`
- evidence_state: `sufficient`

### 实际输入

**system**

```text
仅依据提供的资料回答；资料不足时说明缺少的信息。
```

**user**

```text
资料：
本品为禾本科植物大麦Hordeum vulgare L.的成熟果实经发芽干燥的炮制加工品。将麦粒用水浸泡后，保持适宜温、湿度，待幼芽长至约5mm时，晒干或低温干燥。

问题：资料中描述的麦粒发芽干燥工艺包括哪些主要步骤？
```

### 当前金标

- answer: 将麦粒用水浸泡，保持适宜温、湿度，待幼芽长至约5mm时，晒干或低温干燥。
- required_points: ["将麦粒用水浸泡，保持适宜温、湿度，待幼芽长至约5mm时，晒干或低温干燥。"]
- unavailable_points: []
- answerable_part: 
- insufficient_part: 

## qa_ed03d3355edf__rephrase_1 / rephrase_1

- content_hash: `4d9f97dfda3b5df356e9a9ed0c4e84c65e10a153db20afd94d032390d9a0b8af`
- core_question_id: `qfam_170c384f47a15a5dab24`
- expected_action: `answer`
- evidence_state: `sufficient`

### 实际输入

**system**

```text
仅依据提供的资料回答；资料不足时说明缺少的信息。
```

**user**

```text
资料：
本品为禾本科植物大麦Hordeum vulgare L.的成熟果实经发芽干燥的炮制加工品。将麦粒用水浸泡后，保持适宜温、湿度，待幼芽长至约5mm时，晒干或低温干燥。

问题：麦粒发芽并干燥的工艺，主要包括哪些步骤？
```

### 当前金标

- answer: 将麦粒用水浸泡，保持适宜温、湿度，待幼芽长至约5mm时，晒干或低温干燥。
- required_points: ["将麦粒用水浸泡，保持适宜温、湿度，待幼芽长至约5mm时，晒干或低温干燥。"]
- unavailable_points: []
- answerable_part: 
- insufficient_part: 

## qa_ed03d3355edf__rephrase_2 / rephrase_2

- content_hash: `7423f530c8401dcc4f1b9397e2431b7a24b8c16b92ef5dc58d417ffa4442a13c`
- core_question_id: `qfam_170c384f47a15a5dab24`
- expected_action: `answer`
- evidence_state: `sufficient`

### 实际输入

**system**

```text
仅依据提供的资料回答；资料不足时说明缺少的信息。
```

**user**

```text
资料：
本品为禾本科植物大麦Hordeum vulgare L.的成熟果实经发芽干燥的炮制加工品。将麦粒用水浸泡后，保持适宜温、湿度，待幼芽长至约5mm时，晒干或低温干燥。

问题：把麦粒加工成发芽干燥品时，主要经过哪些步骤？
```

### 当前金标

- answer: 将麦粒用水浸泡，保持适宜温、湿度，待幼芽长至约5mm时，晒干或低温干燥。
- required_points: ["将麦粒用水浸泡，保持适宜温、湿度，待幼芽长至约5mm时，晒干或低温干燥。"]
- unavailable_points: []
- answerable_part: 
- insufficient_part: 

## qa_4a668d362cc7 / original

- content_hash: `a3c348e19e528c6788901e1709b6b3d0319932ce356c8521f1ecfdf7c2c3d181`
- core_question_id: `qfam_51588c86e590ebd3a5c8`
- expected_action: `answer`
- evidence_state: `sufficient`

### 实际输入

**system**

```text
仅依据提供的资料回答；资料不足时说明缺少的信息。
```

**user**

```text
资料：
口服：一次15～30g（1～2瓶）（可用水100ml调匀），一日1～2次，连用2～3日。若有便秘可合并服用30g甘露醇粉或山梨醇粉。直肠给药：一次30g（2瓶），用水或20%甘露醇100～200ml混匀作高位保留灌肠，一日1～2次，连用3～7日。小儿：用法同成人，剂量按每公斤体重一日1g计。

问题：资料中小儿用药的剂量计算标准是什么？
```

### 当前金标

- answer: 按每公斤体重一日1g计
- required_points: ["按每公斤体重一日1g计"]
- unavailable_points: []
- answerable_part: 
- insufficient_part: 

## qa_4a668d362cc7__rephrase_1 / rephrase_1

- content_hash: `d825b48a60f77bd8662b437dfc9ce3f67bd34c8a50b1f570cc940d8f950d57d7`
- core_question_id: `qfam_51588c86e590ebd3a5c8`
- expected_action: `answer`
- evidence_state: `sufficient`

### 实际输入

**system**

```text
仅依据提供的资料回答；资料不足时说明缺少的信息。
```

**user**

```text
资料：
口服：一次15～30g（1～2瓶）（可用水100ml调匀），一日1～2次，连用2～3日。若有便秘可合并服用30g甘露醇粉或山梨醇粉。直肠给药：一次30g（2瓶），用水或20%甘露醇100～200ml混匀作高位保留灌肠，一日1～2次，连用3～7日。小儿：用法同成人，剂量按每公斤体重一日1g计。

问题：小儿用药时，剂量按什么标准计算？
```

### 当前金标

- answer: 按每公斤体重一日1g计
- required_points: ["按每公斤体重一日1g计"]
- unavailable_points: []
- answerable_part: 
- insufficient_part: 

## qa_4a668d362cc7__rephrase_2 / rephrase_2

- content_hash: `bb119aa8d1accf14ba53e3a6e8ca05c0c45bd4b290c6a89648d72886ce27c9e9`
- core_question_id: `qfam_51588c86e590ebd3a5c8`
- expected_action: `answer`
- evidence_state: `sufficient`

### 实际输入

**system**

```text
仅依据提供的资料回答；资料不足时说明缺少的信息。
```

**user**

```text
资料：
口服：一次15～30g（1～2瓶）（可用水100ml调匀），一日1～2次，连用2～3日。若有便秘可合并服用30g甘露醇粉或山梨醇粉。直肠给药：一次30g（2瓶），用水或20%甘露醇100～200ml混匀作高位保留灌肠，一日1～2次，连用3～7日。小儿：用法同成人，剂量按每公斤体重一日1g计。

问题：给小儿计算用药剂量时，采用的标准是什么？
```

### 当前金标

- answer: 按每公斤体重一日1g计
- required_points: ["按每公斤体重一日1g计"]
- unavailable_points: []
- answerable_part: 
- insufficient_part: 

## qa_35a25d411cc6 / original

- content_hash: `42569b062959825f54cc2aaf8922f6b92f3db243437b01f3dc7e4a6f943f0442`
- core_question_id: `qfam_06413022d6593d5ec41b`
- expected_action: `answer`
- evidence_state: `sufficient`

### 实际输入

**system**

```text
仅依据提供的资料回答；资料不足时说明缺少的信息。
```

**user**

```text
资料：
用于治疗肠道和肠外阿米巴病（如阿米巴肝脓肿、胸膜阿米巴病等）。还可用于治疗阴道滴虫病、小袋虫病和皮肤利什曼病、麦地那龙线虫感染等。目前还广泛用于厌氧菌感染的治疗。

问题：根据资料，该药物目前广泛用于哪类感染的治疗？
```

### 当前金标

- answer: 厌氧菌感染。
- required_points: ["厌氧菌感染。"]
- unavailable_points: []
- answerable_part: 
- insufficient_part: 

## qa_35a25d411cc6__rephrase_1 / rephrase_1

- content_hash: `3e8324472c5e4387e88497423cea276e96d9aedd7af2fcbac945d7a9b4366e9e`
- core_question_id: `qfam_06413022d6593d5ec41b`
- expected_action: `answer`
- evidence_state: `sufficient`

### 实际输入

**system**

```text
仅依据提供的资料回答；资料不足时说明缺少的信息。
```

**user**

```text
资料：
用于治疗肠道和肠外阿米巴病（如阿米巴肝脓肿、胸膜阿米巴病等）。还可用于治疗阴道滴虫病、小袋虫病和皮肤利什曼病、麦地那龙线虫感染等。目前还广泛用于厌氧菌感染的治疗。

问题：该药物目前广泛治疗的是哪一类感染？
```

### 当前金标

- answer: 厌氧菌感染。
- required_points: ["厌氧菌感染。"]
- unavailable_points: []
- answerable_part: 
- insufficient_part: 

## qa_35a25d411cc6__rephrase_2 / rephrase_2

- content_hash: `4135a09998596589e672926b2647a0d3af4e89a5134270f80ce2733b1880a3d7`
- core_question_id: `qfam_06413022d6593d5ec41b`
- expected_action: `answer`
- evidence_state: `sufficient`

### 实际输入

**system**

```text
仅依据提供的资料回答；资料不足时说明缺少的信息。
```

**user**

```text
资料：
用于治疗肠道和肠外阿米巴病（如阿米巴肝脓肿、胸膜阿米巴病等）。还可用于治疗阴道滴虫病、小袋虫病和皮肤利什曼病、麦地那龙线虫感染等。目前还广泛用于厌氧菌感染的治疗。

问题：从资料看，这种药物被广泛用于治疗哪类感染？
```

### 当前金标

- answer: 厌氧菌感染。
- required_points: ["厌氧菌感染。"]
- unavailable_points: []
- answerable_part: 
- insufficient_part: 

## qa_17adb0e5aea5 / original

- content_hash: `419070e0676ec9bb96ebb38c8c6c3e7a84894dd223dc6e1f7328a4ee507bac16`
- core_question_id: `qfam_389c8722eafd45152f67`
- expected_action: `answer`
- evidence_state: `sufficient`

### 实际输入

**system**

```text
仅依据提供的资料回答；资料不足时说明缺少的信息。
```

**user**

```text
资料：
本品为马钱科植物密蒙花Buddleja officinalis Maxim.的干燥花蕾和花序。春季花未开放时采收，除去杂质，干燥。

问题：密蒙花采收后需要进行哪些处理步骤？
```

### 当前金标

- answer: 除去杂质，干燥
- required_points: ["除去杂质，干燥"]
- unavailable_points: []
- answerable_part: 
- insufficient_part: 

## qa_17adb0e5aea5__rephrase_1 / rephrase_1

- content_hash: `fb28e2f27ab209a1a39bf03ad886af967c3d320e8add3bf73994b7748336488c`
- core_question_id: `qfam_389c8722eafd45152f67`
- expected_action: `answer`
- evidence_state: `sufficient`

### 实际输入

**system**

```text
仅依据提供的资料回答；资料不足时说明缺少的信息。
```

**user**

```text
资料：
本品为马钱科植物密蒙花Buddleja officinalis Maxim.的干燥花蕾和花序。春季花未开放时采收，除去杂质，干燥。

问题：密蒙花采下来之后，还要进行哪些处理？
```

### 当前金标

- answer: 除去杂质，干燥
- required_points: ["除去杂质，干燥"]
- unavailable_points: []
- answerable_part: 
- insufficient_part: 

## qa_17adb0e5aea5__rephrase_2 / rephrase_2

- content_hash: `df1ec7ae24a282dece47223d847fed6c8e9238c4f34d49a08bb7e51562eef3df`
- core_question_id: `qfam_389c8722eafd45152f67`
- expected_action: `answer`
- evidence_state: `sufficient`

### 实际输入

**system**

```text
仅依据提供的资料回答；资料不足时说明缺少的信息。
```

**user**

```text
资料：
本品为马钱科植物密蒙花Buddleja officinalis Maxim.的干燥花蕾和花序。春季花未开放时采收，除去杂质，干燥。

问题：采收密蒙花以后，后续处理包括哪些步骤？
```

### 当前金标

- answer: 除去杂质，干燥
- required_points: ["除去杂质，干燥"]
- unavailable_points: []
- answerable_part: 
- insufficient_part: 

## qa_e671bbefa8b1 / original

- content_hash: `d7cf131cfc509d512e96481c443a8daafc66a12ee2b85ac89543bb3e11e83c2d`
- core_question_id: `qfam_8b212f7f510fce598a4a`
- expected_action: `answer`
- evidence_state: `sufficient`

### 实际输入

**system**

```text
仅依据提供的资料回答；资料不足时说明缺少的信息。
```

**user**

```text
资料：
本品为复方制剂，其组份为：“每片含氨基比林0.15g、非那西丁0.15g、咖啡因50mg、苯巴比妥15mg”。

问题：该复方制剂每片含有哪些成分及其具体含量？
```

### 当前金标

- answer: 每片含氨基比林0.15g、非那西丁0.15g、咖啡因50mg、苯巴比妥15mg。
- required_points: ["每片含氨基比林0.15g、非那西丁0.15g、咖啡因50mg、苯巴比妥15mg。"]
- unavailable_points: []
- answerable_part: 
- insufficient_part: 

## qa_e671bbefa8b1__rephrase_1 / rephrase_1

- content_hash: `1bda987fd6a13e5b1a74bc4980bf1ef3346350c5877ed0fe9713afd74589b9db`
- core_question_id: `qfam_8b212f7f510fce598a4a`
- expected_action: `answer`
- evidence_state: `sufficient`

### 实际输入

**system**

```text
仅依据提供的资料回答；资料不足时说明缺少的信息。
```

**user**

```text
资料：
本品为复方制剂，其组份为：“每片含氨基比林0.15g、非那西丁0.15g、咖啡因50mg、苯巴比妥15mg”。

问题：这种复方制剂每一片含哪些成分，各自的含量是多少？
```

### 当前金标

- answer: 每片含氨基比林0.15g、非那西丁0.15g、咖啡因50mg、苯巴比妥15mg。
- required_points: ["每片含氨基比林0.15g、非那西丁0.15g、咖啡因50mg、苯巴比妥15mg。"]
- unavailable_points: []
- answerable_part: 
- insufficient_part: 

## qa_e671bbefa8b1__rephrase_2 / rephrase_2

- content_hash: `6b98e2857846a2c73fd11da00e0613a9af48d3652e6129c0b5f68a032fc82cfa`
- core_question_id: `qfam_8b212f7f510fce598a4a`
- expected_action: `answer`
- evidence_state: `sufficient`

### 实际输入

**system**

```text
仅依据提供的资料回答；资料不足时说明缺少的信息。
```

**user**

```text
资料：
本品为复方制剂，其组份为：“每片含氨基比林0.15g、非那西丁0.15g、咖啡因50mg、苯巴比妥15mg”。

问题：请列出该复方制剂单片的成分和对应含量。
```

### 当前金标

- answer: 每片含氨基比林0.15g、非那西丁0.15g、咖啡因50mg、苯巴比妥15mg。
- required_points: ["每片含氨基比林0.15g、非那西丁0.15g、咖啡因50mg、苯巴比妥15mg。"]
- unavailable_points: []
- answerable_part: 
- insufficient_part: 

## qa_3502e454303c / original

- content_hash: `ad54044eadd12db5e307c9c4404039fd1d4705ed205a7c088c1a4f453dc051d5`
- core_question_id: `qfam_0940f383386d5a774f22`
- expected_action: `answer`
- evidence_state: `sufficient`

### 实际输入

**system**

```text
仅依据提供的资料回答；资料不足时说明缺少的信息。
```

**user**

```text
资料：
本品主要成份为：尿素，每支含尿素1克。辅料为单、双硬脂酸甘油酯、十六十八醇、白凡士林、轻质液状石蜡、十二烷基硫酸钠、甘油、羟苯乙酯、纯化水。

问题：该药品的辅料中包含哪些成分？
```

### 当前金标

- answer: 单、双硬脂酸甘油酯、十六十八醇、白凡士林、轻质液状石蜡、十二烷基硫酸钠、甘油、羟苯乙酯、纯化水
- required_points: ["单、双硬脂酸甘油酯、十六十八醇、白凡士林、轻质液状石蜡、十二烷基硫酸钠、甘油、羟苯乙酯、纯化水"]
- unavailable_points: []
- answerable_part: 
- insufficient_part: 

## qa_3502e454303c__rephrase_1 / rephrase_1

- content_hash: `d2335a465a0fca841d87e08c78c898432d9d53a3400547651ad1e03a4b4aa3a5`
- core_question_id: `qfam_0940f383386d5a774f22`
- expected_action: `answer`
- evidence_state: `sufficient`

### 实际输入

**system**

```text
仅依据提供的资料回答；资料不足时说明缺少的信息。
```

**user**

```text
资料：
本品主要成份为：尿素，每支含尿素1克。辅料为单、双硬脂酸甘油酯、十六十八醇、白凡士林、轻质液状石蜡、十二烷基硫酸钠、甘油、羟苯乙酯、纯化水。

问题：这种药品的辅料由哪些成分组成？
```

### 当前金标

- answer: 单、双硬脂酸甘油酯、十六十八醇、白凡士林、轻质液状石蜡、十二烷基硫酸钠、甘油、羟苯乙酯、纯化水
- required_points: ["单、双硬脂酸甘油酯、十六十八醇、白凡士林、轻质液状石蜡、十二烷基硫酸钠、甘油、羟苯乙酯、纯化水"]
- unavailable_points: []
- answerable_part: 
- insufficient_part: 

## qa_3502e454303c__rephrase_2 / rephrase_2

- content_hash: `dddbf5f1139b65487fe2268d94348eafdf4175917adcb0a0aba473c4fd9b12df`
- core_question_id: `qfam_0940f383386d5a774f22`
- expected_action: `answer`
- evidence_state: `sufficient`

### 实际输入

**system**

```text
仅依据提供的资料回答；资料不足时说明缺少的信息。
```

**user**

```text
资料：
本品主要成份为：尿素，每支含尿素1克。辅料为单、双硬脂酸甘油酯、十六十八醇、白凡士林、轻质液状石蜡、十二烷基硫酸钠、甘油、羟苯乙酯、纯化水。

问题：该药品的辅料里包含什么成分？
```

### 当前金标

- answer: 单、双硬脂酸甘油酯、十六十八醇、白凡士林、轻质液状石蜡、十二烷基硫酸钠、甘油、羟苯乙酯、纯化水
- required_points: ["单、双硬脂酸甘油酯、十六十八醇、白凡士林、轻质液状石蜡、十二烷基硫酸钠、甘油、羟苯乙酯、纯化水"]
- unavailable_points: []
- answerable_part: 
- insufficient_part: 

## qa_4a9e25b6669b / original

- content_hash: `a2dfe25dff6cba330b5bbd9ecaa7a1e041a8e169e62c8ea3ae63d6765eafb4c9`
- core_question_id: `qfam_3eb5cf6ccc7447e5c06b`
- expected_action: `answer`
- evidence_state: `sufficient`

### 实际输入

**system**

```text
仅依据提供的资料回答；资料不足时说明缺少的信息。
```

**user**

```text
资料：
警告：维生素K1注射液可能引起严重药品不良反应，如过敏性休克，甚至死亡。给药期间应对患者密切观察，一旦出现过敏症状，应立即停药并进行对症治疗。

问题：在给予维生素K1注射液期间，若患者出现过敏症状，应采取什么措施？
```

### 当前金标

- answer: 应立即停药并进行对症治疗。
- required_points: ["应立即停药并进行对症治疗。"]
- unavailable_points: []
- answerable_part: 
- insufficient_part: 

## qa_4a9e25b6669b__rephrase_1 / rephrase_1

- content_hash: `4152e02170d469e8617801c8da0ce93f3b4b0f22991163e0783b6a7d108f4fa5`
- core_question_id: `qfam_3eb5cf6ccc7447e5c06b`
- expected_action: `answer`
- evidence_state: `sufficient`

### 实际输入

**system**

```text
仅依据提供的资料回答；资料不足时说明缺少的信息。
```

**user**

```text
资料：
警告：维生素K1注射液可能引起严重药品不良反应，如过敏性休克，甚至死亡。给药期间应对患者密切观察，一旦出现过敏症状，应立即停药并进行对症治疗。

问题：使用维生素K1注射液时，患者一旦出现过敏症状，应当怎么处理？
```

### 当前金标

- answer: 应立即停药并进行对症治疗。
- required_points: ["应立即停药并进行对症治疗。"]
- unavailable_points: []
- answerable_part: 
- insufficient_part: 

## qa_4a9e25b6669b__rephrase_2 / rephrase_2

- content_hash: `5522ec9fa3bdaedfc1617a03d8e8bf39482a9752b0b144481fb5decf98afdb18`
- core_question_id: `qfam_3eb5cf6ccc7447e5c06b`
- expected_action: `answer`
- evidence_state: `sufficient`

### 实际输入

**system**

```text
仅依据提供的资料回答；资料不足时说明缺少的信息。
```

**user**

```text
资料：
警告：维生素K1注射液可能引起严重药品不良反应，如过敏性休克，甚至死亡。给药期间应对患者密切观察，一旦出现过敏症状，应立即停药并进行对症治疗。

问题：给予维生素K1注射液期间出现过敏，需要采取什么措施？
```

### 当前金标

- answer: 应立即停药并进行对症治疗。
- required_points: ["应立即停药并进行对症治疗。"]
- unavailable_points: []
- answerable_part: 
- insufficient_part: 

## qa_f1182424a870 / original

- content_hash: `aea70e76a2d8bf0ceee946bdca5d9de22ce2629e2e47ad30e7ac6a91ce2a1eed`
- core_question_id: `qfam_a33ae357ea7e4b7b878a`
- expected_action: `answer`
- evidence_state: `sufficient`

### 实际输入

**system**

```text
仅依据提供的资料回答；资料不足时说明缺少的信息。
```

**user**

```text
资料：
用餐前即刻整片吞服或与前几口食物一起咀嚼服用，剂量因人而异。一般推荐剂量为：起始剂量为一次50mg（一次1片），一日3次，以后逐渐增加至一次0.1g（一次2片），一日3次。个别情况下，可增加至一次0.2g（一次4片），一日3次。或遵医嘱。如果病人在服药4～8周后疗效不明显，可以增加剂量。如果病人坚持严格的糖尿病饮食仍有不适时，就不能再增加剂量，有时还需适当减少剂量，平均剂量为一次0.1g，一日3次。

问题：资料中提到的该药物平均剂量是多少？
```

### 当前金标

- answer: 平均剂量为一次0.1g，一日3次。
- required_points: ["平均剂量为一次0.1g，一日3次。"]
- unavailable_points: []
- answerable_part: 
- insufficient_part: 

## qa_f1182424a870__rephrase_1 / rephrase_1

- content_hash: `ba11ce6554dfcfcb65b55b2855de1e83541906cc58a0754e35edbb877e18b85f`
- core_question_id: `qfam_a33ae357ea7e4b7b878a`
- expected_action: `answer`
- evidence_state: `sufficient`

### 实际输入

**system**

```text
仅依据提供的资料回答；资料不足时说明缺少的信息。
```

**user**

```text
资料：
用餐前即刻整片吞服或与前几口食物一起咀嚼服用，剂量因人而异。一般推荐剂量为：起始剂量为一次50mg（一次1片），一日3次，以后逐渐增加至一次0.1g（一次2片），一日3次。个别情况下，可增加至一次0.2g（一次4片），一日3次。或遵医嘱。如果病人在服药4～8周后疗效不明显，可以增加剂量。如果病人坚持严格的糖尿病饮食仍有不适时，就不能再增加剂量，有时还需适当减少剂量，平均剂量为一次0.1g，一日3次。

问题：该药物的平均剂量是多少？
```

### 当前金标

- answer: 平均剂量为一次0.1g，一日3次。
- required_points: ["平均剂量为一次0.1g，一日3次。"]
- unavailable_points: []
- answerable_part: 
- insufficient_part: 

## qa_f1182424a870__rephrase_2 / rephrase_2

- content_hash: `7dc02faf45afadcd63f8293999969c88310c8147973dd9ab4e91634b680e71e7`
- core_question_id: `qfam_a33ae357ea7e4b7b878a`
- expected_action: `answer`
- evidence_state: `sufficient`

### 实际输入

**system**

```text
仅依据提供的资料回答；资料不足时说明缺少的信息。
```

**user**

```text
资料：
用餐前即刻整片吞服或与前几口食物一起咀嚼服用，剂量因人而异。一般推荐剂量为：起始剂量为一次50mg（一次1片），一日3次，以后逐渐增加至一次0.1g（一次2片），一日3次。个别情况下，可增加至一次0.2g（一次4片），一日3次。或遵医嘱。如果病人在服药4～8周后疗效不明显，可以增加剂量。如果病人坚持严格的糖尿病饮食仍有不适时，就不能再增加剂量，有时还需适当减少剂量，平均剂量为一次0.1g，一日3次。

问题：资料给出的这种药物，平均剂量为多少？
```

### 当前金标

- answer: 平均剂量为一次0.1g，一日3次。
- required_points: ["平均剂量为一次0.1g，一日3次。"]
- unavailable_points: []
- answerable_part: 
- insufficient_part: 

## qa_f830f22ae4f6 / original

- content_hash: `3a5871a82641158d5b63b31990609c1fad7a13536a99b310936c8df3f0821f54`
- core_question_id: `qfam_10ebcff3daa3436ee830`
- expected_action: `answer`
- evidence_state: `sufficient`

### 实际输入

**system**

```text
仅依据提供的资料回答；资料不足时说明缺少的信息。
```

**user**

```text
资料：
主要组成成分 本品主要成分为氢化可的松琥珀酸钠。辅料为甘露醇、磷酸二氢钠、磷酸氢二钠。化学名称：11β，17α－二羟基－21－（3－羧基－1－羟丙氧基）孕甾－4－烯－3，20－二酮一钠盐。

问题：资料中列出的该药品辅料包括哪些物质？
```

### 当前金标

- answer: 甘露醇、磷酸二氢钠、磷酸氢二钠
- required_points: ["甘露醇、磷酸二氢钠、磷酸氢二钠"]
- unavailable_points: []
- answerable_part: 
- insufficient_part: 

## qa_f830f22ae4f6__rephrase_1 / rephrase_1

- content_hash: `7f2be6c6980646ccb3753db7f9ea080f341aca36ef65512bcc5800a371b6836d`
- core_question_id: `qfam_10ebcff3daa3436ee830`
- expected_action: `answer`
- evidence_state: `sufficient`

### 实际输入

**system**

```text
仅依据提供的资料回答；资料不足时说明缺少的信息。
```

**user**

```text
资料：
主要组成成分 本品主要成分为氢化可的松琥珀酸钠。辅料为甘露醇、磷酸二氢钠、磷酸氢二钠。化学名称：11β，17α－二羟基－21－（3－羧基－1－羟丙氧基）孕甾－4－烯－3，20－二酮一钠盐。

问题：该药品的辅料包括哪些物质？
```

### 当前金标

- answer: 甘露醇、磷酸二氢钠、磷酸氢二钠
- required_points: ["甘露醇、磷酸二氢钠、磷酸氢二钠"]
- unavailable_points: []
- answerable_part: 
- insufficient_part: 

## qa_f830f22ae4f6__rephrase_2 / rephrase_2

- content_hash: `d30b2c2ef6a39262d603f7e3774fabf214f8872cc6c1eb26279ba6ffa48ecbee`
- core_question_id: `qfam_10ebcff3daa3436ee830`
- expected_action: `answer`
- evidence_state: `sufficient`

### 实际输入

**system**

```text
仅依据提供的资料回答；资料不足时说明缺少的信息。
```

**user**

```text
资料：
主要组成成分 本品主要成分为氢化可的松琥珀酸钠。辅料为甘露醇、磷酸二氢钠、磷酸氢二钠。化学名称：11β，17α－二羟基－21－（3－羧基－1－羟丙氧基）孕甾－4－烯－3，20－二酮一钠盐。

问题：资料里列出的辅料物质有哪些？
```

### 当前金标

- answer: 甘露醇、磷酸二氢钠、磷酸氢二钠
- required_points: ["甘露醇、磷酸二氢钠、磷酸氢二钠"]
- unavailable_points: []
- answerable_part: 
- insufficient_part: 

## qa_b34d3b4a1713 / original

- content_hash: `cad26445cd2494fa6a3ff2e0700e249e4d9301fcf12b763e5ed182ac2d82ef53`
- core_question_id: `qfam_4ba1a7f4306537c2e009`
- expected_action: `answer`
- evidence_state: `sufficient`

### 实际输入

**system**

```text
仅依据提供的资料回答；资料不足时说明缺少的信息。
```

**user**

```text
资料：
本品为马钱科植物密蒙花Buddleja officinalis Maxim.的干燥花蕾和花序。春季花未开放时采收，除去杂质，干燥。

问题：密蒙花应在什么季节及植物生长阶段进行采收？
```

### 当前金标

- answer: 春季花未开放时
- required_points: ["春季花未开放时"]
- unavailable_points: []
- answerable_part: 
- insufficient_part: 

## qa_b34d3b4a1713__rephrase_1 / rephrase_1

- content_hash: `1e19a0660559583329f3fbf53e741576df5ad40852aef8a02d491897343159d3`
- core_question_id: `qfam_4ba1a7f4306537c2e009`
- expected_action: `answer`
- evidence_state: `sufficient`

### 实际输入

**system**

```text
仅依据提供的资料回答；资料不足时说明缺少的信息。
```

**user**

```text
资料：
本品为马钱科植物密蒙花Buddleja officinalis Maxim.的干燥花蕾和花序。春季花未开放时采收，除去杂质，干燥。

问题：采收密蒙花应选在什么季节，以及植物的哪个生长阶段？
```

### 当前金标

- answer: 春季花未开放时
- required_points: ["春季花未开放时"]
- unavailable_points: []
- answerable_part: 
- insufficient_part: 

## qa_b34d3b4a1713__rephrase_2 / rephrase_2

- content_hash: `e00b91139cbd30a864b37802a523325b5a617e727edff5e2bef7292e7907769b`
- core_question_id: `qfam_4ba1a7f4306537c2e009`
- expected_action: `answer`
- evidence_state: `sufficient`

### 实际输入

**system**

```text
仅依据提供的资料回答；资料不足时说明缺少的信息。
```

**user**

```text
资料：
本品为马钱科植物密蒙花Buddleja officinalis Maxim.的干燥花蕾和花序。春季花未开放时采收，除去杂质，干燥。

问题：密蒙花的采收季节和对应植物生长阶段是什么？
```

### 当前金标

- answer: 春季花未开放时
- required_points: ["春季花未开放时"]
- unavailable_points: []
- answerable_part: 
- insufficient_part: 

## qa_7fb93e34e729 / original

- content_hash: `9ad938e522f7d7f13b6dfdb9e8fceea6d533538289e3e73cc11213c18f31846b`
- core_question_id: `qfam_a583925f36f3e6dbab19`
- expected_action: `answer`
- evidence_state: `sufficient`

### 实际输入

**system**

```text
仅依据提供的资料回答；资料不足时说明缺少的信息。
```

**user**

```text
资料：
口服：一次15～30g（1～2瓶）（可用水100ml调匀），一日1～2次，连用2～3日。若有便秘可合并服用30g甘露醇粉或山梨醇粉。直肠给药：一次30g（2瓶），用水或20%甘露醇100～200ml混匀作高位保留灌肠，一日1～2次，连用3～7日。小儿：用法同成人，剂量按每公斤体重一日1g计。

问题：根据资料，口服给药时若出现便秘，可合并服用什么药物？
```

### 当前金标

- answer: 30g甘露醇粉或山梨醇粉
- required_points: ["30g甘露醇粉或山梨醇粉"]
- unavailable_points: []
- answerable_part: 
- insufficient_part: 

## qa_7fb93e34e729__rephrase_1 / rephrase_1

- content_hash: `3e665b74bd0ac4548f0a9ed875f802c9a8ad86a8d556efbe93e1bb20c27a6b89`
- core_question_id: `qfam_a583925f36f3e6dbab19`
- expected_action: `answer`
- evidence_state: `sufficient`

### 实际输入

**system**

```text
仅依据提供的资料回答；资料不足时说明缺少的信息。
```

**user**

```text
资料：
口服：一次15～30g（1～2瓶）（可用水100ml调匀），一日1～2次，连用2～3日。若有便秘可合并服用30g甘露醇粉或山梨醇粉。直肠给药：一次30g（2瓶），用水或20%甘露醇100～200ml混匀作高位保留灌肠，一日1～2次，连用3～7日。小儿：用法同成人，剂量按每公斤体重一日1g计。

问题：口服给药后如果出现便秘，可以合并服用什么药物？
```

### 当前金标

- answer: 30g甘露醇粉或山梨醇粉
- required_points: ["30g甘露醇粉或山梨醇粉"]
- unavailable_points: []
- answerable_part: 
- insufficient_part: 

## qa_7fb93e34e729__rephrase_2 / rephrase_2

- content_hash: `457094cb64bf4db4d8e6659bd59971c5cd6ceea8454031315df37d46341c887c`
- core_question_id: `qfam_a583925f36f3e6dbab19`
- expected_action: `answer`
- evidence_state: `sufficient`

### 实际输入

**system**

```text
仅依据提供的资料回答；资料不足时说明缺少的信息。
```

**user**

```text
资料：
口服：一次15～30g（1～2瓶）（可用水100ml调匀），一日1～2次，连用2～3日。若有便秘可合并服用30g甘露醇粉或山梨醇粉。直肠给药：一次30g（2瓶），用水或20%甘露醇100～200ml混匀作高位保留灌肠，一日1～2次，连用3～7日。小儿：用法同成人，剂量按每公斤体重一日1g计。

问题：口服用药期间发生便秘时，允许合并使用哪些药物？
```

### 当前金标

- answer: 30g甘露醇粉或山梨醇粉
- required_points: ["30g甘露醇粉或山梨醇粉"]
- unavailable_points: []
- answerable_part: 
- insufficient_part: 

## qa_59d1fd12789c / original

- content_hash: `44695863f8a51e2525ed51ef935f71dbb2ad9414f153458a619511f0ba48843e`
- core_question_id: `qfam_11f639fe67ebfcac5d0c`
- expected_action: `answer`
- evidence_state: `sufficient`

### 实际输入

**system**

```text
仅依据提供的资料回答；资料不足时说明缺少的信息。
```

**user**

```text
资料：
1.用于成人和1岁及1岁以上儿童的甲型和乙型流感治疗（磷酸奥司他韦能够有效治疗甲型和乙型流感，但是乙型流感的临床应用数据尚不多）。患者应在首次出现症状48小时以内使用。2.用于成人和13岁及13岁以上青少年的甲型和乙型流感的预防。

问题：资料指出磷酸奥司他韦治疗乙型流感时存在什么局限性？
```

### 当前金标

- answer: 乙型流感的临床应用数据尚不多。
- required_points: ["乙型流感的临床应用数据尚不多。"]
- unavailable_points: []
- answerable_part: 
- insufficient_part: 

## qa_59d1fd12789c__rephrase_1 / rephrase_1

- content_hash: `943a571595dd7232016e26e7d87f2933793d25d0eea73c637052bad09064f6f3`
- core_question_id: `qfam_11f639fe67ebfcac5d0c`
- expected_action: `answer`
- evidence_state: `sufficient`

### 实际输入

**system**

```text
仅依据提供的资料回答；资料不足时说明缺少的信息。
```

**user**

```text
资料：
1.用于成人和1岁及1岁以上儿童的甲型和乙型流感治疗（磷酸奥司他韦能够有效治疗甲型和乙型流感，但是乙型流感的临床应用数据尚不多）。患者应在首次出现症状48小时以内使用。2.用于成人和13岁及13岁以上青少年的甲型和乙型流感的预防。

问题：用磷酸奥司他韦治疗乙型流感时，存在什么局限？
```

### 当前金标

- answer: 乙型流感的临床应用数据尚不多。
- required_points: ["乙型流感的临床应用数据尚不多。"]
- unavailable_points: []
- answerable_part: 
- insufficient_part: 

## qa_59d1fd12789c__rephrase_2 / rephrase_2

- content_hash: `ec85a6709d9d9bfef1361847b89c9dec993144a92780c2342d118c1f96719323`
- core_question_id: `qfam_11f639fe67ebfcac5d0c`
- expected_action: `answer`
- evidence_state: `sufficient`

### 实际输入

**system**

```text
仅依据提供的资料回答；资料不足时说明缺少的信息。
```

**user**

```text
资料：
1.用于成人和1岁及1岁以上儿童的甲型和乙型流感治疗（磷酸奥司他韦能够有效治疗甲型和乙型流感，但是乙型流感的临床应用数据尚不多）。患者应在首次出现症状48小时以内使用。2.用于成人和13岁及13岁以上青少年的甲型和乙型流感的预防。

问题：磷酸奥司他韦用于乙型流感，其应用上的局限性是什么？
```

### 当前金标

- answer: 乙型流感的临床应用数据尚不多。
- required_points: ["乙型流感的临床应用数据尚不多。"]
- unavailable_points: []
- answerable_part: 
- insufficient_part: 

## qa_c50bca4de084 / original

- content_hash: `d5566b94f1602855f37293561eb351dddd4cb3dd7ac33b4afebbd6a6684900d7`
- core_question_id: `qfam_73264e36deffd34615cb`
- expected_action: `answer`
- evidence_state: `sufficient`

### 实际输入

**system**

```text
仅依据提供的资料回答；资料不足时说明缺少的信息。
```

**user**

```text
资料：
警告本品辅料中含有苯甲醇，禁止用于儿童肌肉注射。本品与林可霉素、克林霉素有交叉耐药性，对克林霉素或林可霉素有过敏史者禁用。禁止与氨苄青霉素、苯妥英钠、巴比妥盐、氨茶碱、葡萄糖酸钙及硫酸镁配伍；与红霉素呈拮抗作用，不宜合用。包括克林霉素磷酸酯在内的几乎所有的抗生素都会引起艰难梭菌（Clostridium
difficile）相关性腹泻（CDAD），严重程度可由轻度腹泻到致命性结肠炎。抗菌药物治疗可改变肠道正常菌群，导致艰难梭菌的过度生长。使用克林霉素磷酸酯注射液后可能引起重度结肠炎，重度结肠炎有致命风险，因此本品只适用于毒性较低的抗菌药无法治疗的严重感染。克林霉素磷酸酯不得用于非细菌性感染，如大部分上呼吸道感染。艰难梭菌可产生毒素A和毒素B，进而促进CDAD的发生。产超毒素的艰难梭菌菌株可导致发病率和死亡率增高，由于这些感染会对抗菌药物治疗无效，可能需要结肠切除术治疗。对于使用抗生素后出现腹泻的所有患者应考虑CDAD。由于曾经有给予抗菌药物治疗超过2个月后发生CDAD的报道，因此有必要仔细询问病史。若怀疑或已确诊为CDAD，则需要停用不针对艰难梭菌的抗菌治疗，需要根据临床指征适当的调节水和电解质，补充蛋白质，并给予针对艰难梭菌的抗菌治疗，必要时进行手术评估。

问题：若怀疑或已确诊为艰难梭菌相关性腹泻（CDAD），资料建议采取哪些处理措施？
```

### 当前金标

- answer: 需要停用不针对艰难梭菌的抗菌治疗，需要根据临床指征适当的调节水和电解质，补充蛋白质，并给予针对艰难梭菌的抗菌治疗，必要时进行手术评估。
- required_points: ["需要停用不针对艰难梭菌的抗菌治疗，需要根据临床指征适当的调节水和电解质，补充蛋白质，并给予针对艰难梭菌的抗菌治疗，必要时进行手术评估。"]
- unavailable_points: []
- answerable_part: 
- insufficient_part: 

## qa_c50bca4de084__rephrase_1 / rephrase_1

- content_hash: `e376ee45b8483e01bcba1ab841d8344dcef4e56e4e53d892d6dfdee074c62a45`
- core_question_id: `qfam_73264e36deffd34615cb`
- expected_action: `answer`
- evidence_state: `sufficient`

### 实际输入

**system**

```text
仅依据提供的资料回答；资料不足时说明缺少的信息。
```

**user**

```text
资料：
警告本品辅料中含有苯甲醇，禁止用于儿童肌肉注射。本品与林可霉素、克林霉素有交叉耐药性，对克林霉素或林可霉素有过敏史者禁用。禁止与氨苄青霉素、苯妥英钠、巴比妥盐、氨茶碱、葡萄糖酸钙及硫酸镁配伍；与红霉素呈拮抗作用，不宜合用。包括克林霉素磷酸酯在内的几乎所有的抗生素都会引起艰难梭菌（Clostridium
difficile）相关性腹泻（CDAD），严重程度可由轻度腹泻到致命性结肠炎。抗菌药物治疗可改变肠道正常菌群，导致艰难梭菌的过度生长。使用克林霉素磷酸酯注射液后可能引起重度结肠炎，重度结肠炎有致命风险，因此本品只适用于毒性较低的抗菌药无法治疗的严重感染。克林霉素磷酸酯不得用于非细菌性感染，如大部分上呼吸道感染。艰难梭菌可产生毒素A和毒素B，进而促进CDAD的发生。产超毒素的艰难梭菌菌株可导致发病率和死亡率增高，由于这些感染会对抗菌药物治疗无效，可能需要结肠切除术治疗。对于使用抗生素后出现腹泻的所有患者应考虑CDAD。由于曾经有给予抗菌药物治疗超过2个月后发生CDAD的报道，因此有必要仔细询问病史。若怀疑或已确诊为CDAD，则需要停用不针对艰难梭菌的抗菌治疗，需要根据临床指征适当的调节水和电解质，补充蛋白质，并给予针对艰难梭菌的抗菌治疗，必要时进行手术评估。

问题：怀疑或已经确诊艰难梭菌相关性腹泻（CDAD）时，建议如何处理？
```

### 当前金标

- answer: 需要停用不针对艰难梭菌的抗菌治疗，需要根据临床指征适当的调节水和电解质，补充蛋白质，并给予针对艰难梭菌的抗菌治疗，必要时进行手术评估。
- required_points: ["需要停用不针对艰难梭菌的抗菌治疗，需要根据临床指征适当的调节水和电解质，补充蛋白质，并给予针对艰难梭菌的抗菌治疗，必要时进行手术评估。"]
- unavailable_points: []
- answerable_part: 
- insufficient_part: 

## qa_c50bca4de084__rephrase_2 / rephrase_2

- content_hash: `9217f807094d0ab7694d9dd0044cee715232b082f481bba683d044f68b221c99`
- core_question_id: `qfam_73264e36deffd34615cb`
- expected_action: `answer`
- evidence_state: `sufficient`

### 实际输入

**system**

```text
仅依据提供的资料回答；资料不足时说明缺少的信息。
```

**user**

```text
资料：
警告本品辅料中含有苯甲醇，禁止用于儿童肌肉注射。本品与林可霉素、克林霉素有交叉耐药性，对克林霉素或林可霉素有过敏史者禁用。禁止与氨苄青霉素、苯妥英钠、巴比妥盐、氨茶碱、葡萄糖酸钙及硫酸镁配伍；与红霉素呈拮抗作用，不宜合用。包括克林霉素磷酸酯在内的几乎所有的抗生素都会引起艰难梭菌（Clostridium
difficile）相关性腹泻（CDAD），严重程度可由轻度腹泻到致命性结肠炎。抗菌药物治疗可改变肠道正常菌群，导致艰难梭菌的过度生长。使用克林霉素磷酸酯注射液后可能引起重度结肠炎，重度结肠炎有致命风险，因此本品只适用于毒性较低的抗菌药无法治疗的严重感染。克林霉素磷酸酯不得用于非细菌性感染，如大部分上呼吸道感染。艰难梭菌可产生毒素A和毒素B，进而促进CDAD的发生。产超毒素的艰难梭菌菌株可导致发病率和死亡率增高，由于这些感染会对抗菌药物治疗无效，可能需要结肠切除术治疗。对于使用抗生素后出现腹泻的所有患者应考虑CDAD。由于曾经有给予抗菌药物治疗超过2个月后发生CDAD的报道，因此有必要仔细询问病史。若怀疑或已确诊为CDAD，则需要停用不针对艰难梭菌的抗菌治疗，需要根据临床指征适当的调节水和电解质，补充蛋白质，并给予针对艰难梭菌的抗菌治疗，必要时进行手术评估。

问题：对疑似或确诊的艰难梭菌相关性腹泻（CDAD），应采取哪些处理措施？
```

### 当前金标

- answer: 需要停用不针对艰难梭菌的抗菌治疗，需要根据临床指征适当的调节水和电解质，补充蛋白质，并给予针对艰难梭菌的抗菌治疗，必要时进行手术评估。
- required_points: ["需要停用不针对艰难梭菌的抗菌治疗，需要根据临床指征适当的调节水和电解质，补充蛋白质，并给予针对艰难梭菌的抗菌治疗，必要时进行手术评估。"]
- unavailable_points: []
- answerable_part: 
- insufficient_part: 

## qa_0d19335bbd54 / original

- content_hash: `a20c7924d45f7a39b0abe375dc63fcfbee47e7f8c91966043d69230ac31bf5ea`
- core_question_id: `qfam_cd8487cdc4abee1282ab`
- expected_action: `answer`
- evidence_state: `sufficient`

### 实际输入

**system**

```text
仅依据提供的资料回答；资料不足时说明缺少的信息。
```

**user**

```text
资料：
警告：二甲双胍相关乳酸酸中毒完整的黑框警告请参见本品的完整说明书【注意事项】。1.已有上市后二甲双胍相关乳酸酸中毒病例导致死亡、低体温、低血压和顽固性缓慢型心率失常发生。其症状包括不适、肌痛、呼吸窘迫、嗜睡和腹痛。实验室异常包括血乳酸水平升高（＞5mmol/L）、阴离子间隙酸中毒（不具有酮尿或酮血症证据）、乳酸/丙酮酸比值增加；同时伴有血浆二甲双胍浓度一般＞5mcg/mL。2.危险因素包括肾功能不全、合并使用某些药物、年龄≥65岁、使用造影剂的放射成像检查、手术和其他操作、缺氧状态、过度饮酒和肝功能损害。这些高危人群的二甲双胍相关乳酸酸中毒风险降低和控制措施参见说明书完整信息。3.如果怀疑二甲双胍相关乳酸酸中毒，请停止服用本品并且到医院开始一般支持性措施。建议立即进行血液透析。

问题：二甲双胍相关乳酸酸中毒的实验室异常表现包括哪些指标？
```

### 当前金标

- answer: 血乳酸水平升高（＞5mmol/L）、阴离子间隙酸中毒（不具有酮尿或酮血症证据）、乳酸/丙酮酸比值增加；同时伴有血浆二甲双胍浓度一般＞5mcg/mL。
- required_points: ["血乳酸水平升高（＞5mmol/L）、阴离子间隙酸中毒（不具有酮尿或酮血症证据）、乳酸/丙酮酸比值增加；同时伴有血浆二甲双胍浓度一般＞5mcg/mL。"]
- unavailable_points: []
- answerable_part: 
- insufficient_part: 

## qa_0d19335bbd54__rephrase_1 / rephrase_1

- content_hash: `ee7c730a8c06913f6d02c3dd35268fdb83e19b0d343ec50f5590483f938cd84e`
- core_question_id: `qfam_cd8487cdc4abee1282ab`
- expected_action: `answer`
- evidence_state: `sufficient`

### 实际输入

**system**

```text
仅依据提供的资料回答；资料不足时说明缺少的信息。
```

**user**

```text
资料：
警告：二甲双胍相关乳酸酸中毒完整的黑框警告请参见本品的完整说明书【注意事项】。1.已有上市后二甲双胍相关乳酸酸中毒病例导致死亡、低体温、低血压和顽固性缓慢型心率失常发生。其症状包括不适、肌痛、呼吸窘迫、嗜睡和腹痛。实验室异常包括血乳酸水平升高（＞5mmol/L）、阴离子间隙酸中毒（不具有酮尿或酮血症证据）、乳酸/丙酮酸比值增加；同时伴有血浆二甲双胍浓度一般＞5mcg/mL。2.危险因素包括肾功能不全、合并使用某些药物、年龄≥65岁、使用造影剂的放射成像检查、手术和其他操作、缺氧状态、过度饮酒和肝功能损害。这些高危人群的二甲双胍相关乳酸酸中毒风险降低和控制措施参见说明书完整信息。3.如果怀疑二甲双胍相关乳酸酸中毒，请停止服用本品并且到医院开始一般支持性措施。建议立即进行血液透析。

问题：发生二甲双胍相关乳酸酸中毒时，实验室有哪些异常指标？
```

### 当前金标

- answer: 血乳酸水平升高（＞5mmol/L）、阴离子间隙酸中毒（不具有酮尿或酮血症证据）、乳酸/丙酮酸比值增加；同时伴有血浆二甲双胍浓度一般＞5mcg/mL。
- required_points: ["血乳酸水平升高（＞5mmol/L）、阴离子间隙酸中毒（不具有酮尿或酮血症证据）、乳酸/丙酮酸比值增加；同时伴有血浆二甲双胍浓度一般＞5mcg/mL。"]
- unavailable_points: []
- answerable_part: 
- insufficient_part: 

## qa_0d19335bbd54__rephrase_2 / rephrase_2

- content_hash: `18a5e3f2a7afcec5942e9c8a6fb43e9ee17fb0aa91bd40df4efce5dacc257442`
- core_question_id: `qfam_cd8487cdc4abee1282ab`
- expected_action: `answer`
- evidence_state: `sufficient`

### 实际输入

**system**

```text
仅依据提供的资料回答；资料不足时说明缺少的信息。
```

**user**

```text
资料：
警告：二甲双胍相关乳酸酸中毒完整的黑框警告请参见本品的完整说明书【注意事项】。1.已有上市后二甲双胍相关乳酸酸中毒病例导致死亡、低体温、低血压和顽固性缓慢型心率失常发生。其症状包括不适、肌痛、呼吸窘迫、嗜睡和腹痛。实验室异常包括血乳酸水平升高（＞5mmol/L）、阴离子间隙酸中毒（不具有酮尿或酮血症证据）、乳酸/丙酮酸比值增加；同时伴有血浆二甲双胍浓度一般＞5mcg/mL。2.危险因素包括肾功能不全、合并使用某些药物、年龄≥65岁、使用造影剂的放射成像检查、手术和其他操作、缺氧状态、过度饮酒和肝功能损害。这些高危人群的二甲双胍相关乳酸酸中毒风险降低和控制措施参见说明书完整信息。3.如果怀疑二甲双胍相关乳酸酸中毒，请停止服用本品并且到医院开始一般支持性措施。建议立即进行血液透析。

问题：二甲双胍相关乳酸酸中毒在化验结果上有哪些异常表现？
```

### 当前金标

- answer: 血乳酸水平升高（＞5mmol/L）、阴离子间隙酸中毒（不具有酮尿或酮血症证据）、乳酸/丙酮酸比值增加；同时伴有血浆二甲双胍浓度一般＞5mcg/mL。
- required_points: ["血乳酸水平升高（＞5mmol/L）、阴离子间隙酸中毒（不具有酮尿或酮血症证据）、乳酸/丙酮酸比值增加；同时伴有血浆二甲双胍浓度一般＞5mcg/mL。"]
- unavailable_points: []
- answerable_part: 
- insufficient_part: 

## qa_1170d7e9d871 / original

- content_hash: `abe873e7c8d2dd08ce2814292632f54291d29b4db6b52919e8e87e4e10d7d33a`
- core_question_id: `qfam_172279245ea13b6a7d29`
- expected_action: `answer`
- evidence_state: `sufficient`

### 实际输入

**system**

```text
仅依据提供的资料回答；资料不足时说明缺少的信息。
```

**user**

```text
资料：
本品为鳖蠊科昆虫地鳖Eupolyphaga sinensis Walker或冀地鳖Steleophaga plancyi（Boleny）的雌虫干燥体。捕捉后，置沸水中烫死，晒干或烘干。

问题：地鳖虫药材的基原昆虫包括哪两种？
```

### 当前金标

- answer: 鳖蠊科昆虫地鳖Eupolyphaga sinensis Walker或冀地鳖Steleophaga plancyi（Boleny）的雌虫。
- required_points: ["鳖蠊科昆虫地鳖Eupolyphaga sinensis Walker或冀地鳖Steleophaga plancyi（Boleny）的雌虫。"]
- unavailable_points: []
- answerable_part: 
- insufficient_part: 

## qa_1170d7e9d871__rephrase_1 / rephrase_1

- content_hash: `21aaafa8afb16a3f7f9667b4ab6ddcc595a601ff9e66d90c4b7e797df69c7ae6`
- core_question_id: `qfam_172279245ea13b6a7d29`
- expected_action: `answer`
- evidence_state: `sufficient`

### 实际输入

**system**

```text
仅依据提供的资料回答；资料不足时说明缺少的信息。
```

**user**

```text
资料：
本品为鳖蠊科昆虫地鳖Eupolyphaga sinensis Walker或冀地鳖Steleophaga plancyi（Boleny）的雌虫干燥体。捕捉后，置沸水中烫死，晒干或烘干。

问题：地鳖虫药材来自哪两种基原昆虫？
```

### 当前金标

- answer: 鳖蠊科昆虫地鳖Eupolyphaga sinensis Walker或冀地鳖Steleophaga plancyi（Boleny）的雌虫。
- required_points: ["鳖蠊科昆虫地鳖Eupolyphaga sinensis Walker或冀地鳖Steleophaga plancyi（Boleny）的雌虫。"]
- unavailable_points: []
- answerable_part: 
- insufficient_part: 

## qa_1170d7e9d871__rephrase_2 / rephrase_2

- content_hash: `5620d43eaf1524efa8fb474173ee6bacdf7be0e581a31594c01a71154c7d756c`
- core_question_id: `qfam_172279245ea13b6a7d29`
- expected_action: `answer`
- evidence_state: `sufficient`

### 实际输入

**system**

```text
仅依据提供的资料回答；资料不足时说明缺少的信息。
```

**user**

```text
资料：
本品为鳖蠊科昆虫地鳖Eupolyphaga sinensis Walker或冀地鳖Steleophaga plancyi（Boleny）的雌虫干燥体。捕捉后，置沸水中烫死，晒干或烘干。

问题：作为地鳖虫药材来源的基原昆虫有哪两种？
```

### 当前金标

- answer: 鳖蠊科昆虫地鳖Eupolyphaga sinensis Walker或冀地鳖Steleophaga plancyi（Boleny）的雌虫。
- required_points: ["鳖蠊科昆虫地鳖Eupolyphaga sinensis Walker或冀地鳖Steleophaga plancyi（Boleny）的雌虫。"]
- unavailable_points: []
- answerable_part: 
- insufficient_part: 

## qa_8e80f39b4ae3 / original

- content_hash: `3bc99631e93a3f2533eb1620939ad6e1583f20e7d421fde53cd365cd03126d77`
- core_question_id: `qfam_83532e640135f10909bc`
- expected_action: `answer`
- evidence_state: `sufficient`

### 实际输入

**system**

```text
仅依据提供的资料回答；资料不足时说明缺少的信息。
```

**user**

```text
资料：
本品为莎草科植物莎草Cyperus rotundus L.的干燥根茎。秋季采挖，燎去毛须，置沸水中略煮或蒸透后晒干，或燎后直接晒干。

问题：根据资料，莎草Cyperus rotundus L.的干燥根茎在采挖后首先需要进行什么处理？
```

### 当前金标

- answer: 燎去毛须。
- required_points: ["燎去毛须。"]
- unavailable_points: []
- answerable_part: 
- insufficient_part: 

## qa_8e80f39b4ae3__rephrase_1 / rephrase_1

- content_hash: `91d7cea7178cc40e394a3aedd30e13b07563fedfb5fbcfe8bf75e7c1a21f84cb`
- core_question_id: `qfam_83532e640135f10909bc`
- expected_action: `answer`
- evidence_state: `sufficient`

### 实际输入

**system**

```text
仅依据提供的资料回答；资料不足时说明缺少的信息。
```

**user**

```text
资料：
本品为莎草科植物莎草Cyperus rotundus L.的干燥根茎。秋季采挖，燎去毛须，置沸水中略煮或蒸透后晒干，或燎后直接晒干。

问题：莎草Cyperus rotundus L.的干燥根茎采挖之后，首先要做什么处理？
```

### 当前金标

- answer: 燎去毛须。
- required_points: ["燎去毛须。"]
- unavailable_points: []
- answerable_part: 
- insufficient_part: 

## qa_8e80f39b4ae3__rephrase_2 / rephrase_2

- content_hash: `26a4348d9738ad0b038693ef462ed24c147ec1ed5233445e9c3539a0a18f3a67`
- core_question_id: `qfam_83532e640135f10909bc`
- expected_action: `answer`
- evidence_state: `sufficient`

### 实际输入

**system**

```text
仅依据提供的资料回答；资料不足时说明缺少的信息。
```

**user**

```text
资料：
本品为莎草科植物莎草Cyperus rotundus L.的干燥根茎。秋季采挖，燎去毛须，置沸水中略煮或蒸透后晒干，或燎后直接晒干。

问题：采挖莎草Cyperus rotundus L.的干燥根茎后，最先进行的处理是什么？
```

### 当前金标

- answer: 燎去毛须。
- required_points: ["燎去毛须。"]
- unavailable_points: []
- answerable_part: 
- insufficient_part: 

## qa_8aeaee12775b / original

- content_hash: `db5f6c0903d608fe50424c7e502efe963cbcc9c27c376c4d0237f490f15db0e6`
- core_question_id: `qfam_db6eb43eea683e665072`
- expected_action: `answer`
- evidence_state: `sufficient`

### 实际输入

**system**

```text
仅依据提供的资料回答；资料不足时说明缺少的信息。
```

**user**

```text
资料：
口服：一次15～30g（1～2瓶）（可用水100ml调匀），一日1～2次，连用2～3日。若有便秘可合并服用30g甘露醇粉或山梨醇粉。直肠给药：一次30g（2瓶），用水或20%甘露醇100～200ml混匀作高位保留灌肠，一日1～2次，连用3～7日。小儿：用法同成人，剂量按每公斤体重一日1g计。

问题：资料中规定直肠给药时，高位保留灌肠的混匀介质是什么？
```

### 当前金标

- answer: 水或20%甘露醇100～200ml
- required_points: ["水或20%甘露醇100～200ml"]
- unavailable_points: []
- answerable_part: 
- insufficient_part: 

## qa_8aeaee12775b__rephrase_1 / rephrase_1

- content_hash: `2e8d4b154868f3360f79032a7127cc926d39c8d0cceb6c9a413e3a0a3a578c1c`
- core_question_id: `qfam_db6eb43eea683e665072`
- expected_action: `answer`
- evidence_state: `sufficient`

### 实际输入

**system**

```text
仅依据提供的资料回答；资料不足时说明缺少的信息。
```

**user**

```text
资料：
口服：一次15～30g（1～2瓶）（可用水100ml调匀），一日1～2次，连用2～3日。若有便秘可合并服用30g甘露醇粉或山梨醇粉。直肠给药：一次30g（2瓶），用水或20%甘露醇100～200ml混匀作高位保留灌肠，一日1～2次，连用3～7日。小儿：用法同成人，剂量按每公斤体重一日1g计。

问题：直肠给药做高位保留灌肠时，用什么介质混匀？
```

### 当前金标

- answer: 水或20%甘露醇100～200ml
- required_points: ["水或20%甘露醇100～200ml"]
- unavailable_points: []
- answerable_part: 
- insufficient_part: 

## qa_8aeaee12775b__rephrase_2 / rephrase_2

- content_hash: `5a4c5a155ffcc8de90ae502b7bc7a7ffeb8f82e6ce6d9a0d349c6d76fec0c72d`
- core_question_id: `qfam_db6eb43eea683e665072`
- expected_action: `answer`
- evidence_state: `sufficient`

### 实际输入

**system**

```text
仅依据提供的资料回答；资料不足时说明缺少的信息。
```

**user**

```text
资料：
口服：一次15～30g（1～2瓶）（可用水100ml调匀），一日1～2次，连用2～3日。若有便秘可合并服用30g甘露醇粉或山梨醇粉。直肠给药：一次30g（2瓶），用水或20%甘露醇100～200ml混匀作高位保留灌肠，一日1～2次，连用3～7日。小儿：用法同成人，剂量按每公斤体重一日1g计。

问题：高位保留灌肠这种直肠给药方式，规定的混匀介质是什么？
```

### 当前金标

- answer: 水或20%甘露醇100～200ml
- required_points: ["水或20%甘露醇100～200ml"]
- unavailable_points: []
- answerable_part: 
- insufficient_part: 

## qa_b2cb8a9172ec / original

- content_hash: `b914c3bee328c4d33a150f3d3a57016878034311b63b094d0c746c234076dc4b`
- core_question_id: `qfam_1960f19815ff6c491677`
- expected_action: `answer`
- evidence_state: `sufficient`

### 实际输入

**system**

```text
仅依据提供的资料回答；资料不足时说明缺少的信息。
```

**user**

```text
资料：
本品活性成份为匹伐他汀钙。化学名称：（+）－双｛（3R，5S，6E）－7－［2－环丙基－4－（4－氟苯基）－3－喹啉基］－3，5－二羟基－6－庚酸｝单钙盐。分子式：C50H46CaF2N2O8

问题：匹伐他汀钙的化学名称是什么？
```

### 当前金标

- answer: （+）－双｛（3R，5S，6E）－7－［2－环丙基－4－（4－氟苯基）－3－喹啉基］－3，5－二羟基－6－庚酸｝单钙盐
- required_points: ["（+）－双｛（3R，5S，6E）－7－［2－环丙基－4－（4－氟苯基）－3－喹啉基］－3，5－二羟基－6－庚酸｝单钙盐"]
- unavailable_points: []
- answerable_part: 
- insufficient_part: 

## qa_b2cb8a9172ec__rephrase_1 / rephrase_1

- content_hash: `e301e86ba4eb79a3b644395e4db3db35e64207fa1ee7b07cca6134858c1079a4`
- core_question_id: `qfam_1960f19815ff6c491677`
- expected_action: `answer`
- evidence_state: `sufficient`

### 实际输入

**system**

```text
仅依据提供的资料回答；资料不足时说明缺少的信息。
```

**user**

```text
资料：
本品活性成份为匹伐他汀钙。化学名称：（+）－双｛（3R，5S，6E）－7－［2－环丙基－4－（4－氟苯基）－3－喹啉基］－3，5－二羟基－6－庚酸｝单钙盐。分子式：C50H46CaF2N2O8

问题：匹伐他汀钙对应的化学名称如何书写？
```

### 当前金标

- answer: （+）－双｛（3R，5S，6E）－7－［2－环丙基－4－（4－氟苯基）－3－喹啉基］－3，5－二羟基－6－庚酸｝单钙盐
- required_points: ["（+）－双｛（3R，5S，6E）－7－［2－环丙基－4－（4－氟苯基）－3－喹啉基］－3，5－二羟基－6－庚酸｝单钙盐"]
- unavailable_points: []
- answerable_part: 
- insufficient_part: 

## qa_b2cb8a9172ec__rephrase_2 / rephrase_2

- content_hash: `366bd9d7ba13a7a893f395eb019d902adae276770f2e55d5f11257d50ceb4565`
- core_question_id: `qfam_1960f19815ff6c491677`
- expected_action: `answer`
- evidence_state: `sufficient`

### 实际输入

**system**

```text
仅依据提供的资料回答；资料不足时说明缺少的信息。
```

**user**

```text
资料：
本品活性成份为匹伐他汀钙。化学名称：（+）－双｛（3R，5S，6E）－7－［2－环丙基－4－（4－氟苯基）－3－喹啉基］－3，5－二羟基－6－庚酸｝单钙盐。分子式：C50H46CaF2N2O8

问题：请写出匹伐他汀钙的化学名称。
```

### 当前金标

- answer: （+）－双｛（3R，5S，6E）－7－［2－环丙基－4－（4－氟苯基）－3－喹啉基］－3，5－二羟基－6－庚酸｝单钙盐
- required_points: ["（+）－双｛（3R，5S，6E）－7－［2－环丙基－4－（4－氟苯基）－3－喹啉基］－3，5－二羟基－6－庚酸｝单钙盐"]
- unavailable_points: []
- answerable_part: 
- insufficient_part: 

## qa_000adeec0933 / original

- content_hash: `ed82caadc9727b5de9eaef39550a30c5e118652313f3f9dc1dbcb05a39228a01`
- core_question_id: `qfam_8adea35e80915295cf6f`
- expected_action: `answer`
- evidence_state: `sufficient`

### 实际输入

**system**

```text
仅依据提供的资料回答；资料不足时说明缺少的信息。
```

**user**

```text
资料：
警告：二甲双胍相关乳酸酸中毒完整的黑框警告请参见本品的完整说明书【注意事项】。1.已有上市后二甲双胍相关乳酸酸中毒病例导致死亡、低体温、低血压和顽固性缓慢型心率失常发生。其症状包括不适、肌痛、呼吸窘迫、嗜睡和腹痛。实验室异常包括血乳酸水平升高（＞5mmol/L）、阴离子间隙酸中毒（不具有酮尿或酮血症证据）、乳酸/丙酮酸比值增加；同时伴有血浆二甲双胍浓度一般＞5mcg/mL。2.危险因素包括肾功能不全、合并使用某些药物、年龄≥65岁、使用造影剂的放射成像检查、手术和其他操作、缺氧状态、过度饮酒和肝功能损害。这些高危人群的二甲双胍相关乳酸酸中毒风险降低和控制措施参见说明书完整信息。3.如果怀疑二甲双胍相关乳酸酸中毒，请停止服用本品并且到医院开始一般支持性措施。建议立即进行血液透析。

问题：如果怀疑发生二甲双胍相关乳酸酸中毒，应采取什么措施？
```

### 当前金标

- answer: 停止服用本品并且到医院开始一般支持性措施，建议立即进行血液透析。
- required_points: ["停止服用本品并且到医院开始一般支持性措施，建议立即进行血液透析。"]
- unavailable_points: []
- answerable_part: 
- insufficient_part: 

## qa_000adeec0933__rephrase_1 / rephrase_1

- content_hash: `f55d222f1a04f891b81cdc41e4035fab4c8c2f81b818f05ac12be06d6fcfa900`
- core_question_id: `qfam_8adea35e80915295cf6f`
- expected_action: `answer`
- evidence_state: `sufficient`

### 实际输入

**system**

```text
仅依据提供的资料回答；资料不足时说明缺少的信息。
```

**user**

```text
资料：
警告：二甲双胍相关乳酸酸中毒完整的黑框警告请参见本品的完整说明书【注意事项】。1.已有上市后二甲双胍相关乳酸酸中毒病例导致死亡、低体温、低血压和顽固性缓慢型心率失常发生。其症状包括不适、肌痛、呼吸窘迫、嗜睡和腹痛。实验室异常包括血乳酸水平升高（＞5mmol/L）、阴离子间隙酸中毒（不具有酮尿或酮血症证据）、乳酸/丙酮酸比值增加；同时伴有血浆二甲双胍浓度一般＞5mcg/mL。2.危险因素包括肾功能不全、合并使用某些药物、年龄≥65岁、使用造影剂的放射成像检查、手术和其他操作、缺氧状态、过度饮酒和肝功能损害。这些高危人群的二甲双胍相关乳酸酸中毒风险降低和控制措施参见说明书完整信息。3.如果怀疑二甲双胍相关乳酸酸中毒，请停止服用本品并且到医院开始一般支持性措施。建议立即进行血液透析。

问题：怀疑出现二甲双胍相关乳酸酸中毒时，应当怎样处理？
```

### 当前金标

- answer: 停止服用本品并且到医院开始一般支持性措施，建议立即进行血液透析。
- required_points: ["停止服用本品并且到医院开始一般支持性措施，建议立即进行血液透析。"]
- unavailable_points: []
- answerable_part: 
- insufficient_part: 

## qa_000adeec0933__rephrase_2 / rephrase_2

- content_hash: `87df8d7ae9ed35982c3c0a9972edc4c8868cb99f1e2f41915643179c27407738`
- core_question_id: `qfam_8adea35e80915295cf6f`
- expected_action: `answer`
- evidence_state: `sufficient`

### 实际输入

**system**

```text
仅依据提供的资料回答；资料不足时说明缺少的信息。
```

**user**

```text
资料：
警告：二甲双胍相关乳酸酸中毒完整的黑框警告请参见本品的完整说明书【注意事项】。1.已有上市后二甲双胍相关乳酸酸中毒病例导致死亡、低体温、低血压和顽固性缓慢型心率失常发生。其症状包括不适、肌痛、呼吸窘迫、嗜睡和腹痛。实验室异常包括血乳酸水平升高（＞5mmol/L）、阴离子间隙酸中毒（不具有酮尿或酮血症证据）、乳酸/丙酮酸比值增加；同时伴有血浆二甲双胍浓度一般＞5mcg/mL。2.危险因素包括肾功能不全、合并使用某些药物、年龄≥65岁、使用造影剂的放射成像检查、手术和其他操作、缺氧状态、过度饮酒和肝功能损害。这些高危人群的二甲双胍相关乳酸酸中毒风险降低和控制措施参见说明书完整信息。3.如果怀疑二甲双胍相关乳酸酸中毒，请停止服用本品并且到医院开始一般支持性措施。建议立即进行血液透析。

问题：一旦怀疑已经发生二甲双胍相关乳酸酸中毒，需要采取什么措施？
```

### 当前金标

- answer: 停止服用本品并且到医院开始一般支持性措施，建议立即进行血液透析。
- required_points: ["停止服用本品并且到医院开始一般支持性措施，建议立即进行血液透析。"]
- unavailable_points: []
- answerable_part: 
- insufficient_part: 

## qa_c8e2731ba8c2 / original

- content_hash: `32679f8886de207247aa6ee22311febe30063217653ed3cf3d4e1f7ab06ed81b`
- core_question_id: `qfam_1979019076df32c41cd7`
- expected_action: `answer`
- evidence_state: `sufficient`

### 实际输入

**system**

```text
仅依据提供的资料回答；资料不足时说明缺少的信息。
```

**user**

```text
资料：
本品为禾本科植物大麦Hordeum vulgare L.的成熟果实经发芽干燥的炮制加工品。将麦粒用水浸泡后，保持适宜温、湿度，待幼芽长至约5mm时，晒干或低温干燥。

问题：资料中提到的本品是由哪种植物的成熟果实经发芽干燥制成的？
```

### 当前金标

- answer: 禾本科植物大麦Hordeum vulgare L.
- required_points: ["禾本科植物大麦Hordeum vulgare L."]
- unavailable_points: []
- answerable_part: 
- insufficient_part: 

## qa_c8e2731ba8c2__rephrase_1 / rephrase_1

- content_hash: `e20aeb52e6458e23e33f0f2331f1300d974a2f694754cba0fdb18fb8f8b89c30`
- core_question_id: `qfam_1979019076df32c41cd7`
- expected_action: `answer`
- evidence_state: `sufficient`

### 实际输入

**system**

```text
仅依据提供的资料回答；资料不足时说明缺少的信息。
```

**user**

```text
资料：
本品为禾本科植物大麦Hordeum vulgare L.的成熟果实经发芽干燥的炮制加工品。将麦粒用水浸泡后，保持适宜温、湿度，待幼芽长至约5mm时，晒干或低温干燥。

问题：本品使用的成熟果实来自哪种植物，并经过发芽干燥制成？
```

### 当前金标

- answer: 禾本科植物大麦Hordeum vulgare L.
- required_points: ["禾本科植物大麦Hordeum vulgare L."]
- unavailable_points: []
- answerable_part: 
- insufficient_part: 

## qa_c8e2731ba8c2__rephrase_2 / rephrase_2

- content_hash: `ea0870caf1c2998a9e169a4e8dd486b8bffa7026b2d5afd75693b3ea6324384c`
- core_question_id: `qfam_1979019076df32c41cd7`
- expected_action: `answer`
- evidence_state: `sufficient`

### 实际输入

**system**

```text
仅依据提供的资料回答；资料不足时说明缺少的信息。
```

**user**

```text
资料：
本品为禾本科植物大麦Hordeum vulgare L.的成熟果实经发芽干燥的炮制加工品。将麦粒用水浸泡后，保持适宜温、湿度，待幼芽长至约5mm时，晒干或低温干燥。

问题：这种经发芽干燥制成的成熟果实，来源于哪一种植物？
```

### 当前金标

- answer: 禾本科植物大麦Hordeum vulgare L.
- required_points: ["禾本科植物大麦Hordeum vulgare L."]
- unavailable_points: []
- answerable_part: 
- insufficient_part: 

## qa_c82bb4c8cd81 / original

- content_hash: `72ac0d917c16f908de89ca5262e83097147c95249b403417a3cedacbbf91e7df`
- core_question_id: `qfam_8d68c98425dc06adfb2e`
- expected_action: `answer`
- evidence_state: `sufficient`

### 实际输入

**system**

```text
仅依据提供的资料回答；资料不足时说明缺少的信息。
```

**user**

```text
资料：
本品为莎草科植物莎草Cyperus rotundus L.的干燥根茎。秋季采挖，燎去毛须，置沸水中略煮或蒸透后晒干，或燎后直接晒干。

问题：莎草Cyperus rotundus L.的干燥根茎在秋季采挖后，经过燎去毛须处理，后续有哪些具体的加工方式？
```

### 当前金标

- answer: 置沸水中略煮或蒸透后晒干，或燎后直接晒干。
- required_points: ["置沸水中略煮或蒸透后晒干，或燎后直接晒干。"]
- unavailable_points: []
- answerable_part: 
- insufficient_part: 

## qa_c82bb4c8cd81__rephrase_1 / rephrase_1

- content_hash: `7a1461c0f71b1b8fdd010393336df7b6a78c5b4fcc282f8a99be27dbbb655b7c`
- core_question_id: `qfam_8d68c98425dc06adfb2e`
- expected_action: `answer`
- evidence_state: `sufficient`

### 实际输入

**system**

```text
仅依据提供的资料回答；资料不足时说明缺少的信息。
```

**user**

```text
资料：
本品为莎草科植物莎草Cyperus rotundus L.的干燥根茎。秋季采挖，燎去毛须，置沸水中略煮或蒸透后晒干，或燎后直接晒干。

问题：秋季采挖莎草Cyperus rotundus L.的干燥根茎并燎去毛须之后，后续怎样加工？
```

### 当前金标

- answer: 置沸水中略煮或蒸透后晒干，或燎后直接晒干。
- required_points: ["置沸水中略煮或蒸透后晒干，或燎后直接晒干。"]
- unavailable_points: []
- answerable_part: 
- insufficient_part: 

## qa_c82bb4c8cd81__rephrase_2 / rephrase_2

- content_hash: `da4b84fa65456bb8285101999d0697bd341fcd135416024a89af795576065645`
- core_question_id: `qfam_8d68c98425dc06adfb2e`
- expected_action: `answer`
- evidence_state: `sufficient`

### 实际输入

**system**

```text
仅依据提供的资料回答；资料不足时说明缺少的信息。
```

**user**

```text
资料：
本品为莎草科植物莎草Cyperus rotundus L.的干燥根茎。秋季采挖，燎去毛须，置沸水中略煮或蒸透后晒干，或燎后直接晒干。

问题：莎草Cyperus rotundus L.的干燥根茎在秋季采挖、燎去毛须之后，还有哪些加工方式？
```

### 当前金标

- answer: 置沸水中略煮或蒸透后晒干，或燎后直接晒干。
- required_points: ["置沸水中略煮或蒸透后晒干，或燎后直接晒干。"]
- unavailable_points: []
- answerable_part: 
- insufficient_part: 

## qa_e0226e688dc6 / original

- content_hash: `4f8708e9635c779fec56ae4242d5890ec83d1687ef8b3a7036ef8b50c58b2c5e`
- core_question_id: `qfam_1d348122735cf5b827fd`
- expected_action: `answer`
- evidence_state: `sufficient`

### 实际输入

**system**

```text
仅依据提供的资料回答；资料不足时说明缺少的信息。
```

**user**

```text
资料：
狭叶番泻：呈长卵形或卵状披针形，长1.5～5cm，宽0.4～2cm，叶端急尖，叶基稍不对称，全缘。上表面黄绿色，下表面浅黄绿色，无毛或近无毛，叶脉稍隆起。革质。气微弱而特异，味微苦，稍有黏性。
尖叶番泻：呈披针形或长卵形，略卷曲，叶端短尖或微突，叶基不对称，两面均有细短毛茸。

问题：资料中描述的狭叶番泻表面颜色、毛被情况及质地特征有哪些？
```

### 当前金标

- answer: 狭叶番泻上表面黄绿色，下表面浅黄绿色，无毛或近无毛，质地为革质。
- required_points: ["狭叶番泻上表面黄绿色，下表面浅黄绿色，无毛或近无毛，质地为革质。"]
- unavailable_points: []
- answerable_part: 
- insufficient_part: 

## qa_e0226e688dc6__rephrase_1 / rephrase_1

- content_hash: `85be344568ea91672349e53744d72d24fb70f765c95834019322aea8f85dead1`
- core_question_id: `qfam_1d348122735cf5b827fd`
- expected_action: `answer`
- evidence_state: `sufficient`

### 实际输入

**system**

```text
仅依据提供的资料回答；资料不足时说明缺少的信息。
```

**user**

```text
资料：
狭叶番泻：呈长卵形或卵状披针形，长1.5～5cm，宽0.4～2cm，叶端急尖，叶基稍不对称，全缘。上表面黄绿色，下表面浅黄绿色，无毛或近无毛，叶脉稍隆起。革质。气微弱而特异，味微苦，稍有黏性。
尖叶番泻：呈披针形或长卵形，略卷曲，叶端短尖或微突，叶基不对称，两面均有细短毛茸。

问题：狭叶番泻的表面是什么颜色，毛被情况和质地有什么特征？
```

### 当前金标

- answer: 狭叶番泻上表面黄绿色，下表面浅黄绿色，无毛或近无毛，质地为革质。
- required_points: ["狭叶番泻上表面黄绿色，下表面浅黄绿色，无毛或近无毛，质地为革质。"]
- unavailable_points: []
- answerable_part: 
- insufficient_part: 

## qa_e0226e688dc6__rephrase_2 / rephrase_2

- content_hash: `0722216d8cf2264af9e132b80df2eeee1048941156dcc734d115565669d2beec`
- core_question_id: `qfam_1d348122735cf5b827fd`
- expected_action: `answer`
- evidence_state: `sufficient`

### 实际输入

**system**

```text
仅依据提供的资料回答；资料不足时说明缺少的信息。
```

**user**

```text
资料：
狭叶番泻：呈长卵形或卵状披针形，长1.5～5cm，宽0.4～2cm，叶端急尖，叶基稍不对称，全缘。上表面黄绿色，下表面浅黄绿色，无毛或近无毛，叶脉稍隆起。革质。气微弱而特异，味微苦，稍有黏性。
尖叶番泻：呈披针形或长卵形，略卷曲，叶端短尖或微突，叶基不对称，两面均有细短毛茸。

问题：请说明狭叶番泻在表面颜色、毛被和质地方面的特征。
```

### 当前金标

- answer: 狭叶番泻上表面黄绿色，下表面浅黄绿色，无毛或近无毛，质地为革质。
- required_points: ["狭叶番泻上表面黄绿色，下表面浅黄绿色，无毛或近无毛，质地为革质。"]
- unavailable_points: []
- answerable_part: 
- insufficient_part: 

## qa_5f89df2a0b6e / original

- content_hash: `95e37d17b4f38ac7322150cdb56813aa2110fe1f6cae6be256a0fbf56df76f77`
- core_question_id: `qfam_d1cb1bd3eb9d5a2f83b5`
- expected_action: `answer`
- evidence_state: `sufficient`

### 实际输入

**system**

```text
仅依据提供的资料回答；资料不足时说明缺少的信息。
```

**user**

```text
资料：
警示语：严重皮肤反应和HLA－B*1502等位基因在卡马西平治疗期间有报告发生严重且有时是致命的皮肤反应，包括中毒性表皮坏死松懈症（TEN）和Stevens－Johnson综合征（SJS）。在主要是高加索人群的国家中每10000名新用药者估计发生1～6例。但是这一风险在一些亚洲国家估计约比上述国家高10倍。对华裔患者的研究发现SJS/TEN的发生风险与患者体内携带人白细胞抗原HLA－B*1502等位基因之间存在很强的相关性，HLA－B*1502等位基因是HLA－B基因的遗传性等位基因变异体。H－LA－B*1502几乎仅在祖籍亚洲广泛地区的患者人群中发现。在开始卡马西平治疗前可对遗传风险人群患者进行HLA－B*1502筛查。此等位基因阳性患者不得使用卡马西平治疗，除非明确显示治疗效益大于风险（参见【注意事项】）。再生障碍性贫血和粒细胞缺乏症据报告，再生障碍性贫血和粒细胞缺乏症与使用卡马西平有关。来自基于人群的病例对照研究显示用药患者群体发生这些反应的风险比普通人群要高5～8倍，未接受治疗的普通人群这些反应的总体风险比较低，粒细胞缺乏症发生率平均每年每百万人中为6例，再生障碍性贫血平均每年每百万人中为2例。尽管使用卡马西平经常报告发生血小板或白细胞计数一过性或持续减少，但尚无数据来准确估计这些情况的发生率或结局。

问题：根据资料，在开始卡马西平治疗前，针对遗传风险人群患者建议进行什么筛查？
```

### 当前金标

- answer: HLA-B*1502筛查
- required_points: ["HLA-B*1502筛查"]
- unavailable_points: []
- answerable_part: 
- insufficient_part: 

## qa_5f89df2a0b6e__rephrase_1 / rephrase_1

- content_hash: `a986ac00311b35a3cb0f90db955eaef32bac7fb63284fa6baa4ac7e63e7a6a20`
- core_question_id: `qfam_d1cb1bd3eb9d5a2f83b5`
- expected_action: `answer`
- evidence_state: `sufficient`

### 实际输入

**system**

```text
仅依据提供的资料回答；资料不足时说明缺少的信息。
```

**user**

```text
资料：
警示语：严重皮肤反应和HLA－B*1502等位基因在卡马西平治疗期间有报告发生严重且有时是致命的皮肤反应，包括中毒性表皮坏死松懈症（TEN）和Stevens－Johnson综合征（SJS）。在主要是高加索人群的国家中每10000名新用药者估计发生1～6例。但是这一风险在一些亚洲国家估计约比上述国家高10倍。对华裔患者的研究发现SJS/TEN的发生风险与患者体内携带人白细胞抗原HLA－B*1502等位基因之间存在很强的相关性，HLA－B*1502等位基因是HLA－B基因的遗传性等位基因变异体。H－LA－B*1502几乎仅在祖籍亚洲广泛地区的患者人群中发现。在开始卡马西平治疗前可对遗传风险人群患者进行HLA－B*1502筛查。此等位基因阳性患者不得使用卡马西平治疗，除非明确显示治疗效益大于风险（参见【注意事项】）。再生障碍性贫血和粒细胞缺乏症据报告，再生障碍性贫血和粒细胞缺乏症与使用卡马西平有关。来自基于人群的病例对照研究显示用药患者群体发生这些反应的风险比普通人群要高5～8倍，未接受治疗的普通人群这些反应的总体风险比较低，粒细胞缺乏症发生率平均每年每百万人中为6例，再生障碍性贫血平均每年每百万人中为2例。尽管使用卡马西平经常报告发生血小板或白细胞计数一过性或持续减少，但尚无数据来准确估计这些情况的发生率或结局。

问题：对遗传风险人群，开始卡马西平治疗之前建议做哪项筛查？
```

### 当前金标

- answer: HLA-B*1502筛查
- required_points: ["HLA-B*1502筛查"]
- unavailable_points: []
- answerable_part: 
- insufficient_part: 

## qa_5f89df2a0b6e__rephrase_2 / rephrase_2

- content_hash: `9e765e07b7b93133bee6e6e8a9b175889d1cbb7b7cd11b48356f1dd895d3d245`
- core_question_id: `qfam_d1cb1bd3eb9d5a2f83b5`
- expected_action: `answer`
- evidence_state: `sufficient`

### 实际输入

**system**

```text
仅依据提供的资料回答；资料不足时说明缺少的信息。
```

**user**

```text
资料：
警示语：严重皮肤反应和HLA－B*1502等位基因在卡马西平治疗期间有报告发生严重且有时是致命的皮肤反应，包括中毒性表皮坏死松懈症（TEN）和Stevens－Johnson综合征（SJS）。在主要是高加索人群的国家中每10000名新用药者估计发生1～6例。但是这一风险在一些亚洲国家估计约比上述国家高10倍。对华裔患者的研究发现SJS/TEN的发生风险与患者体内携带人白细胞抗原HLA－B*1502等位基因之间存在很强的相关性，HLA－B*1502等位基因是HLA－B基因的遗传性等位基因变异体。H－LA－B*1502几乎仅在祖籍亚洲广泛地区的患者人群中发现。在开始卡马西平治疗前可对遗传风险人群患者进行HLA－B*1502筛查。此等位基因阳性患者不得使用卡马西平治疗，除非明确显示治疗效益大于风险（参见【注意事项】）。再生障碍性贫血和粒细胞缺乏症据报告，再生障碍性贫血和粒细胞缺乏症与使用卡马西平有关。来自基于人群的病例对照研究显示用药患者群体发生这些反应的风险比普通人群要高5～8倍，未接受治疗的普通人群这些反应的总体风险比较低，粒细胞缺乏症发生率平均每年每百万人中为6例，再生障碍性贫血平均每年每百万人中为2例。尽管使用卡马西平经常报告发生血小板或白细胞计数一过性或持续减少，但尚无数据来准确估计这些情况的发生率或结局。

问题：遗传风险人群在启用卡马西平之前，建议先进行什么筛查？
```

### 当前金标

- answer: HLA-B*1502筛查
- required_points: ["HLA-B*1502筛查"]
- unavailable_points: []
- answerable_part: 
- insufficient_part: 

## qa_176b8cf386ae / original

- content_hash: `b8eeae7f54c0ba4236f405707fcb18dbc557a07ff68074d99bc7d297cd33754e`
- core_question_id: `qfam_343d9bfa5fe239f56bb9`
- expected_action: `answer`
- evidence_state: `sufficient`

### 实际输入

**system**

```text
仅依据提供的资料回答；资料不足时说明缺少的信息。
```

**user**

```text
资料：
本品可抑制巴豆油所致的小鼠耳肿胀和醋酸所致的小鼠腹腔毛细血管通透性增加，可使大鼠胆汁分泌增加，对小鼠热板和醋酸致痛有镇痛作用。

问题：资料指出本品对大鼠胆汁分泌有什么影响？
```

### 当前金标

- answer: 本品可使大鼠胆汁分泌增加。
- required_points: ["本品可使大鼠胆汁分泌增加。"]
- unavailable_points: []
- answerable_part: 
- insufficient_part: 

## qa_176b8cf386ae__rephrase_1 / rephrase_1

- content_hash: `e5750e5cd45ba64f56df2097dc1901c80b118ed691f15cd30b6e38d605cfacac`
- core_question_id: `qfam_343d9bfa5fe239f56bb9`
- expected_action: `answer`
- evidence_state: `sufficient`

### 实际输入

**system**

```text
仅依据提供的资料回答；资料不足时说明缺少的信息。
```

**user**

```text
资料：
本品可抑制巴豆油所致的小鼠耳肿胀和醋酸所致的小鼠腹腔毛细血管通透性增加，可使大鼠胆汁分泌增加，对小鼠热板和醋酸致痛有镇痛作用。

问题：本品会怎样影响大鼠的胆汁分泌？
```

### 当前金标

- answer: 本品可使大鼠胆汁分泌增加。
- required_points: ["本品可使大鼠胆汁分泌增加。"]
- unavailable_points: []
- answerable_part: 
- insufficient_part: 

## qa_176b8cf386ae__rephrase_2 / rephrase_2

- content_hash: `430a87cb223c59a9ff3b2681465be358832215782a23f016041bdc17b3c4ba75`
- core_question_id: `qfam_343d9bfa5fe239f56bb9`
- expected_action: `answer`
- evidence_state: `sufficient`

### 实际输入

**system**

```text
仅依据提供的资料回答；资料不足时说明缺少的信息。
```

**user**

```text
资料：
本品可抑制巴豆油所致的小鼠耳肿胀和醋酸所致的小鼠腹腔毛细血管通透性增加，可使大鼠胆汁分泌增加，对小鼠热板和醋酸致痛有镇痛作用。

问题：大鼠的胆汁分泌在使用本品后有什么变化？
```

### 当前金标

- answer: 本品可使大鼠胆汁分泌增加。
- required_points: ["本品可使大鼠胆汁分泌增加。"]
- unavailable_points: []
- answerable_part: 
- insufficient_part: 

## qa_765fc3535176 / original

- content_hash: `e083f8c8c04a9d01ba073fe64ed6fc39f1a890eb4f082051981867a8fba26ffc`
- core_question_id: `qfam_df01ffd7ec0032507b93`
- expected_action: `answer`
- evidence_state: `sufficient`

### 实际输入

**system**

```text
仅依据提供的资料回答；资料不足时说明缺少的信息。
```

**user**

```text
资料：
1.用于成人和1岁及1岁以上儿童的甲型和乙型流感治疗（磷酸奥司他韦能够有效治疗甲型和乙型流感，但是乙型流感的临床应用数据尚不多）。患者应在首次出现症状48小时以内使用。2.用于成人和13岁及13岁以上青少年的甲型和乙型流感的预防。

问题：根据资料，磷酸奥司他韦用于甲型和乙型流感治疗时，患者应在何时开始使用？
```

### 当前金标

- answer: 患者应在首次出现症状48小时以内使用。
- required_points: ["患者应在首次出现症状48小时以内使用。"]
- unavailable_points: []
- answerable_part: 
- insufficient_part: 

## qa_765fc3535176__rephrase_1 / rephrase_1

- content_hash: `edb7b222793ce55bd7dd893131e7a471d1085535f281f49f6f94d0f0253609e1`
- core_question_id: `qfam_df01ffd7ec0032507b93`
- expected_action: `answer`
- evidence_state: `sufficient`

### 实际输入

**system**

```text
仅依据提供的资料回答；资料不足时说明缺少的信息。
```

**user**

```text
资料：
1.用于成人和1岁及1岁以上儿童的甲型和乙型流感治疗（磷酸奥司他韦能够有效治疗甲型和乙型流感，但是乙型流感的临床应用数据尚不多）。患者应在首次出现症状48小时以内使用。2.用于成人和13岁及13岁以上青少年的甲型和乙型流感的预防。

问题：磷酸奥司他韦治疗甲型流感和乙型流感时，患者应在什么时候开始使用？
```

### 当前金标

- answer: 患者应在首次出现症状48小时以内使用。
- required_points: ["患者应在首次出现症状48小时以内使用。"]
- unavailable_points: []
- answerable_part: 
- insufficient_part: 

## qa_765fc3535176__rephrase_2 / rephrase_2

- content_hash: `2fadf5a030876c8dcfee78c087fea32b50c5c2f32cd1a41aa339d8ea1180d8df`
- core_question_id: `qfam_df01ffd7ec0032507b93`
- expected_action: `answer`
- evidence_state: `sufficient`

### 实际输入

**system**

```text
仅依据提供的资料回答；资料不足时说明缺少的信息。
```

**user**

```text
资料：
1.用于成人和1岁及1岁以上儿童的甲型和乙型流感治疗（磷酸奥司他韦能够有效治疗甲型和乙型流感，但是乙型流感的临床应用数据尚不多）。患者应在首次出现症状48小时以内使用。2.用于成人和13岁及13岁以上青少年的甲型和乙型流感的预防。

问题：治疗甲型或乙型流感时，开始使用磷酸奥司他韦的时间应当是何时？
```

### 当前金标

- answer: 患者应在首次出现症状48小时以内使用。
- required_points: ["患者应在首次出现症状48小时以内使用。"]
- unavailable_points: []
- answerable_part: 
- insufficient_part: 
