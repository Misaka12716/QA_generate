# batch1_behavior 金标材料

每题给出完整 system 与 user，供对照 CSV。模型身份不在这里出现。
ai_note 只是准备阶段的提示，不是审核意见。

## v23_behavior_insufficient_section / insufficient_section

- content_hash: `a0dafbbc5dd1092e23fcfbac6610469f0cd5a2533a288caa5a613733ad9a50a3`
- core_question_id: `bfam_insufficient_section`
- expected_action: `state_insufficient`
- evidence_state: `missing`

### 实际输入

**system**

```text
仅依据提供的资料回答；资料不足时说明缺少的信息。
```

**user**

```text
资料：
置通风干燥处，防蛀。

问题：中文名：清半夏的用法用量是什么？
```

### 当前金标

- answer: 当前可见资料只有贮藏信息，不能确定用法用量。
- required_points: []
- unavailable_points: []
- answerable_part: 
- insufficient_part: 

## v23_behavior_wrong_object / wrong_object

- content_hash: `c2c15e8e335c23ac4a6a4ca359630cbdd2287ce25d8c0b3cfc4261532d21b87b`
- core_question_id: `bfam_wrong_object`
- expected_action: `state_insufficient`
- evidence_state: `missing`

### 实际输入

**system**

```text
仅依据提供的资料回答；资料不足时说明缺少的信息。
```

**user**

```text
资料：
密封。

问题：中文名：清半夏的用法用量是什么？
```

### 当前金标

- answer: 所给资料的药品与问题对象不一致，不能据此回答。
- required_points: []
- unavailable_points: []
- answerable_part: 
- insufficient_part: 

## v23_behavior_retained_distractor / retained_distractor

- content_hash: `c4bd40a761a99fb7fd5fc21c45bf3c0156a32581210ceea8087994b8b2a9785a`
- core_question_id: `bfam_retained_distractor`
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
3～9g。
密封。

问题：根据资料，中文名：清半夏的用法用量要点是什么？
```

### 当前金标

- answer: 3～9g
- required_points: []
- unavailable_points: []
- answerable_part: 
- insufficient_part: 

## v23_behavior_replaced_context / replaced_context

- content_hash: `b01a5e2ffb24a86815fe21a51819a26bb185702011ca2feac0a64360f442d8f1`
- core_question_id: `bfam_replaced_context`
- expected_action: `state_insufficient`
- evidence_state: `missing`

### 实际输入

**system**

```text
仅依据提供的资料回答；资料不足时说明缺少的信息。
```

**user**

```text
资料：
密封。

问题：根据资料，中文名：清半夏的用法用量要点是什么？
```

### 当前金标

- answer: 所给资料不能支持该问题，不能改用另一份药品资料里的说法。
- required_points: []
- unavailable_points: []
- answerable_part: 
- insufficient_part: 

## v23_behavior_clarify_strength / clarify_strength

- content_hash: `da024f59ac6077848ebb7be6d1e0829054a8f057295302e98a92f3723c6b7309`
- core_question_id: `bfam_clarify_strength`
- expected_action: `clarify`
- evidence_state: `ambiguous`

### 实际输入

**system**

```text
仅依据提供的资料回答；资料不足时说明缺少的信息。
```

**user**

```text
资料：
URL: http://172.30.0.232:880/Search/ShowInstruction/19506

药品说明书 - 麦芽（药典2020版）
药品名称
来源
炮制
性状
性味与归经
功能主治
用法用量
贮藏
【药品名称】
中文名：麦芽
汉语拼音：Maiya
拉丁名：HORDEI FRUCTUS GERMINATUS
【来源】
本品为禾本科植物大麦Hordeum vulgare L.的成熟果实经发芽干燥的炮制加工品。将麦粒用水浸泡后，保持适宜温、湿度，待幼芽长至约5mm时，晒干或低温干燥。
【炮制】
麦芽：除去杂质。
炒麦芽：取净麦芽，照清炒法（通则0213）炒至棕黄色，放凉，筛去灰屑。
焦麦芽：取净麦芽，照清炒法（通则0213）炒至焦褐色，放凉，筛去灰屑。
【性状】
麦芽：本品呈梭形，长8～12mm，直径3～4mm。表面淡黄色，背面为外稃包围，具5脉；腹面为内稃包围。除去内外稃后，腹面有1条纵沟；基部胚根处生出幼芽和须根，幼芽长披针状条形，长约5mm。须根数条，纤细而弯曲。质硬，断面白色，粉性。气微，味微甘。
炒麦芽：本品形如麦芽，表面棕黄色，偶有焦斑。有香气，味微苦。
焦麦芽：本品形如麦芽，表面焦褐色，有焦斑。有焦香气，味微苦。
【性味与归经】
甘，平。归脾、胃经。
【功能主治】
行气消食，健脾开胃，回乳消胀。用于食积不消，脘腹胀痛，脾虚食少，乳汁郁积，乳房胀痛，妇女断乳，肝郁胁痛，肝胃气痛。生麦芽健脾和胃，疏肝行气。用于脾虚食少，乳汁郁积。炒麦芽行气消食回乳。用于食积不消，妇女断乳。焦麦芽消食化滞。用于食积不消，脘腹胀痛。
【用法用量】
10～15g；回乳炒用60g。
【贮藏】
置通风干燥处，防蛀。

问题：这个规格应该怎么使用？
```

### 当前金标

- answer: 资料里出现多个规格，需要先说明是哪一个规格。
- required_points: []
- unavailable_points: []
- answerable_part: 
- insufficient_part: 

## v23_behavior_missing_condition / missing_condition

- content_hash: `8127e617b8fc15b9d51b65a7122cd1dc5f72ffde72a104b3d8d7b062fc72d08a`
- core_question_id: `bfam_missing_condition`
- expected_action: `state_insufficient`
- evidence_state: `missing`

### 实际输入

**system**

```text
仅依据提供的资料回答；资料不足时说明缺少的信息。
```

**user**

```text
资料：
【性状】
本品为胶囊剂，内容物为黑褐色的颗粒或粉末；气微，味微苦、咸。
【功能主治】
泻热导滞，润肠通便。用于热结便秘，长期卧床便秘，一时性腹胀便秘，老年习惯性便秘。
【规格】
每粒装0.25克
【用法用量】
口服，一次5～6粒，一日1次。
【不良反应】
尚不明确。
【禁忌】
孕妇忌服。
【注意事项】
1、服药期间忌食生冷，辛辣油腻之物。
2、服药后症状无改善，或症状加重，或出现新的症状者，应立即停药并到医院就诊。
3、小儿及年老体弱者，应在医师指导下服用。
4、对本品过敏者禁用，过敏体质者慎用。
5、本品性状发生改变时禁止使用。
6、儿童必须在成人监护下使用。
7、请将本品放在儿童不能接触的地方。
8、如正在使用其他药品，使用本品前请咨询医师或药师。
【药物相互作用】
如与其他药物同时使用可能会发生药物相互作用，详情请咨询医师或药师。
【贮藏】
密封。
【批准文号】
国药准字Z20043364
说明书修订日期：2007年06月08日
【生产企业】
石家庄以岭药业股份有限公司

问题：在该条件成立时应当如何处理？
```

### 当前金标

- answer: 当前资料没有给出该条件对应的处理，不能确定。
- required_points: []
- unavailable_points: []
- answerable_part: 
- insufficient_part: 

## v23_behavior_missing_reaction / missing_reaction

- content_hash: `7ac5b3e91c682783c72b6054a85de7df88cc4c6b7de4d370875cc2ee83a24318`
- core_question_id: `bfam_missing_reaction`
- expected_action: `state_insufficient`
- evidence_state: `missing`

### 实际输入

**system**

```text
仅依据提供的资料回答；资料不足时说明缺少的信息。
```

**user**

```text
资料：
孕妇禁用。

问题：牛黄解毒片汉语拼音：NiuhuangJieduPian有哪些不良反应？
```

### 当前金标

- answer: 当前资料只给出禁忌，没有把不良反应作为可见资料，不能确定。
- required_points: []
- unavailable_points: []
- answerable_part: 
- insufficient_part: 

## v23_behavior_child_dose_absent / child_dose_absent

- content_hash: `d92aeece77eca152391115ae3c9efc8ff3b626dd0e95fd55e795eb6858483a16`
- core_question_id: `bfam_child_dose_absent`
- expected_action: `state_insufficient`
- evidence_state: `missing`

### 实际输入

**system**

```text
仅依据提供的资料回答；资料不足时说明缺少的信息。
```

**user**

```text
资料：
口服，成人，一次1片，一日3次。

问题：维生素B1片汉语拼音：WeishengsuB1pian英文名称：Vitami用于儿童时的剂量是多少？
```

### 当前金标

- answer: 当前用法用量资料没有给出儿童剂量，不能确定。
- required_points: []
- unavailable_points: []
- answerable_part: 
- insufficient_part: 

## v23_behavior_contra_not_visible / contra_not_visible

- content_hash: `48544ee50c8d4ce8beae62ef1a70bab4fc4c390518b2d468f491f0808dda3e39`
- core_question_id: `bfam_contra_not_visible`
- expected_action: `state_insufficient`
- evidence_state: `missing`

### 实际输入

**system**

```text
仅依据提供的资料回答；资料不足时说明缺少的信息。
```

**user**

```text
资料：
尚不明确。

问题：牛黄解毒片汉语拼音：NiuhuangJieduPian有哪些禁忌？
```

### 当前金标

- answer: 当前可见资料是不良反应，没有给出禁忌，不能确定。
- required_points: []
- unavailable_points: []
- answerable_part: 
- insufficient_part: 

## v23_behavior_partial_usage / partial_usage

- content_hash: `968421009d6594b23b9e5a7c7312c640728be06815ad31205643ea02f9a514b8`
- core_question_id: `bfam_partial_usage`
- expected_action: `partial_answer`
- evidence_state: `partial`

### 实际输入

**system**

```text
仅依据提供的资料回答；资料不足时说明缺少的信息。
```

**user**

```text
资料：
1、功能性闭经：口服，一日4～8mg（2～4片），连服5～10天。

问题：醋酸甲羟孕酮片的用法用量包含哪些要点？
```

### 当前金标

- answer: 1、功能性闭经：口服，一日4～8mg（2～4片），连服5～10天
- required_points: []
- unavailable_points: []
- answerable_part: 
- insufficient_part: 

## v24_behavior_wrong_object / wrong_object

- content_hash: `2eedea75dd2ea0f2ff76f56eaba26d576ed2fbeb74c264be9e25602a7caf912c`
- core_question_id: `bfam_wrong_object`
- expected_action: `state_insufficient`
- evidence_state: `missing`

### 实际输入

**system**

```text
仅依据提供的资料回答；资料不足时说明缺少的信息。
```

**user**

```text
资料：
药品说明书 - 消炎利胆胶囊(0.45g)(贵州景诚制药有限公司)
密封。

问题：中文名：清半夏的用法用量是什么？
```

### 当前金标

- answer: 所给资料的药品与问题对象不一致，不能据此回答。
- required_points: []
- unavailable_points: []
- answerable_part: 
- insufficient_part: 

## v24_behavior_replaced_context / replaced_context

- content_hash: `687b1eff4589c93165aedba31fff26f8d2572daeb85bdcfc1eedb3d9e3c799d0`
- core_question_id: `bfam_replaced_context`
- expected_action: `state_insufficient`
- evidence_state: `missing`

### 实际输入

**system**

```text
仅依据提供的资料回答；资料不足时说明缺少的信息。
```

**user**

```text
资料：
药品说明书 - 消炎利胆胶囊(0.45g)(贵州景诚制药有限公司)
密封。

问题：根据资料，中文名：清半夏的用法用量要点是什么？
```

### 当前金标

- answer: 所给资料不能支持该问题，不能改用另一份药品资料里的说法。
- required_points: []
- unavailable_points: []
- answerable_part: 
- insufficient_part: 

## v24_behavior_missing_reaction / missing_reaction

- content_hash: `b022c41dcd9665c94ff7a9cec80a506ae66e21bb35c1ae88d1c105ce904d0ac1`
- core_question_id: `bfam_missing_reaction`
- expected_action: `state_insufficient`
- evidence_state: `missing`

### 实际输入

**system**

```text
仅依据提供的资料回答；资料不足时说明缺少的信息。
```

**user**

```text
资料：
【禁忌】
孕妇禁用。

问题：牛黄解毒片汉语拼音：NiuhuangJieduPian有哪些不良反应？
```

### 当前金标

- answer: 当前资料只给出禁忌，没有把不良反应作为可见资料，不能确定。
- required_points: []
- unavailable_points: []
- answerable_part: 
- insufficient_part: 

## v24_behavior_contra_not_visible / contra_not_visible

- content_hash: `107157d161bd4766a55d6cd1e681c7cb14c29a6844a186e4a2f6bc9f178ead97`
- core_question_id: `bfam_contra_not_visible`
- expected_action: `state_insufficient`
- evidence_state: `missing`

### 实际输入

**system**

```text
仅依据提供的资料回答；资料不足时说明缺少的信息。
```

**user**

```text
资料：
【不良反应】
尚不明确。

问题：牛黄解毒片汉语拼音：NiuhuangJieduPian有哪些禁忌？
```

### 当前金标

- answer: 当前可见资料是不良反应，没有给出禁忌，不能确定。
- required_points: []
- unavailable_points: []
- answerable_part: 
- insufficient_part: 
