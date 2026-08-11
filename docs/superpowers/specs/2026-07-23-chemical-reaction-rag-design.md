# 化学反应 RAG 第一阶段设计

## 目标

在不改变现有 `data/china_textbook_k9_vectordb`、教材 RAG 或数学教学行为的前提下，建立独立的化学反应知识库。它以 `data/示例.xlsx` 的 501 条记录为唯一权威反应数据，以 31 篇 PDF 为可追溯的原文证据，并用 RDKit 而不是 BGE-M3 处理分子结构相似性。

第一阶段的交付是可重复构建、可独立查询、可测试的化学检索能力；它默认不注入普通 Teacher Agent 对话。

## 已核验的数据边界

- `data/示例.xlsx`：501 条数据行、29 个字段、21 个 DOI。每行是一项具体底物—催化剂—条件组合，包含反应物、产物、配体、催化剂 SMILES，以及金属、溶剂、添加剂、温度、压力、S/C、ee、烯烃类型和 DOI。
- `data/数据集初始文献.zip`：31 个 PDF、约 254 页、30 个唯一 DOI。Excel 未覆盖的 9 个 DOI 是综述、Mini Review 或补充研究。
- Excel 是反应字段的权威来源。PDF 不能覆盖、修正或补全 Excel 的反应字段；PDF 的职责是提供正文证据和可定位引用。

## 非目标

- 不用 OpenChemIE 重新解析这批 PDF，也不把其输出与 Excel 做字段融合。
- 不修改既有教材 Chroma 路径、collection、检索阈值或数学 Agent 工作流。
- 不在第一阶段启用用户上传文件、OpenChemIE 多模态处理或自动把新抽取结果写入正式库。
- 不使用 BGE-M3、文本 embedding 或 LLM 判断任何分子结构相似性。

## 架构

所有第一阶段产物位于 `data/chem_reaction_rag/`，与教材库物理隔离：

```text
示例.xlsx ──> 反应规范化 ──> chem_reactions (Chroma)
                    └─────> RDKit SQLite 索引

31 篇 PDF ──> 保页码的文本解析 ──> chem_literature (Chroma)
                         └───────> DOI / 页码证据元数据

查询 ──> 字段过滤 / RDKit 结构检索 / Chroma 文本检索
     └─> 按 DOI 关联反应记录与论文证据
```

### 1. 独立 Chroma 持久目录

使用 `data/chem_reaction_rag/chroma/`，不复用 `data/china_textbook_k9_vectordb`。在该目录中创建两个 collection：

- `chem_reactions`：一条 Excel 行对应一个 document。
- `chem_literature`：一个 PDF 页或其可追溯文本块对应一个 document。

Chroma 的 embedding 仅用于自然语言证据检索。每个 collection 保持来源类型单一，避免结构化实验记录与文献叙述在同一相似度空间互相挤占结果。

### 2. `chem_reactions` 数据模型

每条 document 以可读的键值文本表达完整反应条件，便于回答自然语言问题。metadata 使用 Chroma 可稳定保存的标量字段：

```text
reaction_id, doi, metal, reaction_type, olefin_type,
scaffold_type, multi_scaffold_type, temperature_c,
pressure_atm, ee, coordination_atom_count
```

完整 SMILES、溶剂、添加剂、配体和数值条件保留在 document 与结构化 sidecar 中。`reaction_id` 必须由 Excel 的行标识稳定生成；DOI 必须统一小写、去除 `https://doi.org/` 前缀和首尾空白。

### 3. `chem_literature` 数据模型

PDF 使用保持页码的普通文本解析。每个文档块记录：

```text
literature_chunk_id, doi, source_pdf, page_number, chunk_index,
document_kind="literature"
```

正文块不重新推断反应字段；它们用于回答机理、实验说明、局限性和作者结论，并向用户展示 DOI、PDF 文件名及页码。无法从文件名解析 DOI 的 PDF 保留其文件名，并将 DOI 显式标为缺失而不是猜测。

### 4. RDKit 结构索引

使用 `data/chem_reaction_rag/rdkit_index.sqlite` 保存结构检索 sidecar。每种可检索结构字段建立独立记录：

```text
reaction_id, structure_role, original_smiles, canonical_smiles,
morgan_fingerprint, valid_rdkit
```

`structure_role` 取值为 `reactant`、`product`、`ligand`、`catalyst`。导入时用 RDKit 标准化 SMILES 并记录无效 SMILES；无效记录不进入结构相似度排序，但仍保留在 `chem_reactions` 原始文本中。

RDKit 查询支持：

- 精确 canonical SMILES 匹配；
- Morgan 指纹 Tanimoto 相似度；
- 子结构匹配；
- 指定结构角色的检索，例如“与该配体相似的 Ir 反应”。

结构搜索结果必须返回分数、结构角色、reaction_id 和原始 SMILES，供上层解释。BGE-M3 不参与此路径。

### 5. 查询协调器与隔离原则

提供独立的化学检索接口，而不是修改数学 Graph RAG：

- 结构查询：调用 RDKit 索引，可叠加 `metal`、`olefin_type`、`reaction_type`、温度、压力和 ee 条件过滤。
- DOI 查询：精确返回对应 Excel 反应记录及 `chem_literature` 的论文证据。
- 自然语言查询：只查询 `chem_reactions` 和/或 `chem_literature`；结果按 DOI 关联，但不把向量距离当作结构相似度。

第一阶段接口只由显式化学调用者使用。普通 Teacher Agent 回合仍走现有教材/题库 RAG；化学库可用性、错误或空结果均不得阻断教学流程。

## 构建与可重复性

构建命令必须：

1. 读取指定 Excel 与 PDF ZIP；
2. 生成新的独立目标目录，或要求显式覆盖一个已存在的化学库目录；
3. 输出 manifest，记录输入文件 SHA-256、记录数、DOI 数、无效 SMILES 数、PDF 数、页数、文献块数和构建时间；
4. 对同一输入产生稳定的 reaction_id、literature_chunk_id 和 DOI 归一化结果；
5. 不创建、删除或修改现有教材 Chroma 目录中的文件。

## 质量门与验收标准

第一阶段完成前必须满足：

- Excel 导入数为 501；
- 结构化反应记录的 DOI 归一化后与 Excel 一致；
- `chem_reactions` 与 RDKit 索引中的 reaction_id 可一一关联；
- PDF ZIP 中所有 31 个 PDF 都在 manifest 中有成功或明确失败状态；
- 文献结果可以按 DOI 和页码追溯到原 PDF；
- RDKit 精确、相似度、子结构三类查询在测试中验证；
- 检索测试证明 BGE-M3 未参与结构相似度排序；
- 现有 `python -m pytest` 与 `python scripts/run_eval.py` 均保持通过。

## 第二阶段边界：OpenChemIE

OpenChemIE 只用于未来的化学知识预处理：重新处理现有化学教材，或处理用户新上传的 PDF、图片、表格和反应图。其输出应先进入带来源、页码/区域、模型置信度和人工审核状态的候选库；不得自动写入本第一阶段的权威反应库。

在 Agent 层，未来的多模态路由应仅在文件被判断为化学材料、常规文本/视觉理解不足且用户明确请求分析时调用 OpenChemIE。调用失败必须降级为现有文件文本/视觉处理，并将“化学抽取不可用”作为非阻断状态返回。
