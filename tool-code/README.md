# 暖气片型号识别系统

这是一个基于 FastAPI 的暖气片/产品型号识别系统。系统分为用户端和管理端：

- 用户端通过决策流程逐步输入产品维度，支持在流程中随时上传一张或多张图片进行图片理解，并把识别结果作为后续步骤的建议值。
- 管理端负责上传 Excel 数据、配置维度、训练决策树模型、查看模型效果、配置流程步骤 UI、配置图片理解字段，并切换用户端当前使用的模型。

## 当前功能

- Excel 数据上传与维度识别
- 维度名称、展示名称、类型、权重、默认启用状态配置
- 基于决策树的模型训练、重训、删除、启用
- 不同训练结果独立保存到 `models/<model_id>/`
- Accuracy、Coverage 和未命中明细展示
- 决策流程图总览，支持查看任意模型的流程图
- 每个流程步骤可配置输入方式、说明图、帮助文案、是否参与图片理解
- 用户端流程输入、上一步、跳过、置信度达标提前结束
- 用户端流程中悬浮图片理解入口
- 多图片上传、图片预处理、Qwen 图片理解流式返回、结果表单确认
- 图片理解结果只会填充未完成步骤；已完成步骤保留用户输入，同时显示建议值
- 图片理解详细日志写入 `logs/vision_debug.log`

## 技术栈

- 后端：FastAPI + Uvicorn
- 模板：Jinja2
- 数据处理：Pandas + OpenPyXL
- 图片处理：Pillow
- 图片理解：OpenAI SDK compatible mode + DashScope/Qwen
- 模型：基于信息增益的决策树
- 依赖管理：uv

## 快速启动

安装依赖：

```bash
uv sync
```

启动服务：

```bash
uv run uvicorn main:app --host 127.0.0.1 --port 8000
```

访问页面：

```text
用户端：http://127.0.0.1:8000
管理端：http://127.0.0.1:8000/admin
```

如果 8000 端口被占用，可以先查看并结束占用进程：

```powershell
netstat -ano | findstr :8000
taskkill /PID <PID> /F
```

## 项目结构

```text
.
├── main.py                    # FastAPI 应用装配，只负责注册路由和静态资源
├── admin_routes.py            # 管理端路由：训练、重训、模型管理、流程配置、图片资源
├── user_routes.py             # 用户端流程路由：首页、开始、回答、跳过、返回
├── vision_routes.py           # 图片理解路由：上传页、流式识别、结果确认
├── flowchart_routes.py        # 决策流程图路由
├── flow_views.py              # 用户端下一步/结果页渲染
├── web_templates.py           # Jinja2 模板引擎
├── vision_workflow.py         # 图片理解任务状态、SSE、上传预处理入口
├── app_config.py              # 路径、运行配置、Qwen 配置、JSON 工具
├── app_logging.py             # 图片理解日志
├── decision_tree.py           # Excel 加载、维度推断、决策树训练、评测、流程图数据
├── dimension_defaults.py      # 维度默认配置
├── flow_runtime.py            # 用户端识别会话、候选集、跳过、返回、图片建议值合并
├── model_store.py             # 模型保存、加载、启用、删除、重训配置
├── vision_image_limits.py     # 图片数量/尺寸/体积限制与压缩
├── vision_service.py          # Qwen 图片理解提示词、调用、流式解析
├── templates/                 # 页面模板
├── static/                    # 静态资源
├── models/                    # 运行时模型目录，默认不纳入版本管理
├── uploads/                   # 运行时上传目录，默认不纳入版本管理
└── logs/                      # 运行时日志目录，默认不纳入版本管理
```

## 数据格式

管理端上传的 Excel 文件应类似 `output.xlsx`：

- 第一列作为目标分类字段，也就是最终要识别的型号、产品 ID 或 SKU。
- 其他列作为可参与识别的维度。
- 空值会统一填充为 `MISSING`。
- 如果数据中包含 `Plain`、`Flat top`、`Scroll`、`Round top` 四列，系统会自动派生 `Top Style` 字段。

示例：

| Products ID | Name | Castrads SKU | Section Length (mm) | Leg Section Depth (mm) |
| --- | --- | --- | --- | --- |
| 637 | Product A | SKU-A | 80 | 165 |
| 640 | Product B | SKU-B | 100 | 185 |

## 管理端流程

1. 进入 `http://127.0.0.1:8000/admin`。
2. 上传 Excel 文件。
3. 在训练前配置维度：
   - `Use`：是否参与训练。
   - `Name` / `Display Name`：原始字段名和用户端展示名。
   - `Type`：`Numeric` 或 `Categorical`。
   - `Weight`：人工权重，训练时使用 `information_gain * weight` 选择分裂维度。
   - `Ease`：测量难度，来自维度默认配置。
   - `Measurement Comment`：测量难度说明，训练前配置页展示。
   - `Image Description`：图片理解使用的字段描述。
4. 点击训练后生成独立模型目录。
5. 在模型列表中可以查看详情、重训、删除、启用、配置流程、查看流程图。

## 维度默认配置

管理端 `Dimension Defaults` 用于维护导入新数据时的维度默认值。

规则：

- 所有维度默认 `weight = 1`。
- `Ease` 为 `N/A`、`Low`、`Very Low` 的维度默认不选。
- `Ease` 为 `High` 的维度默认 `weight = 2`。
- `Measurement Comment` 用于说明测量难度。
- `Image Description` 用于生成图片理解提示词。

## 决策树逻辑

训练时会从启用的维度中选择最优分裂字段。

分类维度：

- 按不同取值分支。
- 使用信息增益评估分裂效果。

数值维度：

- 遍历相邻数值之间的候选阈值。
- 选择信息增益最高的阈值。
- 形成 `<= threshold` 和 `> threshold` 两个分支。

最终分裂得分：

```text
score = information_gain * weight
```

因此 `Weight` 不是系统自动算出的置信度，而是管理端给决策树的人工偏好权重。

## Accuracy 和 Coverage

训练完成后会生成 `metrics.json`。

- `Accuracy`：在被模型成功命中的样本中，预测正确的比例。
- `Coverage`：训练数据中能被当前决策树走到有效预测结果的比例。
- 未命中明细会记录哪些训练行没有被有效预测，方便判断是数据缺失、维度不足还是流程分裂不够。

这些指标目前用于训练集上的快速评估，不等同于独立测试集效果。

## 用户端流程

用户从首页点击 `Start Identification` 后进入流程。

每一步会展示：

- 当前模型
- 当前置信度
- 当前候选数量
- 当前问题
- 说明图和帮助文案
- 输入控件
- 图片理解建议值
- 上一步、跳过、提交

输入控件由管理端 `Configure Flow` 决定：

- `Auto`：按决策树节点自动选择。
- `Number`：数值输入框。
- `Text`：文本输入框。
- `Select`：下拉菜单。

当候选集中最高置信度达到模型阈值时，流程会提前结束并展示结果。

## 跳过逻辑

点击 `Skip` 会：

- 不筛选当前候选集。
- 将当前维度标记为已跳过。
- 后续不再询问该维度。
- 快速寻找下一个可用维度，避免重新训练或重建完整决策树。

## 图片理解

图片理解入口位于用户端流程页右侧悬浮按钮。点击后会进入独立上传页：

- 可以一次选择一张图片。
- 可以一次选择多张图片。
- 可以分多次追加图片。
- 页面会预览已选择图片。
- 取消会返回原流程。
- 确认后进入图片理解加载页。

加载页会通过 SSE 流式展示模型返回内容。模型返回完整 JSON 后，系统会展示结果表单，用户可以修改后确认。

确认后：

- 未完成步骤会使用图片理解结果作为默认值。
- 已完成步骤不会被覆盖。
- 已完成步骤仍会显示图片理解建议，便于人工对比。

## 图片理解提示词

通用模板在 `vision_service.py` 的 `GENERIC_IMAGE_PROMPT_TEMPLATE` 中。

结构化字段由 `build_image_prompt(model)` 动态注入，来源包括：

- `Configure Flow` 中勾选了图片理解的字段。
- 维度类型：数值或分类。
- 字段展示名。
- `Image Description`。
- 步骤输入方式。
- 步骤帮助文案。
- 决策树阈值。
- 分类字段可选值。

模型被要求只返回 JSON：

```json
{
  "values": {
    "Field Name": "value or null"
  },
  "notes": "short note"
}
```

## Qwen 配置

图片理解使用 OpenAI SDK 的 compatible mode 调用 DashScope/Qwen。

相关配置在 `app_config.py`：

```python
QWEN_API_KEY = ""
QWEN_API_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
QWEN_VISION_MODEL = "qwen3.6-plus"
```

也可以不在代码中写 key，改用环境变量：

```text
DASHSCOPE_API_KEY
QWEN_API_KEY
```

## 图片限制和日志

图片上传后会先经过 `vision_image_limits.py` 预处理：

- 最多 250 张图片。
- 最短边不少于 11 像素。
- 长宽比不超过 200:1。
- 单图压缩到约 2,000,000 像素以内。
- Base64 后大小控制在 10 MB 以内。
- 非 JPEG 图片会转换成 JPEG。

图片理解链路日志写入：

```text
logs/vision_debug.log
```

日志会记录上传、压缩、编码、提示词生成、请求创建、首个流式片段、返回结束、JSON 解析等步骤，用于定位卡顿。

## 模型目录格式

每次训练都会生成：

```text
models/<model_id>/
├── config.json      # 维度配置、流程 UI、图片理解配置、阈值
├── data.xlsx        # 本次训练使用的数据副本
├── metadata.json    # 模型名称、创建时间、来源文件、行列数
├── metrics.json     # Accuracy、Coverage、未命中明细
└── tree.json        # 决策树结构
```

这里的 `models/` 是应用自己的模型版本目录，不是 sklearn、ONNX、pickle 这类标准机器学习模型格式。模型本体是可读的 JSON 决策树。

## 开发说明

常用检查：

```bash
python -m py_compile main.py admin_routes.py user_routes.py vision_routes.py flowchart_routes.py flow_views.py vision_workflow.py web_templates.py app_config.py app_logging.py decision_tree.py dimension_defaults.py flow_runtime.py model_store.py vision_image_limits.py vision_service.py
```

路由层现在按职责拆分：

- `main.py`：应用生命周期、静态资源、路由注册。
- `admin_routes.py`：后台管理页面和表单。
- `user_routes.py`：用户端识别流程。
- `vision_routes.py`：图片理解上传、流式返回、确认。
- `flowchart_routes.py`：流程图页面。

业务逻辑保持在服务模块中：

- 决策树：`decision_tree.py`
- 模型版本：`model_store.py`
- 流程状态：`flow_runtime.py`
- 图片理解：`vision_service.py`
- 图片限制：`vision_image_limits.py`
- 维度默认值：`dimension_defaults.py`

## 运行时文件

以下内容由系统运行时生成，默认不纳入版本管理：

- `.venv/`
- `active_model.json`
- `models/`
- `uploads/`
- `static/model_assets/`
- `logs/`
- `dimension_defaults.json`

如果需要把项目发给别人，通常只打包源码、模板、静态默认图片、`pyproject.toml`、`uv.lock` 和 `README.md` 即可。

## 已知限制

- 用户识别会话和图片理解任务当前仍是服务端内存状态，更适合本地单人演示；多人并发建议接入数据库或正式 session。
- 模型评估目前基于训练集快速评测，后续可以增加训练集/验证集拆分。
- 图片理解依赖外部 Qwen 服务，速度受图片数量、图片尺寸、网络和模型响应影响。
- API Key 可以按当前需求写在代码中，但生产环境更建议使用环境变量或密钥管理服务。
