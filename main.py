from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import pandas as pd
import numpy as np
from collections import Counter

app = FastAPI()

# ✅ 模板引擎（避免污染）
template_engine = Jinja2Templates(directory="templates")

# ======================
# 1. 数据加载
# ======================
df = pd.read_excel("output.xlsx", dtype=str)
df = df.fillna("MISSING")

MODEL_COL = df.columns[0]
EXCLUDE_COLS = list(df.columns[:3]) + [df.columns[5]]
FEATURES = [c for c in df.columns if c not in EXCLUDE_COLS]

# ======================
# 2. 决策树
# ======================
def entropy(labels):
    total = len(labels)
    counter = Counter(labels)
    return -sum((c/total) * np.log2(c/total) for c in counter.values())

def info_gain(df, feature, target):
    base = entropy(df[target])
    cond = 0
    for v in df[feature].unique():
        sub = df[df[feature] == v]
        cond += len(sub)/len(df) * entropy(sub[target])
    return base - cond

class TreeNode:
    def __init__(self, feature=None, label=None):
        self.feature = feature
        self.label = label
        self.children = {}

def build_tree(df, features):
    labels = df[MODEL_COL]

    # 终止条件
    if len(set(labels)) == 1:
        return TreeNode(label=labels.iloc[0])

    if not features:
        return TreeNode(label=Counter(labels).most_common(1)[0][0])

    best = max(features, key=lambda f: info_gain(df, f, MODEL_COL))
    node = TreeNode(feature=best)

    for v in df[best].unique():
        sub = df[df[best] == v]
        remaining = [f for f in features if f != best]
        node.children[v] = build_tree(sub, remaining)

    return node

print("开始构建树...")
tree = build_tree(df, FEATURES)
print("树构建完成")

# ======================
# 3. 状态
# ======================
session_state = {
    "node": tree,
    "data": df.copy(),
    "history": []
}

# ======================
# 4. 页面路由
# ======================

@app.get("/", response_class=HTMLResponse)
def start(request: Request):
    return template_engine.TemplateResponse(
        request=request,
        name="start.html",
        context={}
    )


@app.post("/start", response_class=HTMLResponse)
def start_flow(request: Request):
    session_state["node"] = tree
    session_state["data"] = df.copy()
    session_state["history"] = []
    return next_question(request)


def next_question(request: Request):
    try:
        node = session_state["node"]
        current_df = session_state["data"]

        if node is None:
            return HTMLResponse("<h2>❌ 匹配失败</h2>")

        # 到叶子
        if node.label:
            # 获取完整的产品信息
            product_row = current_df[current_df[MODEL_COL] == node.label].iloc[0]
            product_info = {}
            for col in df.columns:
                val = product_row[col]
                if val != "MISSING":
                    product_info[col] = val

            return template_engine.TemplateResponse(
                request=request,
                name="result.html",
                context={
                    "result": node.label,
                    "product_info": product_info
                }
            )

        feature = node.feature

        # ===== 排序（数值优先 + 文本 + MISSING最后）=====
        s = current_df[feature]
        num = pd.to_numeric(s, errors="coerce")

        df_sort = pd.DataFrame({"val": s, "num": num})
        df_sort["is_missing"] = df_sort["val"] == "MISSING"
        df_sort["is_text"] = df_sort["num"].isna()

        df_sort = df_sort.sort_values(
            by=["is_missing", "is_text", "num", "val"]
        )

        options = df_sort["val"].drop_duplicates().tolist()

        return template_engine.TemplateResponse(
            request=request,
            name="question.html",
            context={
                "feature": feature,
                "options": options
            }
        )

    except Exception as e:
        return HTMLResponse(f"<h1>错误：</h1><pre>{str(e)}</pre>")


@app.post("/answer", response_class=HTMLResponse)
def answer(request: Request, value: str = Form(...)):
    node = session_state["node"]

    if node is None:
        return HTMLResponse("<h2>❌ 当前节点为空</h2>")

    feature = node.feature

    session_state["history"].append((feature, value))

    # 过滤数据
    session_state["data"] = session_state["data"][
        session_state["data"][feature] == value
    ]

    # 走树
    session_state["node"] = node.children.get(value)

    return next_question(request)


@app.post("/back", response_class=HTMLResponse)
def back(request: Request):
    if not session_state["history"]:
        return next_question(request)

    session_state["history"].pop()

    node = tree
    data = df.copy()

    for f, v in session_state["history"]:
        data = data[data[f] == v]
        node = node.children.get(v)
        if node is None:
            break

    session_state["node"] = node
    session_state["data"] = data

    return next_question(request)


# ======================
# 5. 决策树流程图
# ======================
def tree_to_flowchart(node):
    """将决策树转换为流程图数据"""
    nodes = []
    edges = []
    node_id_counter = [0]

    def traverse(n):
        current_id = f"node_{node_id_counter[0]}"
        node_id_counter[0] += 1

        if n.label:
            nodes.append({
                "id": current_id,
                "type": "result",
                "text": n.label
            })
        else:
            nodes.append({
                "id": current_id,
                "type": "decision",
                "text": n.feature
            })

            for value, child in n.children.items():
                child_id = traverse(child)
                edges.append({
                    "from": current_id,
                    "to": child_id,
                    "label": value
                })

        return current_id

    traverse(node)
    return nodes, edges


@app.get("/flowchart", response_class=HTMLResponse)
def flowchart(request: Request):
    nodes, edges = tree_to_flowchart(tree)
    print(f"Flowchart data: {len(nodes)} nodes, {len(edges)} edges")
    return template_engine.TemplateResponse(
        request=request,
        name="flowchart.html",
        context={
            "nodes": nodes,
            "edges": edges
        }
    )


# ======================
# 6. 启动
# ======================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)