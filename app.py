import os, json, streamlit as st
from openai import OpenAI, RequestsTransport
from utils.diagnose import diagnose

# ---------- 代理（可选） ----------
# 如果你本机有 Clash / V2RayN 等本地 HTTP 代理，就把 PROXY_URL 改成对应端口；
# 没有代理或能直连 OpenAI，就留空字符串 ""。
PROXY_URL = "http://127.0.0.1:7890"    # ← 改成自己的 HTTP 代理端口，没用就设为 ""

_http_client = RequestsHTTPClient(
    proxies={"http": PROXY_URL, "https": PROXY_URL} if PROXY_URL else None
)

# ---------- OpenAI 客户端 ----------
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    http_client=_http_client,
    timeout=60,
    max_retries=3,
)

# ---------- Streamlit 页面 ----------
st.set_page_config(page_title="人生脚本·动态提问", layout="centered")
st.title("人生脚本探索 Demo 🌀")

MAX_Q = 30

SYSTEM_PROMPT = """
你是基于埃里克·伯恩《人生脚本》理论的脚本探索 AI，目标是在一次对话里
生成并提问 3–30 个问题，引导用户描述
① 童年禁止令 / 驱动力 ② 生活立场 ③ 脚本蓝图。

【规则】
1. 每次仅返回 ONE question，格式 JSON：{{"question": "这里写问题"}}。
2. 问题用中文，具体生动，避免心理学行话。
3. 不重复主题，可根据用户上一次回答深挖细节。
4. 若已达到 {max_q} 题，返回 {{"question": "__END__"}}。
""".strip()

# ---------- 初始化会话状态 ----------
if "history" not in st.session_state:
    st.session_state.history = [
        {"role": "assistant", "content": "你好！我是脚本探索伙伴 ECHO，现在我们开始。"}
    ]
    st.session_state.q_count = 0

# ---------- 显示历史 ----------
for m in st.session_state.history:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# ---------- 输入框 ----------
user_text = st.chat_input(
    "请输入…",
    disabled=st.session_state.q_count >= MAX_Q
)

# ---------- 处理用户输入 ----------
if user_text:
    # 显示并保存用户消息
    st.chat_message("user").markdown(user_text)
    st.session_state.history.append({"role": "user", "content": user_text})

    # 组织上下文：system 提示 + 最近 12 条对话
    ctx = [{"role": "system", "content": SYSTEM_PROMPT.format(max_q=MAX_Q)}]
    ctx += st.session_state.history[-12:]

    # 调 OpenAI 拿下一题
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=ctx,
            temperature=0.7,
        )
    except APIConnectionError as e:
        st.error("🚧 无法连接 OpenAI，检查网络/代理后重试。\n\n" + str(e))
        st.stop()

    raw = resp.choices[0].message.content.strip()

    # 解析
    try:
        next_q = json.loads(raw)["question"].strip()
    except Exception:
        next_q = "__PARSE_ERROR__"

    # 处理结果
    if next_q and next_q not in ("__END__", "__PARSE_ERROR__"):
        st.session_state.q_count += 1
        st.chat_message("assistant").markdown(next_q)
        st.session_state.history.append({"role": "assistant", "content": next_q})
    else:
        st.session_state.q_count = MAX_Q
        st.session_state.history.append(
            {"role": "assistant",
             "content": "感谢回答，问题结束！点击下方按钮生成初步报告。"}
        )
    st.rerun()

# ---------- 生成报告 ----------
if st.session_state.q_count >= MAX_Q:
    if st.button("生成报告"):
        answers = [m for m in st.session_state.history if m["role"] == "user"]
        # mock 映射键名方便示例 diagnose
        fake_pairs = [{"q": "inj_x", "a": a["content"]} for a in answers]
        data = diagnose(fake_pairs)
        st.markdown(f"""
### 初步脚本报告
**脚本倾向**：{data['summary']}  
Injunction 线索：{data['inj_cnt']}  
Driver 线索：{data['driver_cnt']}  

*(仅供探索，非专业诊断)*
""")
