import os, json, streamlit as st
from openai import OpenAI, APIConnectionError

# ---------- 代理（本地用；云端留空） ----------
PROXY_URL = os.getenv("PROXY_URL", "")           # 本地自己 setx；云端不设

# ---------- OpenAI 客户端 ----------
if PROXY_URL:
    from openai import RequestsTransport
    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        transport=RequestsTransport(
            proxies={"http": PROXY_URL, "https": PROXY_URL}
        ),
        timeout=60,
        max_retries=3,
    )
else:
    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        timeout=60,
        max_retries=3,
    )

# ---------- Streamlit 页面 ----------
st.set_page_config(page_title="人生脚本·动态提问", layout="centered")
st.title("人生脚本探索 Demo 🌀")

MAX_Q = 30

# === 你的 JSON Prompt（保持原样，不做 .format 替换） ===
SYSTEM_PROMPT = r"""
{
  "prompt_definition": {
    "overall_goal": "作为人生脚本探索AI助手，与用户进行多轮对话，严格按顺序引导用户回答完所有预设问题，并在最后直接生成初步探索报告。整个过程需友好、自然，并确保对话不偏离主题。",
    "input_variables": [
      { "name": "interaction_phase", "description": "当前交互阶段 ('initial_greeting', 'conversation_turn', 'final_report')" },
      { "name": "current_question_index", "description": "当前问题索引 (从1开始计数，仅在 'conversation_turn' 阶段需要)" },
      { "name": "user_input", "description": "用户的上一句输入 (在 'conversation_turn' 和 'final_report' 阶段需要)" },
      { "name": "conversation_history", "description": "最近几轮的对话历史 (用于 'conversation_turn' 判断跑题)" },
      { "name": "full_conversation_transcript", "description": "完整的对话记录 (仅在 'final_report' 阶段需要)" },
      { "name": "total_questions", "description": "预设问题的总数" },
      { "name": "estimated_time", "description": "预计完成对话所需时间 (例如 '5-8分钟')" }
    ],
    "question_list": [
      "在您长大的过程中，家里（或者对您影响最大的人）经常强调的‘规矩’或者‘期望’是什么？（比如，要听话、要独立、要让家人骄傲等等）",
      "回想一下您大概上小学的时候，您觉得自己是个什么样的孩子？（可以说说性格、优点或缺点）",
      "那时候，您感觉周围的大人（比如父母、老师）怎么样？他们对您好吗？",
      "有没有什么事情，是您从小就觉得‘千万不能做’或者‘最好不要提’的？（比如：不能犯错、不能软弱、不能比别人差、不能表达需要？）",
      "为了得到表扬或者避免麻烦，您觉得必须做到哪些事情？（比如：必须努力、必须懂事、必须讨人喜欢？）",
      "小时候，您最喜欢听的故事、看的动画片或者崇拜的英雄人物是谁？能说说为什么喜欢吗？",
      "现在的生活或工作中，有没有一些让您感觉不太舒服，但又好像总是反复发生的情况？（比如：总是吃力不讨好、总被误解、总是不敢拒绝？）",
      "如果用一个词或一种感觉来形容您的人生到目前为止的主基调，会是什么？（比如：奋斗、幸运、平淡、挣扎？）",
      "不考虑现实限制，您内心深处偷偷希望自己的人生最终会是一个什么样的结局？"
    ],
    "rules_and_logic": {
      "phase_initial_greeting": {
        "condition": "`interaction_phase` == 'initial_greeting'",
        "action": "输出固定问候语，包含问题总数和预计时间，并提出第一个问题（即 `question_list` 中索引为0的问题，因为`current_question_index`从1开始计数）。格式如下：'您好！我是人生脚本探索助手。很高兴能和您聊一聊。接下来我会问您 {{total_questions}} 个问题，大概需要 {{estimated_time}} 左右。我们开始吧？第一个问题是：[这里是 question_list[0] 的文本]'",
        "output_format": "纯文本"
      },
      "phase_conversation_turn": {
        "condition": "`interaction_phase` == 'conversation_turn'",
        "sub_rules": [
          "规则1 (严格按序提问): 根据 `current_question_index` 从 `question_list` 找到当前要问的问题（列表索引为 `current_question_index - 1`）。",
          { "rule_name": "规则2 (判断跑题与柔性拉回)", "logic": "分析 `user_input` 是否回应了上一个问题。如果用户提出无关问题或明显偏离主题： a. 用一两句非常简短、中性的话回应，禁止深入探讨； b. 然后自然过渡回原来的问题； c. 最后清晰重复原始问题。" },
          "规则3 (处理模糊回答): 如果回答过短，仅可追问一次“能稍微具体一点说说吗？”。",
          "规则4 (确认与提问): 用户回答后（且非最后一题），用简短确认语，再提出下一题。",
          "规则5 (不回答无关问题): 若用户问 AI 自身或无关话题，视为跑题，执行规则2。",
          "规则6 (进入报告阶段判断): 最后一题答完后，下一次 `interaction_phase` 变为 `final_report`。"
        ],
        "action": "依据子规则生成回应。",
        "output_format": "纯文本"
      },
      "phase_final_report": {
        "condition": "`interaction_phase` == 'final_report'",
        "sub_rules": [
          "规则1 (基于记录): 报告只能基于 `full_conversation_transcript`。",
          "规则2 (结合理论): 将回答与《人生脚本》概念关联，使用“不确定”用语。",
          "规则3 (结构化报告): 引言、回答摘要、脚本元素分析、结语。",
          "规则4 (中性客观): 不评判。",
          "规则5 (简洁明了): 抓要点。",
          "规则6 (直接输出): 仅输出报告 Markdown，本身不说多余话。"
        ],
        "action": "生成 Markdown 格式初步报告。",
        "output_format": "Markdown文本"
      }
    },
    "task_description": "根据 `interaction_phase` 等变量，严格遵循规则生成输出。"
  }
}
"""

# ---------- 初始化会话状态 ----------
if "history" not in st.session_state:
    st.session_state.history = [
        {"role": "assistant", "content": "你好！我是脚本探索伙伴 ECHO，现在我们开始。"}
    ]
    st.session_state.q_count = 0

# ---------- 显示历史 ----------
for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ---------- 输入框 ----------
user_text = st.chat_input("请输入…", disabled=st.session_state.q_count >= MAX_Q)

# ---------- 处理用户输入 ----------
if user_text:
    st.chat_message("user").markdown(user_text)
    st.session_state.history.append({"role": "user", "content": user_text})

    ctx = [{"role": "system", "content": SYSTEM_PROMPT}]
    ctx += st.session_state.history[-12:]

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
    try:
        next_q = json.loads(raw)["question"].strip()
    except Exception:
        next_q = "__PARSE_ERROR__"

    if next_q and next_q not in ("__END__", "__PARSE_ERROR__"):
        st.session_state.q_count += 1
        st.chat_message("assistant").markdown(next_q)
        st.session_state.history.append({"role": "assistant", "content": next_q})
    else:
        st.session_state.q_count = MAX_Q
        st.session_state.history.append(
            {"role": "assistant", "content": "感谢回答，问题结束！点击下方按钮生成初步报告。"}
        )
    st.rerun()

# ---------- 生成报告 ----------
if st.session_state.q_count >= MAX_Q:
    if st.button("生成报告"):
        answers = [m for m in st.session_state.history if m["role"] == "user"]
        from utils.diagnose import diagnose
        fake_pairs = [{"q": "inj_x", "a": a["content"]} for a in answers]
        data = diagnose(fake_pairs)
        st.markdown(f"""
### 初步脚本报告
**脚本倾向**：{data['summary']}
Injunction 线索：{data['inj_cnt']}
Driver 线索：{data['driver_cnt']}

*(仅供探索，非专业诊断)*
""")
