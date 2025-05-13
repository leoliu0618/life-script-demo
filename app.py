import os
import json
import streamlit as st
from openai import OpenAI, APIConnectionError  # 假设您使用的是OpenAI

# ---------- 代理（本地用；云端留空） ----------
PROXY_URL = os.getenv("PROXY_URL", "")

# ---------- OpenAI 客户端 ----------
# (您的OpenAI客户端初始化代码保持不变)
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

# === 加载您的 JSON Prompt 定义 ===
# 确保SYSTEM_PROMPT是正确的、包含所有阶段定义的JSON字符串
SYSTEM_PROMPT_JSON_STRING = r"""
{
  "prompt_definition": {
    "overall_goal": "作为人生脚本探索AI助手，与用户进行多轮对话，严格按顺序引导用户回答完所有预设问题，并在最后直接生成初步探索报告。整个过程需友好、自然，并确保对话不偏离主题，对话风格应体现对人生脚本探索的专业性和耐心引导。",
    "input_variables": [
      {
        "name": "interaction_phase",
        "description": "当前交互阶段 ('initial_greeting', 'conversation_turn', 'final_report')"
      },
      {
        "name": "current_question_index",
        "description": "当前问题索引 (从1开始计数，仅在 'conversation_turn' 阶段需要)"
      },
      {
        "name": "user_input",
        "description": "用户的上一句输入 (在 'conversation_turn' 和 'final_report' 阶段需要)"
      },
      {
        "name": "conversation_history",
        "description": "最近几轮的对话历史 (用于 'conversation_turn' 判断跑题和连接上下文)"
      },
      {
        "name": "full_conversation_transcript",
        "description": "完整的对话记录 (仅在 'final_report' 阶段需要)"
      },
      {
        "name": "total_questions",
        "description": "预设问题的总数"
      },
      {
        "name": "estimated_time",
        "description": "预计完成对话所需时间 (例如 '5-8分钟')"
      }
    ],
    "question_list": [
      "在您长大的过程中，家里（或者对您影响最大的人）经常强调的‘规矩’或者‘期望’是什么？（比如，要听话、要独立、要让家人骄傲等等）",
      "回想一下您大概上小学的时候，您觉得自己是个什么样的孩子？（可以说说性格、优点或缺点）",
      "那时候，您感觉周围的大人（比如父母、老师）怎么样？他们对您好吗？",
      "有没有什么事情，是您从小就觉得‘千万不能做’或者‘最好不要提’的？（比如：不能犯错、不能软弱、不能比别人差、不能表达需要？）",
      "为了得到表扬或者避免麻烦，您觉得必须做到哪些事情？（比如：必须努力、必须懂事、必须讨人喜欢？）",
      "小时候，您最喜欢听的故事、看的动画片或者崇拜的英雄人物是谁？能说说为什么喜欢吗？（这些往往是我们人生脚本最初的参照和渴望）",
      "现在的生活或工作中，有没有一些让您感觉不太舒服，但又好像总是反复发生的情况？（这些重复的模式可能与我们早年形成的脚本有关）",
      "如果用一个词或一种感觉来形容您的人生到目前为止的主基调，会是什么？（比如：奋斗、幸运、平淡、挣扎？）",
      "不考虑现实限制，您内心深处偷偷希望自己的人生最终会是一个什么样的结局？（这可能反映了您脚本的‘理想结局’）"
    ],
    "rules_and_logic": {
      "phase_initial_greeting": {
        "condition": "`interaction_phase` == 'initial_greeting'",
        "action": "输出固定问候语，包含对人生脚本探索的简要介绍、问题总数和预计时间，并提出第一个问题。格式如下：'您好！我是人生脚本探索AI助手，很高兴能和您一起探索您独特的人生故事。这个探索基于埃里克·伯恩的《人生脚本》理论，帮助我们了解一些早年经历可能如何影响了我们现在的生活模式。接下来我会问您 {{total_questions}} 个问题，大概需要 {{estimated_time}} 左右。我们开始吧？第一个问题是：[这里是 question_list[0] 的文本]'",
        "output_format": "纯文本"
      },
      "phase_conversation_turn": {
        "condition": "`interaction_phase` == 'conversation_turn'",
        "sub_rules": [
          "规则1 (严格按序提问): 根据 `current_question_index` 从 `question_list` 找到当前要问的问题（列表索引为 `current_question_index - 1`）。提问时，可以自然地融入问题列表中括号内的提示语，使其更像对话而非简单念出问题。",
          {
            "rule_name": "规则2 (判断跑题与更自然的柔性拉回)",
            "logic": "分析 `user_input` 是否回应了上一个AI提出的脚本探索问题。如果用户提出无关问题或明显偏离主题（例如开始询问AI自身、谈论完全无关的日常琐事等）：\n  a. **承认并连接**: 用一两句非常简短、中性且带有理解意味的话，承认用户提出的内容，并尝试将其与“探索脚本”这一大方向做微弱连接或解释为何需要先完成当前步骤。示例：\n    * '我注意到您提到了[跑题内容简述]。有时候，我们确实容易从一个点联想到很多其他的事情。为了让我们能一步步梳理您脚本的脉络，我们不妨先回到...' \n    * '您说的[跑题内容简述]，我听到了。这可能也是您生活中的一部分。为了确保我们能系统地完成这次探索，我们先聚焦在...' \n    * '这是一个有趣的问题/想法。在人生脚本的探索中，我们主要关注的是那些可能在您早年形成并持续影响您至今的模式。所以，让我们先看看...' \n  禁止深入探讨跑题内容，核心是表达“我已收到，但我们需要回到轨道”。\n  b. **温和引导**: 使用过渡语，自然地将话题引导回原来的问题。示例：\n    * '...我们不妨先回到刚才关于[上一个问题的核心词]的那个问题上，好吗？'\n    * '...我们先聚焦在对您早年经历的回顾上，这能帮助我们更好地理解脚本。所以，关于[上_一个问题的核心词]：'\n    * '...让我们继续刚才关于[上一个问题的核心词]的讨论吧：'\n  c. **清晰重复**: 清晰地重复一遍用户需要回答的那个原始脚本探索问题 '[重复上一个问题]'。\n  核心目标始终是友好而坚定地回到预设问题流程，不允许在跑题内容上停留超过两三句话。"
          },
          "规则3 (处理模糊回答): 如果 `user_input` 针对脚本问题回答得非常简短或模糊 (如“不知道”、“没什么特别的”)，仅可追问一次，鼓励用户再思考一下。示例：'嗯，这个问题可能需要多一点点回想。您能再试着描述一下当时的感觉或情况吗？哪怕只是一点点印象也好。' 若用户仍表示无法具体说明，则温和确认（如“好的，没关系，我们继续下一个。”）并继续下一步。",
          "规则4 (确认与提问): 如果用户有效回答了脚本问题（或模糊回答追问后确认无需再追问），并且当前问题不是最后一个问题（`current_question_index < total_questions`），则使用略带鼓励和连接性的确认语，然后自然地提出 `current_question_index + 1` 对应的问题。确认语示例：\n  * '明白了，谢谢您的分享，这些信息对我们理解脚本的形成很有帮助。接下来，我们聊聊：[下一个问题]'\n  * '好的，谢谢您告诉我这些。这让我们对您当时的[相关方面，如“自我感觉”或“家庭氛围”]有了一些了解。那么，[下一个问题]'\n  * '嗯，我理解了。您提到的这一点可能和[脚本某个概念的微弱联系，如“早年决定”或“父母期望”]有关。我们继续深入看看：[下一个问题]'\n  确认语应简短自然，不进行深入分析，但可以略微点出与脚本探索的关联。",
          "规则5 (不回答无关问题): 若用户反复询问AI自身信息、寻求具体生活建议、或进行与脚本探索完全无关的长时间闲聊，应更坚定地执行规则2的柔性拉回，强调本次对话的核心目标。",
          "规则6 (进入报告阶段判断): 在用户回答完最后一个问题（`current_question_index == total_questions`）后，下一次交互的`interaction_phase`应变为`final_report`。AI在收到最后一个问题的回答后，可以说一句简短的总结性确认，例如：'好的，这是我们最后一个问题了，非常感谢您耐心的分享！'"
        ],
        "action": "依据子规则生成回应。",
        "output_format": "纯文本"
      },
      "phase_final_report": {
        "condition": "`interaction_phase` == 'final_report'",
        "sub_rules": [
          "规则1 (基于记录): 报告只能基于 `full_conversation_transcript`。",
          "规则2 (结合理论): 将回答与《人生脚本》概念（如书中的脚本装置、心理地位、游戏、脚本结局等）联系起来，使用“可能”、“似乎”、“或许反映了”等探索性词语。",
          "规则3 (结构化报告): 引言（感谢参与、强调初步探索、非诊断）、关键回答回顾与串联、初步识别的脚本元素（如可能的禁止信息、驱动力、重复模式等）、对脚本可能结局的初步推测、结语（鼓励自我探索、肯定用户努力、专业帮助建议、AI局限性）。",
          "规则4 (中性与赋能): 保持中性、客观，同时措辞应带有赋能感，让用户感到被理解和有改变的可能。",
          "规则5 (简洁易懂): 使用清晰、易懂的语言，避免过多专业术语堆砌。抓住核心线索。",
          "规则6 (直接输出): 直接生成报告内容（Markdown格式），报告本身应作为一次完整的输出，前面不加“报告如下：”之类的引导。"
        ],
        "action": "生成 Markdown 格式初步报告。",
        "output_format": "Markdown文本"
      }
    },
    "task_description": "根据当前的 `interaction_phase` 和其他输入变量，严格遵循 `rules_and_logic` 中对应阶段的规则，生成所需的输出，并尽可能使对话自然、流畅，体现人生脚本探索的专业氛围。"
  }
}
"""
PROMPT_CONFIG = json.loads(SYSTEM_PROMPT_JSON_STRING)["prompt_definition"]
QUESTION_LIST = PROMPT_CONFIG["question_list"]
TOTAL_QUESTIONS = len(QUESTION_LIST)
ESTIMATED_TIME = "5-8分钟"  # 您可以根据问题数量调整

# ---------- 初始化会话状态 ----------
if "history" not in st.session_state:
    st.session_state.history = []  # 存放 {"role": "user/assistant", "content": "..."}
if "interaction_phase" not in st.session_state:
    st.session_state.interaction_phase = "initial_greeting"
if "current_question_index" not in st.session_state:
    # current_question_index: 1 表示第一个问题, ..., TOTAL_QUESTIONS 表示最后一个问题
    st.session_state.current_question_index = 1
if "report_generated" not in st.session_state:
    st.session_state.report_generated = False


# ---------- 核心函数：调用LLM并处理回复 ----------
def get_ai_response(phase, user_input_text=None, history_for_prompt=None, full_transcript=None):
    """根据当前阶段和输入，构建发送给LLM的prompt并获取回复"""

    # 1. 构建传递给LLM的 messages (包含 system prompt 和对话历史/上下文)
    #    这里的关键是，system_prompt不再是整个大JSON，而是根据当前阶段动态生成的、更直接的指令。
    #    我们将JSON中的规则“翻译”成给LLM的直接指令。

    active_system_prompt = f"你是一个人生脚本探索AI助手。{PROMPT_CONFIG['overall_goal']}\n"
    active_system_prompt += f"当前交互阶段是: {phase}\n"
    active_system_prompt += f"共有 {TOTAL_QUESTIONS} 个预设问题。\n"

    if phase == "initial_greeting":
        active_system_prompt += PROMPT_CONFIG["rules_and_logic"]["phase_initial_greeting"]["action"].replace(
            "{{total_questions}}", str(TOTAL_QUESTIONS)
        ).replace(
            "{{estimated_time}}", ESTIMATED_TIME
        ).replace(
            "[这里是 question_list[0] 的文本]", QUESTION_LIST[0]  # 假设第一个问题索引是0
        )
        active_system_prompt += "\n请严格按照上述格式输出问候和第一个问题。"

    elif phase == "conversation_turn":
        current_q_text = QUESTION_LIST[st.session_state.current_question_index - 1]
        active_system_prompt += f"当前正在处理第 {st.session_state.current_question_index} 个问题。\n"
        active_system_prompt += f"当前问题是：“{current_q_text}”\n"
        if user_input_text:
            active_system_prompt += f"用户的最新回答是：“{user_input_text}”\n"
        if history_for_prompt:
            active_system_prompt += f"最近的对话历史如下：\n{history_for_prompt}\n"

        active_system_prompt += "请严格遵循以下对话规则：\n"
        active_system_prompt += "- " + PROMPT_CONFIG["rules_and_logic"]["phase_conversation_turn"]["sub_rules"][
            0] + "\n"  # 规则1
        # 规则2 (柔性拉回) - 需要更详细的指令给LLM
        pull_back_logic = PROMPT_CONFIG["rules_and_logic"]["phase_conversation_turn"]["sub_rules"][1]["logic"]
        active_system_prompt += f"- {pull_back_logic.replace('[重复上一个问题]', f'“{current_q_text}”')}\n"
        active_system_prompt += "- " + PROMPT_CONFIG["rules_and_logic"]["phase_conversation_turn"]["sub_rules"][
            2] + "\n"  # 规则3
        active_system_prompt += "- " + PROMPT_CONFIG["rules_and_logic"]["phase_conversation_turn"]["sub_rules"][
            3] + "\n"  # 规则4
        if st.session_state.current_question_index < TOTAL_QUESTIONS:
            next_q_text = QUESTION_LIST[st.session_state.current_question_index]  # 下一个问题的文本
            active_system_prompt = active_system_prompt.replace("[下一个问题]", f"“{next_q_text}”")

        active_system_prompt += "\n你的任务是根据用户的回答和上述规则，生成你的下一句回应。"
        active_system_prompt += "如果用户跑题，按规则2柔性拉回；如果用户回答模糊，按规则3追问；如果用户正常回答，按规则4确认并提出下一个问题。"
        active_system_prompt += "如果当前是最后一个问题，并且用户回答了，请只做简短确认，不要提“下一个问题”。"


    elif phase == "final_report":
        active_system_prompt += "现在所有问题已回答完毕。用户的完整对话记录如下：\n"
        active_system_prompt += f"{full_transcript}\n"
        active_system_prompt += "请严格遵循以下报告生成规则：\n"
        for rule in PROMPT_CONFIG["rules_and_logic"]["phase_final_report"]["sub_rules"]:
            active_system_prompt += f"- {rule}\n"
        active_system_prompt += "\n你的任务是直接生成Markdown格式的初步人生脚本探索报告，不要添加任何其他对话性文字。"

    # 2. 构建messages列表
    messages_for_llm = [{"role": "system", "content": active_system_prompt}]
    # 在conversation_turn阶段，可以考虑加入最近几轮的user/assistant历史，但不加入system prompt中的历史
    if phase == "conversation_turn" and st.session_state.history:
        # 只添加最近的几轮对话历史作为上下文，避免过长
        for msg in st.session_state.history[-4:]:  # 例如最近4条
            if msg["role"] != "system":  # 避免重复添加system
                messages_for_llm.append(msg)
        if user_input_text:  # 确保当前用户输入也包含在内（如果适用）
            # 如果history已经包含了当前user_input，则不需要重复添加
            if not (messages_for_llm and messages_for_llm[-1]["role"] == "user" and messages_for_llm[-1][
                "content"] == user_input_text):
                messages_for_llm.append({"role": "user", "content": user_input_text})

    # st.write("DEBUG: Prompt to LLM:") # 调试时可以取消注释
    # st.text(messages_for_llm)

    try:
        resp = client.chat.completions.create(
            model="gpt-4.1-2025-04-14",  # 或者您选择的模型
            messages=messages_for_llm,
            temperature=0.5,  # 降低一点随机性，使其更遵循指令
        )
        ai_content = resp.choices[0].message.content.strip()
        return ai_content
    except APIConnectionError as e:
        st.error("🚧 无法连接 OpenAI，检查网络/代理后重试。\n\n" + str(e))
        return None
    except Exception as e:
        st.error(f"调用LLM时发生未知错误: {e}")
        return None


# ---------- 主流程控制 ----------

# 1. 处理初始问候 (如果还没有历史记录，或者明确是initial_greeting阶段)
if not st.session_state.history and st.session_state.interaction_phase == "initial_greeting":
    with st.spinner("AI正在准备开场白..."):
        initial_greeting_text = get_ai_response(phase="initial_greeting")
    if initial_greeting_text:
        st.session_state.history.append({"role": "assistant", "content": initial_greeting_text})
        st.session_state.interaction_phase = "conversation_turn"  # 进入对话阶段
        st.session_state.current_question_index = 1  # AI问了第一个问题
        st.rerun()

# 2. 显示聊天历史
for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 3. 获取用户输入
if not st.session_state.report_generated and st.session_state.interaction_phase == "conversation_turn":
    user_text = st.chat_input("请输入您的回答…")

    if user_text:
        # 将用户回复添加到历史
        st.session_state.history.append({"role": "user", "content": user_text})
        # 立刻显示用户回复
        with st.chat_message("user"):
            st.markdown(user_text)

        # 准备调用LLM的上下文
        # history_for_prompt = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.history[-5:]]) # 最近5条

        # 调用LLM获取AI的下一句话
        with st.spinner("AI正在思考..."):
            # 注意：这里传递给 get_ai_response 的 history_for_prompt 是为了让 system prompt 知道最近的对话
            # 而 messages_for_llm 列表中的历史是更直接的上下文
            ai_response_text = get_ai_response(
                phase="conversation_turn",
                user_input_text=user_text
                # history_for_prompt=history_for_prompt # 如果system prompt需要格式化的历史
            )

        if ai_response_text:
            st.session_state.history.append({"role": "assistant", "content": ai_response_text})

            # 判断是否所有问题都已问完（基于AI的回复或计数）
            # 这里的逻辑需要非常小心，AI的回复可能不是直接的下一个问题文本
            # 一个更稳妥的方式是严格管理 current_question_index
            # 假设AI的回复如果是提问，会包含问题内容；如果是确认，则我们推进index

            # 简化的推进逻辑：只要AI回复了，我们就认为当前问题结束，准备下一个
            # 除非AI明确在拉回或者追问 (这部分逻辑需要在get_ai_response的prompt中由AI自己判断并输出)
            if st.session_state.current_question_index < TOTAL_QUESTIONS:
                # 只有当AI的回复不是明显的拉回或追问时，才增加索引
                # 这个判断比较复杂，暂时假设AI会正确提出下一个问题或明确指示结束
                # 我们需要在 system prompt 中让AI明确它是否提出了下一个问题
                # 或者，AI的回复中包含特定标记来指示是否进入下一个问题

                # **更简单的做法**: 相信AI会遵循指令，如果它没拉回，那么它就是要提下一个问题或结束了
                # 我们主要靠 `current_question_index` 来控制
                st.session_state.current_question_index += 1

            if st.session_state.current_question_index > TOTAL_QUESTIONS:
                st.session_state.interaction_phase = "final_report"
        else:
            # 如果AI没有回复（比如API错误），也显示一条消息
            st.session_state.history.append({"role": "assistant", "content": "抱歉，我暂时无法回应，请稍后再试。"})

        st.rerun()

# 4. 生成并显示报告
if st.session_state.interaction_phase == "final_report" and not st.session_state.report_generated:
    st.info("所有问题已回答完毕，正在为您生成初步报告...")

    full_transcript_for_report = "\n".join(
        [f"{('用户' if m['role'] == 'user' else 'AI')}: {m['content']}" for m in st.session_state.history])

    with st.spinner("报告生成中..."):
        report_content = get_ai_response(
            phase="final_report",
            full_transcript=full_transcript_for_report
        )

    if report_content:
        st.session_state.report_generated = True
        # 直接显示报告，因为AI被指示直接输出Markdown
        st.markdown("---")
        st.subheader("初步人生脚本探索报告")
        st.markdown(report_content)
        st.success("报告生成完毕！请注意，这仅为初步探索，非专业诊断。")
        # st.session_state.history.append({"role": "assistant", "content": report_content}) # 看是否需要把报告也加入历史
    else:
        st.error("抱歉，生成报告时遇到问题。")

    # 添加重新开始按钮
    if st.button("重新开始探索"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()