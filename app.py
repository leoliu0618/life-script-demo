import streamlit as st
import json
import os
from openai import OpenAI, APIConnectionError

# ---------- 配置区域 ----------
# 建议将此JSON Prompt内容保存为一个单独的 .json 文件读取，
# 但为了“复制粘贴即可运行”，这里直接作为多行字符串嵌入。
SYSTEM_PROMPT_JSON_STRING = r"""
{
  "prompt_definition": {
    "overall_goal": "作为人生脚本探索AI助手，与用户进行多轮对话，严格按顺序引导用户回答完所有预设问题，并在最后直接生成初步探索报告。整个过程需友好、自然，并确保对话不偏离主题，对话风格应体现对人生脚本探索的专业性和耐心引导。在提问时，应避免使用‘第几题’、‘下一个问题是’等明确编号的说法，而是自然过渡到新问题。",
    "input_variables": [
      {
        "name": "interaction_phase",
        "description": "当前交互阶段 ('initial_greeting', 'conversation_turn', 'final_report')"
      },
      {
        "name": "current_question_index",
        "description": "当前问题索引 (从1开始计数，仅在 'conversation_turn' 阶段需要，用于内部逻辑确定问题内容)"
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
        "action": "输出固定问候语，包含对人生脚本探索的简要介绍、问题总数和预计时间，并自然地提出第一个问题。格式如下：'您好！我是人生脚本探索AI助手，很高兴能和您一起探索您独特的人生故事。这个探索能帮助我们了解一些早年经历可能如何影响了我们现在的生活模式。接下来我会和您聊聊大概 {{total_questions}} 个方面的内容，预计需要 {{estimated_time}} 左右。我们开始吧？[这里是 question_list[0] 的文本]'",
        "output_format": "纯文本"
      },
      "phase_conversation_turn": {
        "condition": "`interaction_phase` == 'conversation_turn'",
        "sub_rules": [
          "规则1 (内部逻辑按序提问): AI内部根据 `current_question_index` 从 `question_list` 找到当前要问的问题（列表索引为 `current_question_index - 1`）。向用户提问时，融入问题列表中括号内的提示语，使其更像对话。",
          {
            "rule_name": "规则2 (判断跑题与更自然的柔性拉回)",
            "logic": "分析 `user_input` 是否回应了上一个AI提出的脚本探索问题。如果用户提出无关问题或明显偏离主题（例如开始询问AI自身、谈论完全无关的日常琐事等）：\n  a. **承认并连接**: 用一两句非常简短、中性且带有理解意味的话，承认用户提出的内容，并尝试将其与“探索脚本”这一大方向做微弱连接或解释为何需要先完成当前步骤。示例：\n    * '我注意到您提到了[跑题内容简述]。有时候，我们确实容易从一个点联想到很多其他的事情。为了让我们能一步步梳理您脚本的脉络，我们不妨先回到...' \n    * '您说的[跑题内容简述]，我听到了。这可能也是您生活中的一部分。为了确保我们能系统地完成这次探索，我们先聚焦在...' \n    * '这是一个有趣的问题/想法。在人生脚本的探索中，我们主要关注的是那些可能在您早年形成并持续影响您至今的模式。所以，让我们先看看...' \n  禁止深入探讨跑题内容，核心是表达“我已收到，但我们需要回到轨道”。\n  b. **温和引导**: 使用过渡语，自然地将话题引导回原来的问题。示例：\n    * '...我们不妨先回到刚才关于[上一个问题的核心词]的那个问题上，好吗？'\n    * '...我们先聚焦在对您早年经历的回顾上，这能帮助我们更好地理解脚本。所以，关于[上一个问题的核心词]：'\n    * '...让我们继续刚才关于[上一个问题的核心词]的讨论吧：'\n  c. **清晰重复**: 清晰地重复一遍用户需要回答的那个原始脚本探索问题 '[重复上一个问题]'。\n  核心目标始终是友好而坚定地回到预设问题流程，不允许在跑题内容上停留超过两三句话。"
          },
          "规则3 (处理模糊回答): 如果 `user_input` 针对脚本问题回答得非常简短或模糊 (如“不知道”、“没什么特别的”)，仅可追问一次，鼓励用户再思考一下。示例：'嗯，这个问题可能需要多一点点回想。您能再试着描述一下当时的感觉或情况吗？哪怕只是一点点印象也好。' 若用户仍表示无法具体说明，则温和确认（如“好的，没关系，我们继续聊其他的。”）并继续下一步。",
          {
            "rule_name": "规则4 (确认与自然提问)",
            "logic": "如果用户有效回答了脚本问题（或模糊回答追问后确认无需再追问），并且当前问题不是最后一个问题（`current_question_index < total_questions`），则使用略带鼓励和连接性的确认语，然后自然地提出由`current_question_index + 1`（内部逻辑）确定的下一个问题（即 `question_list` 中索引为 `current_question_index` 的问题）。确认语及提问示例：\n  * '明白了，谢谢您的分享，这些信息对我们理解脚本的形成很有帮助。我们接着聊聊：[下一个问题文本]'\n  * '好的，谢谢您告诉我这些。这让我们对您当时的[相关方面]有了一些了解。那么，关于[下一个问题核心词]，您是怎么看的呢？[下一个问题文本]'\n  * '嗯，我理解了。您提到的这一点可能和[脚本某个概念的微弱联系]有关。我们继续深入看看：[下一个问题文本]'\n  确认语应简短自然，不进行深入分析，但可以略微点出与脚本探索的关联。提问时直接陈述问题，不加编号。"
          },
          "规则5 (不回答无关问题): 若用户反复询问AI自身信息、寻求具体生活建议、或进行与脚本探索完全无关的长时间闲聊，应更坚定地执行规则2的柔性拉回，强调本次对话的核心目标。",
          "规则6 (对话结束的确认): 在用户回答完最后一个问题（`current_question_index == total_questions`）后，AI在收到最后一个问题的回答后，可以说一句简短的总结性确认，例如：'好的，我们探讨的方面就到这里了，非常感谢您耐心的分享！' 之后，外部逻辑应将`interaction_phase`应变为`final_report`。"
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
    "task_description": "根据当前的 `interaction_phase` 和其他输入变量，严格遵循 `rules_and_logic` 中对应阶段的规则，生成所需的输出，并尽可能使对话自然、流畅，避免使用问题编号，体现人生脚本探索的专业氛围。"
  }
}
"""
PROMPT_CONFIG = json.loads(SYSTEM_PROMPT_JSON_STRING)["prompt_definition"]
QUESTION_LIST = PROMPT_CONFIG["question_list"]
TOTAL_QUESTIONS = len(QUESTION_LIST)
ESTIMATED_TIME = "5-8分钟"
OPENAI_MODEL_NAME = "gpt-4.1-2025-04-14"  # 您想使用的模型，例如 "gpt-4o", "gpt-4-turbo"

# ---------- OpenAI 客户端 ----------
try:
    client = OpenAI(
        api_key=st.secrets.get("OPENAI_API_KEY"),  # 从 Streamlit secrets 获取
        timeout=60,
        max_retries=3,
    )
except Exception as e:
    st.error(f"OpenAI API Key 配置错误或缺失。请在 Streamlit Cloud 的 Secrets 中设置 OPENAI_API_KEY。错误: {e}")
    st.stop()

# ---------- Streamlit 页面 ----------
st.set_page_config(page_title="人生脚本探索", layout="centered")
st.title("人生脚本探索 Demo 🌀")
st.caption("与AI一起，初步探索您的人生脚本")

# ---------- 初始化会话状态 ----------
if "history" not in st.session_state:
    st.session_state.history = []
if "interaction_phase" not in st.session_state:
    st.session_state.interaction_phase = "initial_greeting"
if "current_question_index" not in st.session_state:
    st.session_state.current_question_index = 1  # 追踪当前是第几个 *脚本问题*
if "ai_is_waiting_for_user_start_confirmation" not in st.session_state:
    st.session_state.ai_is_waiting_for_user_start_confirmation = True  # 初始状态，等待用户确认开始
if "report_generated" not in st.session_state:
    st.session_state.report_generated = False


# ---------- 核心函数：调用LLM并处理回复 ----------
def get_ai_response(phase, user_input_text=None, current_q_idx=None, conv_history_for_prompt=None,
                    full_conv_transcript=None):
    active_system_prompt = f"{PROMPT_CONFIG['overall_goal']}\n"
    active_system_prompt += f"当前交互阶段是: {phase}\n"

    if phase == "initial_greeting":
        # 直接使用PROMPT_CONFIG中为initial_greeting定义的action文本，并替换占位符
        action_text = PROMPT_CONFIG["rules_and_logic"]["phase_initial_greeting"]["action"]
        action_text = action_text.replace("{{total_questions}}", str(TOTAL_QUESTIONS))
        action_text = action_text.replace("{{estimated_time}}", ESTIMATED_TIME)
        # 移除"[这里是 question_list[0] 的文本]"，因为我们希望AI只说问候语并询问是否开始
        action_text = action_text.split("我们开始吧？")[0] + "我们开始吧？您准备好了吗？"  # 修改问候语，不直接提问
        active_system_prompt += f"你的任务是只说以下这句话，不要添加任何其他内容：'{action_text}'"
        messages_for_llm = [{"role": "system", "content": active_system_prompt}]

    elif phase == "ask_first_question":  # 新增一个阶段或状态来处理提第一个问题
        first_q_text = QUESTION_LIST[0]
        active_system_prompt += f"用户已确认开始。你的任务是自然地提出以下第一个问题，不要加编号：'{first_q_text}'"
        messages_for_llm = [{"role": "system", "content": active_system_prompt}]

    elif phase == "conversation_turn":
        if current_q_idx is None or current_q_idx < 1 or current_q_idx > TOTAL_QUESTIONS:
            return "内部错误：问题索引无效。"

        current_q_text_for_llm = QUESTION_LIST[current_q_idx - 1]  # 当前AI要问或刚问过的问题

        active_system_prompt += f"当前正在处理脚本问题的第 {current_q_idx} 个方面。\n"
        active_system_prompt += f"AI上一个提出的问题是：“{current_q_text_for_llm}”\n"  # 指示AI它“刚刚”问了什么
        if user_input_text:
            active_system_prompt += f"用户的最新回答是：“{user_input_text}”\n"
        if conv_history_for_prompt:  # 最近几轮对话
            active_system_prompt += f"最近的对话历史如下（仅供参考上下文）：\n{conv_history_for_prompt}\n"

        active_system_prompt += "请严格遵循以下对话规则来生成你的回应：\n"
        active_system_prompt += f"- {PROMPT_CONFIG['rules_and_logic']['phase_conversation_turn']['sub_rules'][0]}\n"  # 规则1

        pull_back_logic = PROMPT_CONFIG['rules_and_logic']['phase_conversation_turn']['sub_rules'][1]['logic']
        active_system_prompt += f"- 规则2 (判断跑题与更自然的柔性拉回): {pull_back_logic.replace('[重复上一个问题]', f'“{current_q_text_for_llm}”')}\n"

        active_system_prompt += f"- {PROMPT_CONFIG['rules_and_logic']['phase_conversation_turn']['sub_rules'][2]}\n"  # 规则3

        natural_query_logic = PROMPT_CONFIG['rules_and_logic']['phase_conversation_turn']['sub_rules'][3]['logic']
        if current_q_idx < TOTAL_QUESTIONS:
            next_q_text_for_llm = QUESTION_LIST[current_q_idx]  # 下一个实际要问的问题
            natural_query_logic = natural_query_logic.replace("[下一个问题文本]", f"“{next_q_text_for_llm}”")
            natural_query_logic = natural_query_logic.replace("[下一个问题核心词]", "下一个方面")  # 替换占位符
        else:  # 如果是最后一个问题，规则4逻辑应变为只做确认
            natural_query_logic = PROMPT_CONFIG['rules_and_logic']['phase_conversation_turn']['sub_rules'][
                5]  # 指向规则6的结束确认
            natural_query_logic = natural_query_logic.split("之后，外部逻辑应将")[0]  # 取前半句作为AI的输出指令

        active_system_prompt += f"- 规则4 (确认与自然提问) 或结束确认: {natural_query_logic}\n"
        active_system_prompt += "\n你的任务是根据用户的回答和上述规则，生成你的下一句纯文本回应。"
        active_system_prompt += "如果用户跑题，按规则2柔性拉回；如果用户回答模糊，按规则3追问；如果用户正常回答且不是最后一题，按规则4确认并自然地问出下一个问题；如果用户回答了最后一题，则按规则6的描述进行总结性确认。"

        messages_for_llm = [{"role": "system", "content": active_system_prompt}]
        # 仅当有实际对话历史时才添加最近的user/assistant消息
        if st.session_state.history:
            for msg in st.session_state.history[-4:]:  # 最近4条作为聊天上下文
                messages_for_llm.append(msg)
        # 确保当前用户输入在messages_for_llm的最后（如果存在）
        if user_input_text and (not messages_for_llm or messages_for_llm[-1]["content"] != user_input_text):
            messages_for_llm.append({"role": "user", "content": user_input_text})


    elif phase == "final_report":
        active_system_prompt += "现在所有问题已回答完毕。用户的完整对话记录如下：\n"
        active_system_prompt += f"{full_conv_transcript}\n"
        active_system_prompt += "请严格遵循以下报告生成规则直接输出Markdown报告：\n"
        for rule_detail in PROMPT_CONFIG["rules_and_logic"]["phase_final_report"]["sub_rules"]:
            active_system_prompt += f"- {rule_detail}\n"  # 如果rule_detail是字符串
        messages_for_llm = [{"role": "system", "content": active_system_prompt}]
    else:
        return "内部错误：未知的交互阶段。"

    # st.write("DEBUG: System Prompt to LLM:") # 调试时可以取消注释
    # st.text(active_system_prompt)
    # st.write("DEBUG: Messages to LLM:")
    # st.json(messages_for_llm)

    try:
        resp = client.chat.completions.create(
            model=OPENAI_MODEL_NAME,
            messages=messages_for_llm,
            temperature=0.6,  # 略微降低，使其更稳定
        )
        ai_content = resp.choices[0].message.content.strip()
        return ai_content
    except APIConnectionError as e:
        st.error(f"🚧 无法连接 OpenAI，检查网络/代理后重试。\n\n{e}")
        return None
    except Exception as e:
        st.error(f"调用LLM时发生未知错误: {e}")
        return None


# ---------- 主流程控制 ----------

# 1. AI主动发出问候 (仅在首次加载时)
if not st.session_state.history and st.session_state.interaction_phase == "initial_greeting":
    with st.spinner("AI助手正在准备开场白..."):
        initial_greeting_text = get_ai_response(phase="initial_greeting")
    if initial_greeting_text:
        st.session_state.history.append({"role": "assistant", "content": initial_greeting_text})
        st.session_state.ai_is_waiting_for_user_start_confirmation = True
        st.rerun()

# 2. 显示聊天历史
for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 3. 获取用户输入和处理对话
if not st.session_state.report_generated:
    user_text = st.chat_input("请输入您的回答…", key="user_input")

    if user_text:
        st.session_state.history.append({"role": "user", "content": user_text})
        with st.chat_message("user"):  # 显示用户本轮输入
            st.markdown(user_text)

        ai_response_text = None
        next_phase = st.session_state.interaction_phase
        next_q_idx = st.session_state.current_question_index

        if st.session_state.ai_is_waiting_for_user_start_confirmation:
            if any(affirmative_word in user_text.lower() for affirmative_word in
                   ["好", "准备", "可以", "开始", "是的", "嗯", "ok", "yes"]):
                with st.spinner("好的，我们开始第一个问题..."):
                    ai_response_text = get_ai_response(phase="ask_first_question")
                next_phase = "conversation_turn"
                next_q_idx = 1  # 第一个脚本问题
                st.session_state.ai_is_waiting_for_user_start_confirmation = False
            else:
                ai_response_text = "嗯，如果您准备好了，请告诉我一声，我们就可以开始探索了。"
                # 保持在等待确认阶段

        elif st.session_state.interaction_phase == "conversation_turn":
            with st.spinner("AI正在思考..."):
                # 构建最近对话历史给Prompt参考
                recent_history_str = "\n".join(
                    [f"{m['role']}: {m['content']}" for m in st.session_state.history[-5:-1]])  # 不包括当前用户输入

                ai_response_text = get_ai_response(
                    phase="conversation_turn",
                    user_input_text=user_text,
                    current_q_idx=st.session_state.current_question_index,
                    conv_history_for_prompt=recent_history_str
                )

            # 推进问题索引的逻辑：基于AI的回复内容判断是否成功提问
            # 这是一个难点，最稳妥是AI回复包含特殊标记或LLM能稳定按指令行动
            # 简化：我们假设AI会正确遵循Prompt中的规则4或规则6的逻辑
            # 如果AI的回复不是明显的拉回或追问，并且当前不是最后一个问题，则我们准备问下一个问题
            # 如果是最后一个问题，AI会给出结束确认语，然后我们将进入报告阶段

            is_last_Youtubeed = (st.session_state.current_question_index == TOTAL_QUESTIONS)

            if ai_response_text:
                # 检查AI的回复是否是“结束确认语”，表明最后一个问题已回答
                # (需要Prompt中规则6的AI输出与这里的判断匹配)
                if is_last_Youtubeed and "我们探讨的方面就到这里了" in ai_response_text:  # 根据Prompt中规则6的AI输出调整
                    next_phase = "final_report"
                elif not is_last_Youtubeed:
                    # 假设如果AI没有明确拉回或追问（这部分由LLM内部按prompt处理），
                    # 并且输出了包含下一个问题的内容（或至少是确认了当前回答），
                    # 那么我们就可以安全地增加索引。
                    # 这是一个需要通过测试来验证和调整的薄弱环节。
                    # 最好的方式是让LLM的输出包含一个是否前进的明确信号。
                    # 为了简单，我们先假设LLM会正确处理并提出下一个问题（如果适用）。
                    next_q_idx = st.session_state.current_question_index + 1
                # else: 保持当前问题索引（比如AI在拉回或追问） - 这部分由LLM在Prompt指导下自行决定其输出

        if ai_response_text:
            st.session_state.history.append({"role": "assistant", "content": ai_response_text})
            st.session_state.interaction_phase = next_phase
            st.session_state.current_question_index = next_q_idx
        else:  # API 调用失败或无返回
            if st.session_state.interaction_phase != "initial_greeting":  # 避免初始加载失败时重复添加
                st.session_state.history.append(
                    {"role": "assistant", "content": "抱歉，我暂时无法回应，请检查网络或稍后再试。"})

        st.rerun()

# 4. 生成并显示报告
if st.session_state.interaction_phase == "final_report" and not st.session_state.report_generated:
    st.info("所有问题已回答完毕，正在为您生成初步报告...")

    full_transcript_for_report = "\n\n".join(
        [f"{('用户' if m['role'] == 'user' else 'AI助手')}:\n{m['content']}" for m in st.session_state.history])

    with st.spinner("报告生成中，这可能需要一点时间..."):
        report_content = get_ai_response(
            phase="final_report",
            full_conv_transcript=full_transcript_for_report
        )

    if report_content:
        st.session_state.report_generated = True
        st.markdown("---")
        st.subheader("初步人生脚本探索报告")
        st.markdown(report_content)  # AI被指示直接输出Markdown
        st.success("报告生成完毕！请注意，这仅为初步探索，非专业诊断。")
    else:
        st.error("抱歉，生成报告时遇到问题。")

    if st.button("重新开始探索", key="restart_button"):
        # 清理 session_state 以重新开始
        keys_to_delete = ["history", "interaction_phase", "current_question_index",
                          "ai_is_waiting_for_user_start_confirmation", "report_generated"]
        for key in keys_to_delete:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()