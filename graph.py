report_structure = """使用此结构来创建关于用户提供主题的报告：

1. 引言（无需研究）
   - 主题领域的简要概述

2. 正文部分：
   - 每个部分应关注用户提供主题的一个子主题
   
3. 结论
   - 力求使用1个结构元素（列表或表格）来提炼正文部分的内容
   - 提供报告的简明总结"""


report_planner_query_writer_instructions="""您正在为一份报告进行研究。

<报告主题>
{topic}
</报告主题>

<报告组织结构>
{report_organization}
</报告组织结构>

<任务>
您的目标是生成{number_of_queries}个网络搜索查询，以帮助收集规划报告章节所需的信息。

这些查询应当：

1. 与报告主题相关
2. 有助于满足报告组织结构中指定的要求

请确保查询足够具体，以找到高质量、相关的资源，同时涵盖报告结构所需的广度。
</任务>
"""

report_planner_instructions="""我需要一个简洁且重点突出的报告计划。

<报告主题>
报告的主题是：
{topic}
</报告主题>

<报告组织结构>
报告应遵循以下组织结构：
{report_organization}
</报告组织结构>

<背景信息>
以下是用于规划报告章节的背景信息：
{context}
</背景信息>

<任务>
为报告生成章节列表。您的计划应当紧凑且重点突出，避免章节重叠或不必要的填充内容。

例如，一个良好的报告结构可能如下所示：
1/ 引言
2/ 主题A概述
3/ 主题B概述
4/ A与B的比较
5/ 结论

每个章节应包含以下字段：

- name（名称）- 报告此章节的名称。
- description（描述）- 本章节涵盖的主要主题的简要概述。
- research（研究）- 是否需要为报告的这个章节进行网络研究。
- content（内容）- 章节的内容，现在暂时留空。

整合指南：
- 在主题章节内包含示例和实施细节，而非作为单独章节
- 确保每个章节有明确的目的，内容不重叠
- 合并相关概念而非分开处理

提交前，请检查您的结构，确保没有冗余章节并遵循逻辑流程。
</任务>

<反馈>
以下是审核对报告结构的反馈（如有）：
{feedback}
</反馈>
"""


final_section_writer_instructions="""您是一位专业技术作家，正在撰写一个综合报告其他部分信息的章节。

<报告主题>
{topic}
</报告主题>

<章节名称>
{section_name}
</章节名称>

<章节主题> 
{section_topic}
</章节主题>

<可用报告内容>
{context}
</可用报告内容>

<任务>
1. 章节特定方法：

对于引言：
- 使用#作为报告标题（Markdown格式）
- 限制在50-100字
- 使用简单清晰的语言
- 在1-2段中专注于报告的核心动机
- 使用清晰的叙述结构来介绍报告
- 不包含结构元素（无列表或表格）
- 不需要来源部分

对于结论/总结：
- 使用##作为章节标题（Markdown格式）
- 限制在100-150字
- 对于比较性报告：
    * 必须包含使用Markdown表格语法的重点比较表
    * 表格应提炼报告中的见解
    * 保持表格条目清晰简洁
- 对于非比较性报告： 
    * 仅在有助于提炼报告中的要点时使用一种结构元素：
    * 要么是使用Markdown表格语法比较报告中出现的项目的重点表格
    * 要么是使用正确Markdown列表语法的简短列表：
      - 使用`*`或`-`表示无序列表
      - 使用`1.`表示有序列表
      - 确保正确的缩进和间距
- 以具体的后续步骤或影响结尾
- 不需要来源部分

3. 写作方法：
- 使用具体细节而非一般性陈述
- 让每个词都有价值
- 专注于您最重要的一点
</任务>

<质量检查>
- 对于引言：50-100字限制，#用于报告标题，无结构元素，无来源部分
- 对于结论：100-150字限制，##用于章节标题，最多只有一个结构元素，无来源部分
- Markdown格式
- 不要在回复中包含字数统计或任何前言
</质量检查>"""

section_writer_instructions = """您是一位专业技术作家，正在撰写技术报告的一个章节。

<报告主题>
{topic}
</报告主题>

<章节名称>
{section_name}
</章节名称>

<章节主题>
{section_topic}
</章节主题>

<现有章节内容（如已填写）>
{section_content}
</现有章节内容>

<源材料>
{context}
</源材料>

<写作指南>
1. 如果现有章节内容未填写，请从头开始撰写新章节。
2. 如果现有章节内容已填写，请撰写一个新章节，将现有章节内容与源材料综合起来。
</写作指南>

<长度和风格>
- 严格限制在150-200字
- 不使用营销语言
- 技术重点
- 使用简单、清晰的语言
- 以**粗体**标注您最重要的见解开始
- 使用简短段落（最多2-3句话）
- 使用##作为章节标题（Markdown格式）
- 仅在有助于阐明观点时使用一种结构元素：
  * 要么是比较2-3个关键项目的简明表格（使用Markdown表格语法）
  * 要么是使用正确Markdown列表语法的简短列表（3-5项）：
    - 使用`*`或`-`表示无序列表
    - 使用`1.`表示有序列表
    - 确保正确的缩进和间距
- 以### 参考资料结尾，引用以下源材料，格式为：
  * 列出每个来源的标题、日期和URL
  * 格式：`- 标题：URL`
</长度和风格>

<质量检查>
- 恰好150-200字（不包括标题和来源）
- 谨慎使用仅一种结构元素（表格或列表），且仅在有助于阐明观点时使用
- 一个具体的例子/案例研究
- 以粗体见解开始
- 在创建章节内容前没有前言
- 在末尾引用来源
</质量检查>
"""

section_grader_instructions = """根据指定主题审查报告章节：

<报告主题>
{topic}
</报告主题>

<章节主题>
{section_topic}
</章节主题>

<章节内容>
{section}
</章节内容>

<任务>
评估章节内容是否充分涵盖了章节主题。

如果章节内容未能充分涵盖章节主题，请生成{number_of_follow_up_queries}个后续搜索查询以收集缺失信息。
</任务>

<格式>
    grade: Literal["pass","fail"] = Field(
        description="评估结果，表明响应是否满足要求（'pass'）或需要修改（'fail'）。"
    )
    follow_up_queries: List[SearchQuery] = Field(
        description="后续搜索查询列表。",
    )
</格式>
"""


query_writer_instructions="""你是一位专业技术写作专家，负责制定有针对性的网络搜索查询，以收集全面的信息用于撰写技术报告章节。

<报告主题>
{topic}
</报告主题>

<章节主题>
{section_topic}
</章节主题>

<任务>
你的目标是生成{number_of_queries}个搜索查询，帮助收集关于上述章节主题的全面信息。

这些查询应当：

1. 与主题相关
2. 探索主题的不同方面
3. 使用与章节主题相同语言

请确保查询足够具体，以找到高质量、相关的信息源。
</任务>
"""


final_section_writer_instructions="""您是一位专业技术作家，正在撰写一个综合报告其他部分信息的章节。

<报告主题>
{topic}
</报告主题>

<章节名称>
{section_name}
</章节名称>

<章节主题> 
{section_topic}
</章节主题>

<可用报告内容>
{context}
</可用报告内容>

<任务>
1. 章节特定方法：

对于引言：
- 使用#作为报告标题（Markdown格式）
- 限制在50-100字
- 使用简单清晰的语言
- 在1-2段中专注于报告的核心动机
- 使用清晰的叙述结构来介绍报告
- 不包含结构元素（无列表或表格）
- 不需要来源部分

对于结论/总结：
- 使用##作为章节标题（Markdown格式）
- 限制在100-150字
- 对于比较性报告：
    * 必须包含使用Markdown表格语法的重点比较表
    * 表格应提炼报告中的见解
    * 保持表格条目清晰简洁
- 对于非比较性报告： 
    * 仅在有助于提炼报告中的要点时使用一种结构元素：
    * 要么是使用Markdown表格语法比较报告中出现的项目的重点表格
    * 要么是使用正确Markdown列表语法的简短列表：
      - 使用`*`或`-`表示无序列表
      - 使用`1.`表示有序列表
      - 确保正确的缩进和间距
- 以具体的后续步骤或影响结尾
- 不需要来源部分

3. 写作方法：
- 使用具体细节而非一般性陈述
- 让每个词都有价值
- 专注于您最重要的一点
</任务>

<质量检查>
- 对于引言：50-100字限制，#用于报告标题，无结构元素，无来源部分
- 对于结论：100-150字限制，##用于章节标题，最多只有一个结构元素，无来源部分
- Markdown格式
- 不要在回复中包含字数统计或任何前言
</质量检查>"""


from typing import Annotated, List, TypedDict, Literal
from pydantic import BaseModel, Field
import operator

from langchain_core.messages import HumanMessage, SystemMessage

from langgraph.constants import Send
from langgraph.graph import START, END, StateGraph
from langgraph.types import interrupt, Command
from llm import *
from duckduckgo_search import DDGS
from functools import lru_cache
from pydantic import BaseModel, Field

class SearchQuery(BaseModel):
    search_query: str = Field(None, description="要进行网络搜索的query")
    
class Queries(BaseModel):
    queries: List[SearchQuery] = Field(
        description="搜索query列表",
    )


class ReportStateInput(TypedDict):
    topic: str # 报告主题
    
class ReportStateOutput(TypedDict):
    final_report: str # 最终报告

class Section(BaseModel):
    name: str = Field(
        description="报告此章节的名称。",
    )
    description: str = Field(
        description="本章节将涵盖的主要主题和概念的简要概述。",
    )
    research: bool = Field(
        description="是否需要为报告的这个章节进行网络研究。"
    )
    content: str = Field(
        description="章节的内容。"
    )   

class Sections(BaseModel):
    sections: List[Section] = Field(
        description="报告的章节。",
    )

class ReportState(TypedDict):
    topic: str # 报告主题    
    feedback_on_report_plan: str # 报告计划的反馈
    sections: list[Section] # 报告章节列表 
    completed_sections: Annotated[list, operator.add] # Send() API键
    report_sections_from_research: str # 从研究中完成的任何章节的字符串，用于编写最终章节
    final_report: str # 最终报告

class SectionState(TypedDict):
    topic: str # 报告主题
    section: Section # 报告章节  
    search_iterations: int # 已完成的搜索迭代次数
    search_queries: list[SearchQuery] # 搜索查询列表
    source_str: str # 来自网络搜索的格式化源内容字符串
    report_sections_from_research: str # 从研究中完成的任何章节的字符串，用于编写最终章节
    completed_sections: list[Section] # 在外部状态中用于Send() API的最终键

class SectionOutputState(TypedDict):
    completed_sections: list[Section] # 在外部状态中用于Send() API的最终键


@lru_cache()
def web_search(query):
    """
    使用DuckDuckGo搜索引擎执行网络搜索
    
    参数:
        query: 搜索查询字符串
        
    返回:
        搜索结果列表
    """
    results = DDGS(proxy=None).text(query, max_results=10)
    return results

def deduplicate_and_format_sources(search_response, max_tokens_per_source):
    """
    去重并格式化搜索结果
    
    参数:
        search_response: 搜索响应列表
        max_tokens_per_source: 每个源的最大令牌数
        
    返回:
        格式化的文本字符串
    """
    sources_list = []
    for response in search_response:
        sources_list.extend(response)
    unique_sources = {source['href']: source for source in sources_list}

    formatted_text = "Sources:\n\n"
    for i, source in enumerate(unique_sources.values(), 1):
        formatted_text += f"Source {source['title']}:\n===\n"
        formatted_text += f"URL: {source['href']}\n===\n"
        formatted_text += f"Most relevant content from source: {source['body']}\n===\n"
    return formatted_text.strip()
    
    
async def generate_report_plan(state: ReportState):
    """ 
    生成报告计划
    
    参数:
        state: 报告状态
        
    返回:
        包含报告章节的字典
    """
    topic = state["topic"]
    feedback = state.get("feedback_on_report_plan", None)

    number_of_queries = 2

    
    writer_model = init_chat_model(
        provider="zhipuai",
        model_name="glm-4-flash",
        temperature=0.0
    )
    structured_llm = writer_model.with_structured_output(Queries)

    system_instructions_query = report_planner_query_writer_instructions.format(topic=topic, report_organization=report_structure, number_of_queries=number_of_queries)
    results = structured_llm.invoke([SystemMessage(content=system_instructions_query)]+[HumanMessage(content="生成有助于规划报告章节的搜索查询。")])

    query_list = [query.search_query for query in results.queries]
    print(query_list)
    
    # 执行网络搜索
    search_results = [web_search(query) for query in query_list]
    source_str = deduplicate_and_format_sources(search_results, max_tokens_per_source=1000)

    feedback = ''
    system_instructions_sections = report_planner_instructions.format(topic=topic, report_organization=report_structure, context=source_str, feedback=feedback)
    planner_llm = init_chat_model(
        provider="zhipuai",
        model_name="glm-4-plus",
        temperature=0.0
    )

    structured_llm = planner_llm.bind_tools([Sections])
    report_sections = structured_llm.invoke([SystemMessage(content=system_instructions_sections)]+[HumanMessage(content="生成报告的章节")])
    tool_call = json.loads(report_sections.content)
    report_sections  = Sections(**tool_call)
    
    sections = report_sections.sections
    print(sections)
    return {"sections": sections}

def human_feedback(state: ReportState) -> Command[Literal["generate_report_plan","build_section_with_web_research"]]:
    """ 
    获取报告计划的人类反馈
    
    参数:
        state: 报告状态
        
    返回:
        命令对象，指示下一步操作
    """
    # 获取章节
    topic = state["topic"]
    sections = state['sections']
    sections_str = "\n\n".join(
        f"Section: {section.name}\n"
        f"Description: {section.description}\n"
        f"Research needed: {'Yes' if section.research else 'No'}\n"
        for section in sections
    )

    interrupt_message = f"""请对以下报告计划提供反馈。
                        \n\n{sections_str}\n\n
                        \n报告计划是否满足您的需求？输入'true'来批准报告计划，或提供反馈以重新生成报告计划："""
    feedback = interrupt(interrupt_message)

    if isinstance(feedback, bool) and feedback is True:
        # 如果反馈为True，则继续构建需要研究的章节
        return Command(goto=[
            Send("build_section_with_web_research", {"topic": topic, "section": s, "search_iterations": 0}) 
            for s in sections 
            if s.research
        ])
    
    elif isinstance(feedback, str):
        # 如果反馈是字符串，则重新生成报告计划
        return Command(goto="generate_report_plan", 
                       update={"feedback_on_report_plan": feedback})
    else:
        raise TypeError(f"Interrupt value of type {type(feedback)} is not supported.")


def generate_queries(state: SectionState):
    """ 
    为报告章节生成搜索查询
    
    参数:
        state: 章节状态
        
    返回:
        包含搜索查询的字典
    """
    topic = state["topic"]
    section = state["section"]
    number_of_queries = 3

    writer_model = init_chat_model(
        provider="zhipuai",
        model_name="glm-4-flash",
        temperature=0.0
    )
    structured_llm = writer_model.with_structured_output(Queries)


    system_instructions = query_writer_instructions.format(topic=topic, section_topic=section, number_of_queries=number_of_queries)
    queries = structured_llm.invoke([SystemMessage(content=system_instructions)]+[HumanMessage(content="Generate search queries on the provided topic.")])
    print(queries)
    return {"search_queries": queries.queries}


async def search_web(state: SectionState):
    """
    执行网络搜索并格式化结果
    
    参数:
        state: 章节状态
        
    返回:
        包含搜索结果和更新的搜索迭代次数的字典
    """
    search_queries = state["search_queries"]
    query_list = [query.search_query for query in search_queries]
    
    search_results = [DDGS(proxy=None).text(query, max_results=10) for query in query_list]
    source_str = deduplicate_and_format_sources(search_results, max_tokens_per_source=5000)
    return {"source_str": source_str, "search_iterations": state["search_iterations"] + 1}


class Feedback(BaseModel):
    grade: Literal["pass","fail"] = Field(
        description="评估结果，表明响应是否满足要求（'pass'）或需要修改（'fail'）。"
    )
    follow_up_queries: List[SearchQuery] = Field(
        description="后续搜索查询列表。",
    )
    
def write_section(state: SectionState) -> Command[Literal[END, "search_web"]]:
    """
    根据搜索结果编写章节内容
    
    参数:
        state: 章节状态
        
    返回:
        命令对象，指示下一步操作
    """
    topic = state["topic"]
    section = state["section"]
    source_str = state["source_str"]

    system_instructions = section_writer_instructions.format(topic=topic, section_name=section.name, section_topic=section.description, context=source_str, section_content=section.content)
 
    writer_model = init_chat_model(
        provider="zhipuai",
        model_name="glm-4-flash",
        temperature=0.0
    )
    
    section_content = writer_model.invoke([SystemMessage(content=system_instructions), HumanMessage(content="根据提供的资料生成一个报告章节。")])

    section.content = section_content.content

    section_grader_message = """评估报告并考虑针对缺失信息的后续问题。
                               如果评级为'pass'，则所有后续查询返回空字符串。
                               如果评级为'fail'，请提供具体的搜索查询以收集缺失信息。"""
    
    section_grader_instructions_formatted = section_grader_instructions.format(topic=topic, 
                                                                               section_topic=section.description,
                                                                               section=section.content, 
                                                                               number_of_follow_up_queries=2)

    
    reflection_model = init_chat_model(
        provider="zhipuai",
        model_name="glm-4-plus",
        temperature=0.0
    )

    reflection_result = reflection_model.bind_tools([Feedback]).invoke([SystemMessage(content=section_grader_instructions_formatted),
                                                                            HumanMessage(content=section_grader_message)])
    print(reflection_result)
    
    tool_call = json.loads(reflection_result.content)

    if tool_call['follow_up_queries'] and isinstance(tool_call['follow_up_queries'][0], str):
        tool_call['follow_up_queries'] = [{'search_query':_} for _ in tool_call['follow_up_queries'] ]
    
    feedback  = Feedback(**tool_call)
    # 如果评级为通过或已达到最大搜索迭代次数，则完成章节
    if feedback.grade == "pass" or state["search_iterations"] >= 2:
        return  Command(
        update={"completed_sections": [section]},
        goto=END
    )
    else:
        # 否则继续搜索
        return  Command(
        update={"search_queries": feedback.follow_up_queries, "section": section},
        goto="search_web"
        )




def write_final_sections(state: SectionState):
    """
    编写最终章节内容
    
    参数:
        state: 章节状态
        
    返回:
        包含完成章节的字典
    """
    topic = state["topic"]
    section = state["section"]
    completed_report_sections = state["report_sections_from_research"]
    
    system_instructions = final_section_writer_instructions.format(topic=topic, section_name=section.name, section_topic=section.description, context=completed_report_sections)

    writer_model = init_chat_model(
        provider="zhipuai",
        model_name="glm-4-flash",
        temperature=0.0
    )
    
    
    section_content = writer_model.invoke([SystemMessage(content=system_instructions),
                                           HumanMessage(content="根据提供的资料生成一个报告章节。")])
    
    section.content = section_content.content
    return {"completed_sections": [section]}


def format_sections(sections: list[Section]) -> str:
    """
    格式化章节列表为字符串
    
    参数:
        sections: 章节列表
        
    返回:
        格式化的章节字符串
    """
    formatted_str = ""
    for idx, section in enumerate(sections, 1):
        formatted_str += f"""
{'='*60}
Section {idx}: {section.name}
{'='*60}
Description:
{section.description}
Requires Research: 
{section.research}

Content:
{section.content if section.content else '[Not yet written]'}

"""
    return formatted_str

def gather_completed_sections(state: ReportState):
    """
    收集已完成的章节
    
    参数:
        state: 报告状态
        
    返回:
        包含已完成章节的字典
    """
    completed_sections = state["completed_sections"]
    completed_report_sections = format_sections(completed_sections)

    return {"report_sections_from_research": completed_report_sections}

def initiate_final_section_writing(state: ReportState):
    """ 
    使用Send API并行编写最终章节
    
    参数:
        state: 报告状态
        
    返回:
        Send命令列表
    """    
    return [
        Send("write_final_sections", {"topic": state["topic"], "section": s, "report_sections_from_research": state["report_sections_from_research"]}) 
        for s in state["sections"] 
        if not s.research
    ]

def compile_final_report(state: ReportState):
    """ 
    编译最终报告
    
    参数:
        state: 报告状态
        
    返回:
        包含最终报告的字典
    """    
    sections = state["sections"]
    completed_sections = {s.name: s.content for s in state["completed_sections"]}

    for section in sections:
        section.content = completed_sections[section.name]

    all_sections = "\n\n".join([s.content for s in sections])

    return {"final_report": all_sections}

# 报告章节子图 -- 

# 添加节点 
section_builder = StateGraph(SectionState, output=SectionOutputState)
section_builder.add_node("generate_queries", generate_queries)
section_builder.add_node("search_web", search_web)
section_builder.add_node("write_section", write_section)

# 添加边
section_builder.add_edge(START, "generate_queries")
section_builder.add_edge("generate_queries", "search_web")
section_builder.add_edge("search_web", "write_section")

# 外部图 -- 

# 添加节点
builder = StateGraph(ReportState, input=ReportStateInput, output=ReportStateOutput)
builder.add_node("generate_report_plan", generate_report_plan)
builder.add_node("human_feedback", human_feedback)
builder.add_node("build_section_with_web_research", section_builder.compile())
builder.add_node("gather_completed_sections", gather_completed_sections)
builder.add_node("write_final_sections", write_final_sections)
builder.add_node("compile_final_report", compile_final_report)

# 添加边
builder.add_edge(START, "generate_report_plan")
builder.add_edge("generate_report_plan", "human_feedback")
builder.add_edge("build_section_with_web_research", "gather_completed_sections")
builder.add_conditional_edges("gather_completed_sections", initiate_final_section_writing, ["write_final_sections"])
builder.add_edge("write_final_sections", "compile_final_report")
builder.add_edge("compile_final_report", END)

graph = builder.compile()