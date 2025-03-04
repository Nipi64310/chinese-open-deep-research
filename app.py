import streamlit as st
import uuid
import asyncio
import time
from langgraph.types import Command
from langgraph.checkpoint.memory import MemorySaver
from graph import builder

st.set_page_config(
    page_title="AI研究报告生成器",
    page_icon="📚",
    layout="wide"
)

# 自定义CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 1rem;
        color: #1E88E5;
    }
    .section-header {
        font-size: 1.8rem;
        font-weight: bold;
        margin-top: 2rem;
        margin-bottom: 1rem;
        color: #333;
    }
    .info-box {
        background-color: #E3F2FD;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .report-container {
        border: 1px solid #ddd;
        border-radius: 0.5rem;
        padding: 1.5rem;
        background-color: #f9f9f9;
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">AI研究报告生成器</div>', unsafe_allow_html=True)
st.markdown("基于LangGraph的自动研究报告生成工具，可以根据您的主题自动搜索网络并生成详细的研究报告。")

# 初始化会话状态
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "report_plan_approved" not in st.session_state:
    st.session_state.report_plan_approved = False
if "report_generated" not in st.session_state:
    st.session_state.report_generated = False
if "final_report" not in st.session_state:
    st.session_state.final_report = ""
if "memory" not in st.session_state:
    st.session_state.memory = MemorySaver()
if "graph" not in st.session_state:
    st.session_state.graph = builder.compile(checkpointer=st.session_state.memory)
if "sections_display" not in st.session_state:
    st.session_state.sections_display = ""
if "generating_plan" not in st.session_state:
    st.session_state.generating_plan = False
if "generating_report" not in st.session_state:
    st.session_state.generating_report = False
if "current_section" not in st.session_state:
    st.session_state.current_section = ""

# 创建线程配置
thread = {"configurable": {"thread_id": st.session_state.thread_id}}

# 定义异步函数来处理报告计划生成
async def generate_report_plan(topic):
    st.session_state.generating_plan = True
    
    async for event in st.session_state.graph.astream({"topic": topic}, thread, stream_mode="updates"):
        if '__interrupt__' in event:
            interrupt_value = event['__interrupt__'][0].value
            st.session_state.sections_display = interrupt_value
            break
    
    st.session_state.generating_plan = False
    return st.session_state.sections_display

# 定义异步函数来生成完整报告
async def generate_full_report():
    st.session_state.generating_report = True
    
    section_count = 0
    total_sections = 7  # 假设总共有7个部分
    
    async for event in st.session_state.graph.astream(Command(resume=True), thread, stream_mode="updates"):
        if 'build_section_with_web_research' in event:
            if 'completed_sections' in event['build_section_with_web_research']:
                section = event['build_section_with_web_research']['completed_sections'][0]
                st.session_state.current_section = f"正在生成: {section.name}"
                section_count += 1
            
        if 'compile_final_report' in event:
            if 'final_report' in event['compile_final_report']:
                st.session_state.final_report = event['compile_final_report']['final_report']
                st.session_state.current_section = "报告生成完成！"
                break
    
    st.session_state.generating_report = False
    return st.session_state.final_report

# 侧边栏
with st.sidebar:
    st.markdown("### 关于")
    st.markdown("""
    这个工具使用AI来自动生成研究报告。它会：
    1. 分析您的主题
    2. 创建报告结构
    3. 搜索网络获取信息
    4. 生成完整的研究报告
    """)
    
    st.markdown("### 使用说明")
    st.markdown("""
    1. 输入您感兴趣的研究主题
    2. 查看并批准生成的报告计划
    3. 等待AI生成完整报告
    4. 下载或复制最终报告
    """)
    
    if st.session_state.final_report:
        st.markdown("### 下载选项")
        st.download_button(
            label="下载报告 (Markdown)",
            data=st.session_state.final_report,
            file_name="research_report.md",
            mime="text/markdown"
        )

# 主界面
col1, col2 = st.columns([2, 1])

with col1:
    with st.form("topic_form"):
        topic = st.text_input("请输入研究主题", placeholder="例如：openai deep research")
        submit_button = st.form_submit_button("生成报告计划")

    # 处理报告计划生成
    if submit_button and topic:
        st.session_state.report_plan_approved = False
        st.session_state.report_generated = False
        st.session_state.final_report = ""
        st.session_state.sections_display = ""
        st.session_state.thread_id = str(uuid.uuid4())
        thread = {"configurable": {"thread_id": st.session_state.thread_id}}
        
        with st.spinner("正在分析主题并生成报告计划..."):
            asyncio.run(generate_report_plan(topic))

with col2:
    if st.session_state.generating_plan:
        st.info("正在生成报告计划，请稍候...")
    
    if st.session_state.generating_report:
        st.info(st.session_state.current_section)
        
        # 添加进度条
        progress_placeholder = st.empty()
        progress_bar = progress_placeholder.progress(0)
        
        # 模拟进度
        if 'progress_value' not in st.session_state:
            st.session_state.progress_value = 0
            
        # 更新进度条
        def update_progress():
            while st.session_state.generating_report and st.session_state.progress_value < 100:
                time.sleep(0.1)
                if st.session_state.progress_value < 95:  # 最多到95%，留5%给最终完成
                    st.session_state.progress_value += 0.2
                    progress_bar.progress(int(st.session_state.progress_value))
            
            if not st.session_state.generating_report:
                progress_bar.progress(100)
                
        # 在后台运行进度更新
        import threading
        progress_thread = threading.Thread(target=update_progress)
        progress_thread.start()

# 显示报告计划
if st.session_state.sections_display and not st.session_state.report_plan_approved:
    st.markdown('<div class="section-header">报告计划</div>', unsafe_allow_html=True)
    st.markdown('<div class="info-box">请查看以下报告计划，如果满意请点击"批准报告计划"按钮继续。</div>', unsafe_allow_html=True)
    
    st.markdown(st.session_state.sections_display)
    
    if st.button("批准报告计划", key="approve_plan"):
        st.session_state.report_plan_approved = True
        st.session_state.progress_value = 0
        
        with st.spinner("正在生成完整报告..."):
            asyncio.run(generate_full_report())

# 显示最终报告
if st.session_state.final_report:
    st.markdown('<div class="section-header">最终研究报告</div>', unsafe_allow_html=True)
    
    with st.expander("查看完整报告", expanded=True):
        st.markdown('<div class="report-container">', unsafe_allow_html=True)
        st.markdown(st.session_state.final_report)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # 添加复制按钮
    st.button("复制报告内容到剪贴板", 
              on_click=lambda: st.write('<script>navigator.clipboard.writeText(`' + 
                                       st.session_state.final_report.replace('`', '\\`') + 
                                       '`);</script>', unsafe_allow_html=True)) 