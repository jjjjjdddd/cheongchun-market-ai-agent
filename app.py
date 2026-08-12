from pathlib import Path
from typing import TypedDict

import pandas as pd
import streamlit as st

from langchain_core.documents import Document
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
from langgraph.graph import StateGraph, START, END


# =========================================================
# 1. 기본 설정
# =========================================================
BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / "data" / "products.csv"


class AgentState(TypedDict):
    question: str
    context: list[Document]
    answer: str


# =========================================================
# 2. CSV 상품 데이터를 벡터 DB로 변환
# =========================================================
@st.cache_resource
def get_vectorstore():
    df = pd.read_csv(DATA_PATH)

    docs = []

    for _, row in df.iterrows():
        text = (
            f"상품명: {row['product_name']}\n"
            f"카테고리: {row['category']}\n"
            f"상태: {row['condition']}\n"
            f"출시연도: {row['year']}\n"
            f"정가: {int(row['original_price']):,}원\n"
            f"판매희망가: {int(row['asking_price']):,}원\n"
            f"참고 시세 범위: {row['market_range']}원\n"
            f"거래 지역: {row['location']}\n"
            f"특이사항: {row['note']}"
        )

        docs.append(
            Document(
                page_content=text,
                metadata={"product_name": row["product_name"]}
            )
        )

    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    return Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        collection_name="cheongchun_market"
    )


# =========================================================
# 3. AI Agent Workflow
# =========================================================
def make_graph(vectorstore):

    def retrieve(state: AgentState):
        question = state["question"].replace(" ", "")

        # 현재 products.csv에 실제로 등록된 제품군
        product_groups = [
            "아이패드",
            "맥북",
            "갤럭시버즈",
            "닌텐도스위치"
        ]

        matched_group = None

        # 사용자가 입력한 질문에 등록 제품군이 있는지 확인
        for group in product_groups:
            if group in question:
                matched_group = group
                break

        # 등록되지 않은 제품은 검색하지 않음
        # 예: 에어팟 프로 2 → context가 빈 리스트가 됨
        if matched_group is None:
            return {"context": []}

        # 등록된 제품군일 때만 벡터 DB에서 유사 상품 검색
        results = vectorstore.similarity_search(
            state["question"],
            k=3
        )

        # 검색 결과 중 같은 제품군만 근거로 사용
        filtered_docs = []

        for doc in results:
            product_name = doc.metadata["product_name"].replace(" ", "")

            if matched_group in product_name:
                filtered_docs.append(doc)

        return {"context": filtered_docs}

    def generate(state: AgentState):
        # 검색 결과가 없을 때는 LLM이 임의로 가격을 만들지 않음
        if not state["context"]:
            return {
                "answer": """
현재 등록된 상품 데이터에서는 해당 제품의 정보를 찾지 못했습니다.

정확한 가격 판단을 위해서는 다음 정보가 필요합니다.

- 정확한 모델명과 용량
- 출시 연도
- 제품 상태(A급, B급 등)
- 구성품 포함 여부
- 희망 거래 가격

현재 데이터에 없는 제품이므로, 임의의 시세를 안내하지 않았습니다.
"""
            }

        context_text = "\n\n---\n\n".join(
            doc.page_content for doc in state["context"]
        )

        prompt = f"""당신은 '청춘마켓 안심거래 도우미'입니다.

아래 [검색된 상품 정보]만 근거로 답변하세요.

규칙:
1) 제공된 정보에 없는 사실, 실시간 시세, 안전성을 단정하지 마세요.
2) 가격을 물으면 참고 시세 범위와 판단 근거를 짧게 설명하세요.
3) 상태, 구성품, 거래 전 확인할 항목을 함께 안내하세요.
4) 답변 마지막에는 반드시 '근거: 내부 상품 데이터'를 적으세요.

[검색된 상품 정보]
{context_text}

[사용자 질문]
{state['question']}
"""

        model = ChatOllama(
            model="qwen2.5:3b",
            temperature=0
        )

        result = model.invoke(prompt)

        return {"answer": result.content}

    graph = StateGraph(AgentState)

    graph.add_node("retrieve", retrieve)
    graph.add_node("generate", generate)

    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)

    return graph.compile()


# =========================================================
# 4. Streamlit 화면
# =========================================================
st.set_page_config(
    page_title="청춘마켓 안심거래 도우미",
    page_icon="🛡️"
)

st.title("🛡️ 청춘마켓 안심거래 도우미")
st.caption("상품 정보 검색을 바탕으로 가격 판단을 돕는 AI 거래 가이드")

st.info("로컬 Ollama 모델로 실행 중입니다. API 키와 결제가 필요하지 않습니다.")


# 예시 질문 선택 시 입력창에 문장을 자동 반영하는 함수
def apply_example():
    selected = st.session_state.example_question

    if selected != "직접 입력":
        st.session_state.user_question = selected


try:
    vectorstore = get_vectorstore()
    agent = make_graph(vectorstore)

except Exception as error:
    st.error(f"초기화 오류: {error}")
    st.stop()


examples = [
    "직접 입력",
    "아이패드 9세대 64GB를 28만원에 사도 될까요?",
    "맥북 에어 M1을 살 때 확인할 점은 무엇인가요?",
    "갤럭시 버즈2의 참고 시세를 알려줘"
]

st.selectbox(
    "예시 질문",
    examples,
    key="example_question",
    on_change=apply_example
)

question = st.text_input(
    "궁금한 상품이나 거래 조건을 입력하세요",
    key="user_question",
    placeholder="예: 아이패드 9세대 64GB를 28만원에 사도 될까요?"
)


if st.button("가격·거래 정보 확인하기", type="primary"):
    if not question.strip():
        st.warning("질문을 입력하거나 예시 질문을 선택해 주세요.")

    else:
        with st.spinner("상품 정보를 검색하고 거래 가이드를 만드는 중입니다..."):
            result = agent.invoke({"question": question})

        st.subheader("AI 거래 가이드")
        st.write(result["answer"])

        # 근거 데이터가 있을 때만 표시
        if result["context"]:
            with st.expander("답변에 사용한 상품 정보 보기"):
                for i, doc in enumerate(result["context"], 1):
                    st.markdown(
                        f"**근거 {i} — {doc.metadata['product_name']}**"
                    )
                    st.code(doc.page_content, language=None)

        else:
            st.info(
                "등록된 데이터에서 일치하는 상품을 찾지 못해 "
                "근거 상품 정보는 표시하지 않습니다."
            )


st.divider()

st.caption(
    "안내: 이 서비스는 등록된 데모 상품 데이터를 기반으로 합니다. "
    "실제 거래 전 제품 상태·구성품·거래 안전성을 직접 확인하세요."
)
