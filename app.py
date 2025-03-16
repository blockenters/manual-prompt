import streamlit as st
import os
from llama_index.core import (
    Document, 
    VectorStoreIndex, 
    Settings, 
    StorageContext,
    load_index_from_storage
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI
# OpenAI 임베딩 모델 임포트
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor
from pypdf import PdfReader

# Streamlit 파일 감시 기능 비활성화
os.environ['STREAMLIT_SERVER_WATCH_DIRS'] = 'false'

@st.cache_resource  # 캐싱 추가
def get_huggingface_token():
    """환경 변수 또는 streamlit secrets에서 토큰을 가져오는 함수"""
    token = os.environ.get("HUGGINGFACE_API_TOKEN")
    if token is None:  # 환경 변수에 없으면 streamlit secrets에서 시도
        try:
            token = st.secrets["HUGGINGFACE_API_TOKEN"]
        except:
            st.error("HUGGINGFACE_API_TOKEN이 설정되지 않았습니다.")
            return None
    return token

@st.cache_resource  # 캐싱 추가
def get_openai_api_key():
    """환경 변수 또는 streamlit secrets에서 OpenAI API 키를 가져오는 함수"""
    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key is None:  # 환경 변수에 없으면 streamlit secrets에서 시도
        try:
            api_key = st.secrets["OPENAI_API_KEY"]
        except:
            st.error("OPENAI_API_KEY가 설정되지 않았습니다.")
            return None
    return api_key

@st.cache_resource  # 캐싱 추가
def initialize_models():
    """모델 초기화 함수"""
    # model_name = "mistralai/Mistral-7B-Instruct-v0.2"
    model_name = "google/gemma-2-2b-it"
    
    # Hugging Face 토큰 가져오기
    hf_token = get_huggingface_token()
    if hf_token is None:
        return None, None
    
    # OpenAI API 키 가져오기
    openai_api_key = get_openai_api_key()
    if openai_api_key is None:
        st.error("OpenAI API 키가 없습니다. API 키를 설정해주세요.")
        return None, None
        
    llm = HuggingFaceInferenceAPI(
        model_name=model_name,
        max_new_tokens=512,
        temperature=0.01,
        model_type="text_completion",
        system_prompt="당신은 주어진 문서 내용만 기반으로 답변하는 AI 어시스턴트입니다. 검색된 문서 조각들 내용을 활용하여 질문에 답변하세요. 검색된 문서에 관련 정보가 있다면 반드시 그 내용을 기반으로 답변을 생성하세요. 문서에 명확히 언급된 내용을 우선으로 답변하고, 문서에 있는 내용만 참고하여 한국어로 명확하고 정확하게 답변해주세요. 답변을 생성할 때 관련 문서 내용을 직접 인용하고, 문서에 없는 내용은 추가하지 마세요.",
        token=hf_token
    )
    
    # OpenAI 임베딩 모델
    embed_model = OpenAIEmbedding(
        model="text-embedding-3-small",  # OpenAI의 최신 임베딩 모델 사용
        api_key=openai_api_key,
        embed_batch_size=10,  # 배치 처리 크기 설정
        dimensions=1536  # 임베딩 벡터 차원 수
    )
    
    # 전역 설정 업데이트
    Settings.llm = llm
    Settings.embed_model = embed_model
    
    return llm, embed_model

@st.cache_resource  # 캐싱 추가
def process_pdf_and_create_index(pdf_path):
    """PDF 파일을 처리하고 인덱스를 생성하는 함수"""
    # PDF 파일 로드 및 텍스트 추출
    pdf_reader = PdfReader(pdf_path)
    text_chunks = []
    
    # 모든 페이지의 텍스트 추출
    for page_num, page in enumerate(pdf_reader.pages):
        text = page.extract_text()
        if text.strip():  # 빈 페이지 제외
            # 페이지 번호 정보 포함
            text_with_metadata = f"[페이지 {page_num + 1}] {text}"
            text_chunks.append(text_with_metadata)
    
    # LlamaIndex Document 객체 생성
    documents = [Document(text=chunk) for chunk in text_chunks]
    
    # 문서 분할 (청킹)
    node_parser = SentenceSplitter(
        chunk_size=512,       # 청크 크기
        chunk_overlap=50,     # 오버랩
        paragraph_separator="\n\n",
        secondary_chunking_regex="(?<=\. )",
        include_metadata=True,
        include_prev_next_rel=True  # 이전/다음 청크 관계 포함
    )
    
    # 모델 초기화 및 설정
    llm, embed_model = initialize_models()
    
    # Settings 설정
    Settings.llm = llm
    Settings.embed_model = embed_model
    Settings.node_parser = node_parser
    
    # 노드 파싱
    nodes = node_parser.get_nodes_from_documents(documents)
    
    # 인덱스 생성
    index = VectorStoreIndex(nodes)
    
    # 인덱스 저장
    index_dir = "pdf_index_storage"
    os.makedirs(index_dir, exist_ok=True)
    index.storage_context.persist(persist_dir=index_dir)
    
    return index

@st.cache_resource
def load_saved_index():
    """저장된 인덱스 로드"""
    index_dir = "pdf_index_storage"
    if os.path.exists(index_dir):
        # 인덱스 로드
        storage_context = StorageContext.from_defaults(persist_dir=index_dir)
        index = load_index_from_storage(storage_context)
        return index
    return None

def create_optimized_query_engine(index):
    """최적화된 쿼리 엔진 생성"""
    # 검색기 설정 (Top-k 파라미터 조정)
    retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=5,  # 관련성 높은 문서 5개로 조정
        vector_store_query_mode="default"
    )
    
    # 유사도 임계값 설정
    node_postprocessors = [SimilarityPostprocessor(similarity_cutoff=0.2)]
    
    # 쿼리 엔진 생성
    query_engine = RetrieverQueryEngine.from_args(
        retriever=retriever,
        node_postprocessors=node_postprocessors,
        response_mode="compact",  # 모든 관련 문서를 한번에 고려
        response_kwargs={"verbose": True}  # 디버깅을 위한 verbose 모드
    )
    
    return query_engine

def main():
    st.title('PDF 문서 기반 질의응답 시스템')
    st.write("선진기업복지 업무메뉴얼을 기반으로 질의응답을 제공합니다.")

    # 모델 초기화
    llm, embed_model = initialize_models()
    
    if llm is None or embed_model is None:
        st.error("모델 초기화에 실패했습니다. API 토큰을 확인해주세요.")
        return
    
    # PDF 경로 설정
    pdf_path = "선진기업복지_업무매뉴얼.pdf"
    
    # 인덱스 로드 또는 생성
    index = load_saved_index()
    if index is None:
        with st.spinner('PDF 파일을 처리하고 인덱스를 생성하고 있습니다...'):
            index = process_pdf_and_create_index(pdf_path)
    
    if index is not None:
        # 전역 설정 업데이트
        Settings.llm = llm
        Settings.embed_model = embed_model
        
        # 최적화된 쿼리 엔진 생성
        query_engine = create_optimized_query_engine(index)
        
        # 사용자 입력 받기
        user_question = st.text_input("질문을 입력해주세요:")
        
        if user_question:
            try:
                with st.spinner('답변을 생성하고 있습니다...'):
                    # 검색된 문서 조각 표시 설정
                    st.subheader("검색된 관련 문서:")
                    with st.expander("문서 내용 보기", expanded=False):
                        retriever = VectorIndexRetriever(
                            index=index,
                            similarity_top_k=5
                        )
                        nodes = retriever.retrieve(user_question)
                        for i, node in enumerate(nodes):
                            st.markdown(f"**관련 문서 {i+1}** (유사도: {node.score:.4f})")
                            st.text(node.node.text[:500] + "..." if len(node.node.text) > 500 else node.node.text)
                            st.markdown("---")
                    
                    # 질문에 대한 답변 생성
                    response = query_engine.query(user_question)
                    st.subheader("답변:")
                    st.info(response.response)
                    
                    # 소스 문서 정보 표시
                    if hasattr(response, 'source_nodes') and response.source_nodes:
                        st.subheader("참조 소스:")
                        for i, source_node in enumerate(response.source_nodes):
                            with st.expander(f"소스 {i+1}", expanded=True):  # 소스를 기본적으로 확장하여 표시
                                st.text(source_node.node.text[:500] + "..." if len(source_node.node.text) > 500 else source_node.node.text)
                                st.markdown(f"**유사도 점수**: {source_node.score:.4f}")
            except Exception as e:
                st.error(f"답변 생성 중 오류가 발생했습니다: {str(e)}")
                st.error("잠시 후 다시 시도해주세요.")

if __name__ == "__main__":
    main()