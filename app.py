import streamlit as st
from pypdf import PdfReader
import torch
from llama_index.core import Document, VectorStoreIndex, Settings
from llama_index.llms.huggingface import HuggingFaceInferenceAPI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from huggingface_hub import snapshot_download
from llama_index.core import StorageContext, load_index_from_storage
import os

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
def initialize_models():
    model_name = "mistralai/Mistral-7B-Instruct-v0.2"
    
    token = get_huggingface_token()
    if token is None:
        return None, None
        
    llm = HuggingFaceInferenceAPI(
        model_name=model_name,
        max_new_tokens=512,
        temperature=0,
        system_prompt="당신은 한국어로 대답하는 AI 어시스턴트 입니다. 주어진 질문에 대해서만 한국어로 명확하고 정확하게 답변해주세요. 이전 대화 내용은 포함하지 마세요.",
        token=token
    )

    embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-mpnet-base-v2")
    
    Settings.llm = llm
    Settings.embed_model = embed_model
    
    return llm, embed_model

@st.cache_resource  # 캐싱 추가
def get_index_from_huggingface():
    try:
        repo_id = "blockenters/manula-index2"
        local_dir = "manual_index_storage"
        
        token = get_huggingface_token()
        if token is None:
            return None
            
        snapshot_download(
            repo_id=repo_id, 
            repo_type="dataset", 
            local_dir=local_dir,
            token=token
        )
        
        # 인덱스 로드
        storage_context = StorageContext.from_defaults(persist_dir=local_dir)
        index = load_index_from_storage(storage_context)
        return index
    except Exception as e:
        st.error(f"인덱스 로드 중 오류가 발생했습니다: {str(e)}")
        return None

def main():
    st.title('PDF 문서 기반 질의응답 시스템')

    # 모델 초기화 (먼저 실행)
    llm, embed_model = initialize_models()
    
    index = get_index_from_huggingface()
    
    if index is not None:        
        
        # 쿼리 엔진 생성
        query_engine = index.as_query_engine(response_mode="compact")
        
        # 사용자 입력 받기
        user_question = st.text_input("질문을 입력해주세요:")
        
        if user_question:
            with st.spinner('답변을 생성하고 있습니다...'):
                response = query_engine.query(user_question)
                st.write("답변:")
                st.info(" " + response.response)

if __name__ == "__main__":
    main()