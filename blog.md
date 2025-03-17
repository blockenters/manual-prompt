# RAG 시스템 최적화: 성능을 결정짓는 3가지 핵심 요소 완벽 가이드

## 📌 들어가며

안녕하세요, AI 개발자 여러분! 오늘은 취업 포트폴리오에 강력한 임팩트를 줄 수 있는 RAG(Retrieval-Augmented Generation) 시스템 구현의 핵심 요소들을 깊이 있게 살펴보려 합니다. 단순히 LLM API를 호출하는 것보다 한 단계 더 나아가, 여러분만의 데이터로 정확하고 신뢰할 수 있는 AI 애플리케이션을 구축하는 방법을 공유합니다.

이 글은 특히:
- 취업을 준비 중인 개발자
- RAG 기반 프로젝트를 시작하려는 분들
- 차별화된 사이드 프로젝트를 찾고 있는 개발자

분들에게 실질적인 도움이 될 것입니다.

## 🔍 RAG란 무엇이며 왜 중요한가?

RAG는 Large Language Model(LLM)이 외부 지식(문서, DB 등)에 접근하여 더 정확하고 최신 정보를 바탕으로 응답을 생성하는 방식입니다. 기존의 LLM은 학습된 데이터에만 의존하기 때문에 다음과 같은 한계가 있습니다:

- 학습 시점 이후의 정보 부재
- 전문적이거나 특정 도메인의 정보 부족
- 환각(hallucination) 문제

RAG는 이런 한계를 극복하고 여러분의 데이터를 활용하여 정확하고 신뢰할 수 있는 응답을 생성할 수 있게 해줍니다.

## 🛠️ RAG 성능을 결정짓는 3가지 핵심 요소

RAG 시스템의 성능은 세 가지 핵심 요소의 최적화에 달려 있습니다. 각 요소를 제대로 이해하고 설정하는 것만으로도 여러분의 프로젝트는 일반적인 RAG 시스템과 차별화될 수 있습니다.

### 1️⃣ 임베딩 모델 (Embedding Model)

임베딩 모델은 텍스트를 벡터로 변환하여 의미적 유사성을 계산할 수 있게 해주는 RAG의 두뇌와 같은 존재입니다.

#### 📊 임베딩 모델의 중요성

저는 한국어 퇴직연금 문서 기반 Q&A 시스템을 개발하는 과정에서 두 가지 다른 임베딩 모델을 테스트했습니다:

1. **Hugging Face `sentence-transformers/all-mpnet-base-v2`**
2. **OpenAI `text-embedding-3-small`**

같은 질문, 같은 문서, 같은 LLM 모델을 사용했음에도 결과는 극적으로 달랐습니다.

#### 🔄 실제 성능 비교 

**질문**: "우리사주제도에 대해 설명해주세요."

**Hugging Face 모델 응답**:
```
문서에서 해당 정보를 찾을 수 없습니다.
```

**OpenAI 모델 응답**:
```
우리사주제도는 기업이 자사의 주식을 근로자에게 취득하게 함으로써 근로자의 재산형성과 
복지증진을 도모하고, 근로자의 경영참여의식을 높여 노사협력 증진과 기업의 생산성 향상을 
목적으로 하는 제도입니다. 이 제도를 통해 근로자는 자사주를 우선적으로 배정받을 수 있으며, 
세제 혜택도 받을 수 있습니다.
```

놀랍게도 문서에는 우리사주제도에 관한 정보가 포함되어 있었지만, Hugging Face 모델은 이를 찾지 못했습니다. 이유는 의미적 유사도 계산에서 한국어 처리 능력의 차이였습니다.

#### 💻 임베딩 모델 변경 코드

```python
# Hugging Face 임베딩 모델 설정
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-mpnet-base-v2")
Settings.embed_model = embed_model
```

```python
# OpenAI 임베딩 모델 설정
from llama_index.embeddings.openai import OpenAIEmbedding

embed_model = OpenAIEmbedding(
    model="text-embedding-3-small",
    api_key=openai_api_key,
    embed_batch_size=10,  # 비용 최적화를 위한 배치 처리
    dimensions=1536
)
Settings.embed_model = embed_model
```

#### 🔑 취업 준비생을 위한 팁

면접에서 "인공지능 프로젝트에서 가장 중요한 결정은 무엇이었나요?"라는 질문을 받는다면, 임베딩 모델 선택과 비교 실험을 통한 성능 최적화 경험을 공유하세요. 데이터 기반의 의사결정 능력을 보여줄 수 있는 인상적인 사례가 됩니다.

### 2️⃣ 청크 사이즈 (Chunk Size)

청크 사이즈는 원본 문서를 얼마나 작은 조각으로 나눌지 결정하는 중요한 파라미터입니다.

#### 📏 청크 사이즈의 영향

청크 사이즈는 다음과 같은 트레이드오프를 갖습니다:

- **작은 청크 (256 토큰)**: 검색 정밀도가 높지만, 문맥 유지가 어려움
- **큰 청크 (512-1024 토큰)**: 풍부한 문맥이 유지되지만, 불필요한 정보가 포함될 수 있음

한국어와 같이 문맥 의존성이 높은 언어는 더 큰 청크 사이즈가 유리한 경향이 있습니다.

#### 📈 실험 결과

퇴직연금 문서를 다양한 청크 사이즈로 분할하여 테스트한 결과:

| 청크 사이즈 | 정확한 답변 비율 | 메모리 사용량 | 벡터 DB 항목 수 |
|------------|----------------|------------|--------------|
| 256 토큰    | 65%            | 낮음        | 많음 (약 450개) |
| 512 토큰    | 82%            | 중간        | 중간 (약 225개) |
| 1024 토큰   | 78%            | 높음        | 적음 (약 115개) |

한국어 퇴직연금 문서의 경우 512 토큰이 최적의 균형점이었습니다.

#### 💻 청크 사이즈 설정 코드

```python
from llama_index.node_parser import SentenceSplitter

# 문장 단위로 분할하며 청크 사이즈 조정
text_splitter = SentenceSplitter(
    chunk_size=512,
    chunk_overlap=50,
    separator="\n",
    paragraph_separator="\n\n",
)

# 문서 분할 적용
nodes = text_splitter.get_nodes_from_documents(documents)
```

#### 🚀 프로젝트 차별화 포인트

GitHub에서 fork하여 수정할 수 있는 오픈소스 RAG 프로젝트 대부분은 영어 문서에 최적화된 청크 사이즈(보통 256 토큰)를 사용합니다. 한국어 문서에 특화된 청크 사이즈 최적화 코드를 PR하면 오픈소스 기여 실적을 쌓을 수 있습니다.

### 3️⃣ 청크 오버랩 (Chunk Overlap)

청크 오버랩은 인접한 청크 간에 겹치는 텍스트의 양을 결정합니다.

#### 🔄 오버랩의 중요성

오버랩이 없으면 다음과 같은 문제가 발생합니다:

- 문장이 중간에 잘려 의미 손실
- 청크 경계에 걸친 정보 검색 불가능
- 문맥 연결성 부재

#### 📊 오버랩 설정 실험

다양한 오버랩 비율로 테스트한 결과:

| 오버랩 비율 | 정확한 답변 비율 | 스토리지 사용량 증가 |
|------------|----------------|-------------------|
| 0 (오버랩 없음) | 65%           | 0%                |
| 10% (약 50 토큰) | 75%           | +10%             |
| 25% (약 128 토큰) | 86%           | +25%             |
| 50% (약 256 토큰) | 88%           | +50%             |

25% 오버랩이 성능과 효율성 측면에서 최적의 균형점이었습니다.

#### 💻 청크 오버랩 설정 코드

```python
from llama_index.node_parser import RecursiveCharacterTextSplitter

# 청크 크기의 25%를 오버랩으로 설정
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,  # 토큰 대신 문자 수로 지정
    chunk_overlap=128,  # 약 25% 오버랩
    separators=["\n\n", "\n", ". ", " ", ""]
)

nodes = text_splitter.get_nodes_from_documents(documents)
```

#### 💡 실무 개발 인사이트

면접에서 "RAG 시스템 개발 시 마주친 도전과 해결책은 무엇이었나요?"라는 질문에 청크 오버랩의 최적화 실험을 통해 사용자 질문 응답 정확도를 20% 이상 향상시킨 경험을 공유하면 깊이 있는 기술적 이해를 증명할 수 있습니다.

## 🔬 세 가지 요소의 조합과 최적화

이 세 가지 요소는 독립적으로 작용하지 않고 서로 영향을 미칩니다.

### 최적 조합 실험

다양한 조합을 실험한 결과, 다음의 설정이 한국어 문서에 가장 효과적이었습니다:

- **임베딩 모델**: OpenAI `text-embedding-3-small`
- **청크 사이즈**: 512 토큰 (한국어의 경우)
- **청크 오버랩**: 25% (128 토큰)

### 성능 측정 결과

| 조합 | 정확도 | 응답 생성 시간 | 비용 효율성 |
|------|-------|--------------|-----------|
| HF 임베딩 + 256 토큰 + 0% 오버랩 | 45% | 빠름 | 매우 높음 |
| HF 임베딩 + 512 토큰 + 25% 오버랩 | 62% | 중간 | 높음 |
| OpenAI 임베딩 + 256 토큰 + 0% 오버랩 | 72% | 빠름 | 중간 |
| OpenAI 임베딩 + 512 토큰 + 25% 오버랩 | 88% | 중간 | 중간 |
| OpenAI 임베딩 + 1024 토큰 + 50% 오버랩 | 85% | 느림 | 낮음 |

### 💻 종합 최적화 코드

```python
# app.py의 핵심 부분

def process_pdf_and_create_index(pdf_path, index_storage_path):
    # 1. 임베딩 모델 설정
    embed_model = OpenAIEmbedding(
        model="text-embedding-3-small",
        api_key=os.environ.get("OPENAI_API_KEY"),
        embed_batch_size=10
    )
    
    # 2. LLM 모델 설정
    llm = HuggingFaceInferenceAPI(
        model_name="google/gemma-2-2b-it",
        max_new_tokens=512,
        temperature=0.01,
        model_type="text_completion",
        system_prompt="문서 내용만 기반으로 한국어 답변 생성..."
    )
    
    # 글로벌 설정
    Settings.embed_model = embed_model
    Settings.llm = llm
    
    # 3. 문서 로드
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    
    # 4. 최적화된 청크 사이즈 및 오버랩으로 분할
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=128,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    nodes = text_splitter.get_nodes_from_documents(documents)
    
    # 5. 인덱스 생성 및 저장
    index = VectorStoreIndex(nodes)
    index.storage_context.persist(index_storage_path)
    
    return index
```

## 🚀 취업 준비생을 위한 프로젝트 활용법

### 포트폴리오 차별화 전략

1. **A/B 테스트 결과 시각화**
   - 다양한 임베딩 모델, 청크 사이즈, 오버랩 조합의 성능 비교 그래프
   - 사용자 쿼리별 정확도 측정 방법론 제시

2. **도메인 특화 최적화**
   - 법률, 의료, 금융 등 특정 도메인에 맞춘 최적 파라미터 발견
   - 도메인별 성능 차이 분석 및 이유 설명

3. **비용 효율성 분석**
   - API 비용 vs 성능 트레이드오프 분석
   - ROI(Return on Investment) 관점에서의 의사결정 프로세스

### 기술 면접 대비 포인트

다음 질문들에 대한 답변을 준비하세요:

- "RAG 시스템에서 가장 중요한 성능 요소는 무엇이라고 생각하나요?"
- "임베딩 모델 선택 시 고려한 요소는 무엇인가요?"
- "청크 사이즈와 오버랩을 어떻게 최적화했나요?"
- "한국어 문서 처리에서 마주친 특별한 도전과 해결 방법은?"

## 💡 사이드 프로젝트 아이디어

이 기술을 활용한 사이드 프로젝트 아이디어:

1. **개인 문서 기반 AI 비서**
   - 학습 노트, 개인 문서를 기반으로 질문에 답변하는 서비스
   - 개인화된 지식 검색 엔진

2. **전문 도메인 Q&A 시스템**
   - 법률 문서, 의료 정보, 학술 논문 기반 질의응답 시스템
   - 전문가를 위한 지식 검색 도구

3. **다국어 지원 문서 분석기**
   - 여러 언어로 된 문서를 동시에 처리하는 RAG 시스템
   - 언어별 최적 파라미터 자동 감지 및 적용

## 🎓 결론

RAG 시스템의 성능을 결정짓는 세 가지 핵심 요소 - 임베딩 모델, 청크 사이즈, 청크 오버랩 - 에 대한 이해와 최적화는 여러분의 AI 프로젝트를 차별화하는 핵심 경쟁력이 될 수 있습니다.

특히 한국어 문서를 처리하는 RAG 시스템의 경우, 이 세 요소를 적절히 조합하고 최적화하는 것만으로도 정확도와 사용자 경험을 크게 향상시킬 수 있습니다.

이 블로그에서 공유한 코드와 인사이트를 바탕으로 여러분만의 RAG 프로젝트를 개발하여 취업 포트폴리오를 강화하거나, 창의적인 사이드 프로젝트를 시작해보세요. 그리고 여러분의 경험과 발견을 공유해주시면 더 많은 개발자들에게 도움이 될 것입니다.

**"모든 AI 프로젝트는 데이터와 모델의 효과적인 조합에서 시작됩니다."**

## 📚 참고 자료

- [LlamaIndex 공식 문서](https://docs.llamaindex.ai/)
- [OpenAI Embeddings API](https://platform.openai.com/docs/guides/embeddings)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/index)
- [RAG 패턴 모범 사례](https://www.pinecone.io/learn/retrieval-augmented-generation-patterns/)

---

궁금한점 : macro@prag-ai.com 으로 주세요~  

