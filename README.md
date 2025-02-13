# PDF 문서 기반 질의응답 시스템

이 프로젝트는 "선진기업복지 업무메뉴얼" PDF 문서를 기반으로 질의응답을 제공하는 시스템입니다. 사용자는 PDF 문서에 대한 질문을 입력하면, 시스템은 해당 문서를 분석하여 정확한 답변을 제공합니다.

앱 주소: https://manual-prompt-i54gigmdkaicinzbvvfasm.streamlit.app

## 기능

* PDF 문서 로드 및 분석
* 사용자 질문 입력 및 처리
* 문서 기반으로 답변 생성 및 제공

## 기술 스택

* Llama Index: 문서 기반 질의응답 엔진
* Hugging Face: 언어 모델 및 임베딩 모델
* Torch: 딥 러닝 모델 구현
* Streamlit: 웹 애플리케이션 개발
* PyPDF: PDF 문서 읽기 및 분석


## 실행 방법

1. 이 프로젝트를 로컬에 클론합니다.
2. 필요한 라이브러리를 설치합니다. (`pip install -r requirements.txt`)
3. `app.py` 파일을 실행합니다. (`python app.py`)
4. 웹 브라우저에서 https://manual-prompt-i54gigmdkaicinzbvvfasm.streamlit.app로 접속합니다.
5. PDF 문서를 업로드하고 질문을 입력합니다.

## 주의 사항

* 이 시스템은 한국어로만 작동합니다.
* PDF 문서의 내용이 정확하고 완전해야 합니다.
* 시스템의 성능은 문서의 크기 및 복잡도에 따라 다를 수 있습니다.
