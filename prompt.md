@유튜브 영상 글작성.md 이 글 작성 규칙을 참고하여, 내가 지금 한 플젝 내용을 작성할건데, 
먼저 타겟은, 실전 상용화 RAG 시스템을 구축하려는 취준생이나 현업 개발자이고,
이 사람들에게 상용 RAG를 구축하려면 3가지를 잘 해야 한다는걸 설명하고 싶어.
그 세가지란, 임베딩 모델을 어떤걸 사용하냐에 따라 성능이 달라지고, 청킹사이즈, 그리고 오버랩 사이즈가 성능에 중요하다는 것을 알려주는 거야.
따라서 사례로는, 내가 직접 허깅페이스의  임베딩 모델(sentence-transformers/all-MiniLM-L6-v2) 과 오픈ai의 text-embedding-3-small 임베딩 모델을 사용해봤고, 청킹사이즈는 256과 512를 사용해 봤다는거야. 그리고 오버랩은 50으로 사용했어. LLM은 google/gemma-2-2b-it 를 사용했어. 그래서 가장 성능좋은 것은, 오픈에이아이의 임베딩모델과 청킹사이즈는 512, 오버랩은 50일때, 실제 pdf 와 비교하니까 가장 성능이 좋았어. 이 내용을 니가 잘 정리해줘