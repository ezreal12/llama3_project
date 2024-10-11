# 초록

---

본 연구에서는 대규모 언어 모델(LLM)을 활용한 검색 증강 생성(Retrieval-Augmented Generation, RAG) 시스템을 구현하였다. 특히 Function Call 기능을 통해 웹 검색과 답변 생성을 유기적으로 연결하는 다중 에이전트 시스템을 설계하였다. LangGraph 프레임워크를 활용하여 Query, Generate, Router 세 가지 에이전트를 정의하고, 이들 간의 상호작용을 통해 사용자 질의에 대한 정확하고 최신의 정보를 제공할 수 있는 시스템을 구축하였다. 본 연구는 DuckDuckGo 검색 API와 LLaMA 3 모델을 기반으로 하며, Streamlit을 이용한 웹 인터페이스를 통해 사용자 접근성을 높였다. 연구 결과, 제안된 시스템은 실시간 웹 정보를 활용하여 사용자 질의에 대해 정확하고 상세한 답변을 제공할 수 있음을 확인하였다.

# 1. 서론

---

최근 대규모 언어 모델(Large Language Models, LLMs)의 발전으로 자연어 처리 분야에서 많은 진전이 이루어졌다. 그러나 LLM의 한계점 중 하나는 학습 데이터의 시간적 제약으로 인한 최신 정보의 부재와 특정 도메인에 대한 깊이 있는 지식의 한계이다. 이러한 문제를 해결하기 위해 검색 증강 생성(Retrieval-Augmented Generation, RAG) 기법이 주목받고 있다.
RAG는 외부 데이터베이스나 웹 검색을 통해 얻은 정보를 LLM의 생성 과정에 통합하여, 더욱 정확하고 최신의 정보를 포함한 응답을 생성할 수 있게 한다. 본 연구에서는 이러한 RAG 시스템을 구현하되, 특히 Function Call 기능을 활용하여 웹 검색과 답변 생성을 유기적으로 연결하는 다중 에이전트 시스템을 설계하고자 한다.
본 논문의 구성은 다음과 같다. 2장에서는 RAG와 Function Call의 개념 및 필요성에 대해 설명한다. 3장에서는 LangGraph의 개념과 역할을 소개한다. 4장에서는 제안하는 시스템의 구조와 각 에이전트의 역할을 상세히 설명한다. 5장에서는 구현 결과와 실험을 통해 시스템의 성능을 평가한다. 마지막으로 6장에서는 결론 및 향후 연구 방향을 제시한다.

# 2. RAG와 Function Call

---

### 2.1 RAG의 개념과 필요성

RAG(Retrieval-Augmented Generation)는 검색을 통해 외부 데이터베이스에서 정보를 추출하고, 이를 바탕으로 대형 언어 모델(LLM)이 생성형 응답을 제공하는 시스템을 의미한다. RAG는 단순한 텍스트 생성 모델보다 정확성과 신뢰성이 높으며, 특히 최신 정보를 반영할 수 있다는 점에서 큰 장점을 가진다.

RAG의 필요성은 크게 세 가지 측면에서 설명할 수 있다:

1. 전문성 강화: RAG는 특정 기업의 고유 업무 영역에 대한 구체적이고 전문적인 정보를 제공할 수 있다. 외부 데이터베이스에서 관련 정보를 검색하여 더 정확한 답변을 생성할 수 있기 때문이다.
2. 최신 정보 반영: 생성형 AI의 최종 학습 시점 이후 발생하는 정보도 RAG를 통해 반영할 수 있다. 실시간으로 외부 인터넷 자료를 검색하고 그 정보를 활용할 수 있으므로, 최신 데이터를 반영한 응답을 생성할 수 있다.
3. 정보의 신뢰성 향상: RAG 시스템은 검색된 정보의 출처를 함께 제시할 수 있어, 정보의 신뢰성과 투명성을 높일 수 있다. 이는 기업이나 기관에서 제공된 정보를 신뢰하는 데 큰 도움이 된다.

### 2.2 Function Call의 개념과 역할

Function Call은 대규모 언어 모델(LLM)이 외부 툴이나 API와 상호 작용할 수 있도록 설계된 기능이다. 이 기능을 통해 LLM은 자연어를 분석하여 적절한 함수 호출을 수행할 수 있으며, 이를 통해 단순한 텍스트 생성을 넘어 외부 시스템과 연동된 복합적인 작업을 수행할 수 있게 된다.

Function Call의 주요 역할은 다음과 같다:

1. 외부 시스템 연동: LLM이 외부 API나 데이터베이스와 상호작용할 수 있게 한다.
2. 복잡한 작업 수행: 여러 단계의 작업이나 조건부 실행이 필요한 복잡한 태스크를 수행할 수 있다.
3. 정확성 향상: 특정 도메인의 전문 지식이나 최신 정보가 필요한 경우, 관련 함수를 호출하여 정확한 정보를 얻을 수 있다.
4. 유연성 제공: 다양한 외부 툴과 연동하여 LLM의 기능을 확장할 수 있다.

본 연구에서는 이러한 Function Call 기능을 활용하여 웹 검색과 답변 생성을 유기적으로 연결하는 RAG 시스템을 구현하고자 한다.

# 3. LangGraph의 개념 및 역할

---

### 3.1 LangGraph 소개

LangGraph는 LLM(대형 언어 모델)을 활용한 상태 기반(Stateful) 및 다중 에이전트 애플리케이션을 구축하는 데 사용되는 라이브러리이다. 주로 에이전트 및 다중 에이전트 워크플로우를 생성하는 데 활용되며, 다른 LLM 프레임워크와 비교해 '사이클(cycles)', '제어 가능성(controllability)', '지속성(persistence)'이라는 세 가지 핵심적인 이점을 제공한다.

### 3.2 LangGraph의 주요 특징

1. 사이클과 분기 처리: LangGraph는 반복 루프와 조건문을 애플리케이션에 구현할 수 있어, 에이전트 기반 시스템에서 다양한 흐름을 유연하게 정의할 수 있다. 이는 DAG(Directed Acyclic Graph) 기반의 솔루션과 차별화되는 특징이다.
2. 지속성: 각 그래프의 실행 단계마다 상태를 자동으로 저장하고, 언제든지 작업을 중단하거나 재개할 수 있다. 이를 통해 에러 복구, 시간 여행, 사람의 개입 등 다양한 고급 기능을 지원한다.
3. 제어 가능성: LangGraph는 애플리케이션의 흐름과 상태를 세밀하게 제어할 수 있는 저수준 프레임워크로, 신뢰할 수 있는 에이전트를 구축하는 데 필수적인 제어 기능을 제공한다.
4. 사람의 개입(Human-in-the-Loop): 그래프 실행을 중단하고, 에이전트가 계획한 다음 행동을 사용자가 승인하거나 수정할 수 있는 기능을 제공한다.
5. 스트리밍 지원: 각 노드에서 생성된 출력을 스트리밍으로 처리하여, 결과가 나오는 즉시 실시간으로 확인할 수 있다.

### 3.3 LangGraph의 설계 영감

LangGraph는 Pregel 및 Apache Beam에서 영감을 받았으며, 그 인터페이스는 네트워크 분석 라이브러리인 NetworkX에서 영감을 받아 설계되었다. 이러한 설계는 복잡한 워크플로우를 효과적으로 관리하고 실행할 수 있게 한다.

3.4 LangGraph와 LangChain의 관계

LangGraph는 LangChain Inc.에서 개발했지만, LangChain 없이도 독립적으로 사용할 수 있다. 그러나 LangChain 및 LangSmith와 원활하게 통합될 수 있어, 필요에 따라 이들 도구를 함께 활용할 수 있는 유연성을 제공한다.

# 4. 시스템 구조

---

### 4.1 시스템 개요

본 연구에서 제안하는 시스템은 세 개의 LLM 에이전트를 정의하고, 이들 간의 상호작용을 통해 사용자의 질문에 대한 정확하고 최신의 정보를 제공하는 것을 목표로 한다. 시스템의 주요 구성 요소는 다음과 같다:

1. Query 에이전트
2. Generate 에이전트
3. Router 에이전트
4. DuckDuckGo Web Search Tool
5. Local LLM (LLaMA 3)
6. LangGraph function call
7. Streamlit Web UI

시스템의 주요 흐름을 그림으로 요약하면 다음과 같다:
![LLM_RAG_%EC%97%90%EC%9D%B4%EC%A0%84%ED%8A%B8_(3)](https://github.com/user-attachments/assets/2f96e966-fd3a-4a0a-8f44-ec80d75f717f)

### 4.2 각 에이전트의 역할과 정의

4.2.1 Query 에이전트

Query 에이전트는 입력받은 context를 인터넷 검색에 적절한 검색어로 변환하는 역할을 수행한다. 변환된 검색어는 다른 LLM 에이전트가 이용할 수 있도록 JSON 형태로 반환된다.

Query 에이전트의 프롬프트는 다음과 같다:

```
<|begin_of_text|>

<|start_header_id|>system<|end_header_id|>

You are an expert at crafting web search queries for research questions.
More often than not, a user will ask a basic question that they wish to learn more about,
however it might not be in the best format.
Reword their query to be the most effective web search string possible.
Return the JSON with a single key 'query' with no preamble or explanation.

Question to transform: {question}

<|eot_id|>

<|start_header_id|>assistant<|end_header_id|>

```

4.2.2 Generate 에이전트

Generate 에이전트는 입력받은 context를 기반으로 LLM의 답변을 생성하여 최종 답변을 반환한다. 입력되는 context는 주로 DuckDuckGo API를 통해 검색된 데이터이며, 사용자의 질문에 웹 검색이 필요 없다고 Router 에이전트가 판단하는 경우 사용자의 질문이 그대로 context로 입력될 수 있다.

Generate 에이전트의 프롬프트는 다음과 같다:

```
<|begin_of_text|>

<|start_header_id|>system<|end_header_id|>

You are an AI assistant for Research Question Tasks, that synthesizes web search results.
Strictly use the following pieces of web search context to answer the question.
If you don't know the answer, just say that you don't know.
Keep the answer concise, but provide all of the details you can in the form of a research report.
Only make direct references to material if provided in the context.

<|eot_id|>

<|start_header_id|>user<|end_header_id|>

Question: {question}
Web Search Context: {context}
Answer:

<|eot_id|>

<|start_header_id|>assistant<|end_header_id|>

```

4.2.3 Router 에이전트

Router 에이전트는 입력받은 context를 기반으로 웹 검색을 수행할지, 최종 답변을 생성할지 결정한다. 입력받은 context에 검색이 필요한 경우 'web_search'를 반환하고 답변 생성이 필요한 경우 'generate'를 반환한다.

Router 에이전트의 프롬프트의 프롬프트는 다음과 같다:

```
<|begin_of_text|>

<|start_header_id|>system<|end_header_id|>

You are an expert at routing a user question to either the generation stage or web search.
Use the web search for questions that require more context for a better answer, or recent events.
Otherwise, you can skip and go straight to the generation phase to respond.
You do not need to be stringent with the keywords in the question related to these topics.
Give a binary choice 'web_search' or 'generate' based on the question.
Return the JSON with a single key 'choice' with no preamble or explanation.

Question to route: {question}

<|eot_id|>

<|start_header_id|>assistant<|end_header_id|>

```

### 4.3 LangGraph를 이용한 워크플로우 구현

LangGraph의 StateGraph를 이용하여 워크플로우를 정의함으로써, LLM 에이전트가 입력 컨텍스트를 통해 직접 판단하여 다른 함수를 수행할 수 있도록 구현하였다. 본 프로젝트에서는 Router 에이전트가 입력 컨텍스트를 통해 웹 검색 함수 또는 답변 생성 함수를 호출하게 된다.

```python
# Build the nodes
self.workflow = StateGraph(self.GraphState)
self.workflow.add_node("websearch", self.web_search)
self.workflow.add_node("transform_query", self.transform_query)
self.workflow.add_node("generate", self.generate)
# Build the edges
self.workflow.set_conditional_entry_point(
    self.route_question,
    {
        "websearch": "transform_query",
        "generate": "generate",
    },
)
self.workflow.add_edge("transform_query", "websearch")
self.workflow.add_edge("websearch", "generate")
self.workflow.add_edge("generate", END)

```

LangGraph에서 LLM 멀티 에이전트 구조를 정의하기 위한 StateGraph를 생성하고, 사용자의 질문이 입력되었을 때 Router 에이전트의 답변을 생성하는 "route_question" 함수를 호출하게 된다. Router 에이전트가 "websearch" 문자열을 반환하면 "transform_query" 함수가 호출되고, "generate" 문자열을 반환하면 "generate" 함수를 호출하게 된다. 웹 검색이 필요하여 "transform_query" 함수가 호출되면 웹 검색 데이터를 기반으로 "generate"가 실행되어 에이전트의 최종 답변이 반환된다.

# 5. 시스템 구현 및 실험 결과

---

### 5.1 Streamlit을 이용한 웹 인터페이스 구현

본 연구에서는 Streamlit 프레임워크를 사용하여 웹 기반 사용자 인터페이스를 구현하였다. Streamlit은 데이터 과학 및 머신러닝 애플리케이션을 위한 빠른 웹 개발을 지원하는 프레임워크로, 데이터 시각화 및 대시보드 기능을 쉽게 구현할 수 있다.

구현된 웹 인터페이스는 사용자가 질문을 입력하면 LLM 에이전트가 웹 검색을 수행하고 답변을 생성하는 과정을 시각적으로 보여준다. 이를 통해 사용자는 시스템의 동작을 직관적으로 이해하고 상호작용할 수 있다.

![11](https://github.com/user-attachments/assets/7928acf8-b9d5-41fc-a340-99cbc76c59e9)


### 5.2 실험 결과

Meta의 LLaMA 3 모델과 DuckDuckGo API를 활용하여 LG CNS사의 최근 업적과 행보에 대한 질문을 시스템에 입력하였다. 실험 결과, LLM 에이전트는 DuckDuckGo API를 통해 LG CNS에 관한 최신 정보를 웹에서 검색하고, 이를 바탕으로 답변을 생성하였다.

![22](https://github.com/user-attachments/assets/f699d402-7dcf-48c7-8cbf-35f49c3c0369)

![11 1](https://github.com/user-attachments/assets/019bdada-d505-4dda-a86c-b694fe2c7e35)


생성된 답변에는 LG CNS가 '구글 클라우드 파트너 어워드 2024 올해의 구글 클라우드 서비스 파트너'로 선정된 것과 인도네시아에서 'LG 시나르마스 테크놀로지'를 설립한 내용 등이 포함되었다. 이는 시스템이 최신 웹 정보를 성공적으로 검색하고 관련성 있는 정보를 추출하여 답변을 생성했음을 보여준다.

### 5.3 다양한 LLM을 활용한 RAG 에이전트 실험

본 연구에서는 Meta사의 LLaMA 3 모델 외에도 고려대학교에서 개발한 SOLOR 기반 LLM 파인튜닝 모델인 "KULLM3"를 이용하여 RAG 시스템의 성능을 비교 실험하였다. 실험에는 PC의 성능을 고려하여 양자화된 "bnksys/kullm3-11b:latest" 모델을 사용하였다.

![11 2](https://github.com/user-attachments/assets/25058881-b89b-4cb3-8ae2-1cf9f5cbdb36)


KULLM3 모델을 사용했을 때, LLaMA 3 모델과는 다른 정보가 검색되었으며, 이는 검색 시점과 모델의 특성에 따라 결과가 달라질 수 있음을 보여준다. 특히, 한국어 데이터셋으로 파인튜닝된 KULLM3 모델을 사용했을 때, LLaMA 3 모델과 달리 한국어로 생성된 답변을 받을 수 있었다. 이는 다국어 지원 및 지역화된 정보 제공에 있어 모델 선택의 중요성을 시사한다.

# 6. 결론 및 향후 연구 방향

---

본 연구에서는 Function Call을 이용한 웹 검색 RAG 에이전트를 구현하고, 그 성능을 실험적으로 검증하였다. 구현된 시스템은 LLM, 웹 검색 API, 그리고 LangGraph를 통합하여 사용자 질의에 대해 최신의 정확한 정보를 제공할 수 있음을 보였다.

연구 결과, 다음과 같은 주요 성과를 얻을 수 있었다:

1. Function Call을 통한 효과적인 에이전트 간 상호작용 구현
2. LangGraph를 이용한 유연한 워크플로우 관리
3. 실시간 웹 검색을 통한 최신 정보 제공 능력 확인
4. 다양한 LLM 모델 적용을 통한 시스템의 확장성 검증

향후 연구에서는 다음과 같은 방향으로 시스템을 개선 및 확장할 수 있을 것이다:

1. 다양한 언어 모델 및 검색 엔진의 통합: 더 많은 LLM과 검색 엔진을 시스템에 통합하여 성능을 비교하고 최적화할 수 있다.
2. 멀티모달 정보 처리: 텍스트뿐만 아니라 이미지, 비디오 등 다양한 형태의 정보를 처리할 수 있도록 시스템을 확장할 수 있다.
3. 개인화 및 컨텍스트 인식: 사용자의 이전 질의와 선호도를 고려한 개인화된 응답 생성 기능을 추가할 수 있다.
4. 설명 가능성 향상: 시스템이 어떤 근거로 특정 정보를 선택하고 답변을 생성했는지에 대한 설명 기능을 강화할 수 있다.
5. 윤리적 고려사항: 편향성 감소와 정보의 신뢰성 검증을 위한 메커니즘을 추가로 구현할 수 있다.

본 연구는 RAG 시스템의 실제 구현과 그 성능을 검증했다는 점에서 의의가 있으며, 향후 더욱 발전된 대화형 AI 시스템 개발을 위한 기초 연구로 활용될 수 있을 것이다.

# 참고 자료

---

[**Function Calling Local LLMs!? LLaMa 3 Web Search Agent Breakdown**]

[https://www.youtube.com/watch?v=9K51Leyv3qI&t=1273s](https://www.youtube.com/watch?v=9K51Leyv3qI&t=1273s)

[**Function Calling with LLMs**]

[https://www.promptingguide.ai/applications/function_calling](https://www.promptingguide.ai/applications/function_calling)

[LangGraph Docs]

[https://langchain-ai.github.io/langgraph/](https://langchain-ai.github.io/langgraph/)

[**국내 기업을 위한 RAG 구조 기반 질의응답시스템 설계**]

[https://www.dbpia.co.kr/journal/articleDetail?nodeId=NODE11858560](https://www.dbpia.co.kr/journal/articleDetail?nodeId=NODE11858560)
