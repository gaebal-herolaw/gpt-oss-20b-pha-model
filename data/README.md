# Data Directory

이 폴더에 Personal Health Agent 논문 데이터를 추가하세요.

## 파일 형식

- **지원 형식**: Markdown (.md)
- **인코딩**: UTF-8

## 예시 파일

다음과 같은 형식의 마크다운 파일을 추가하세요:

```
personal-health-agent.md
nature-medicine-ph-llm.md
```

## 데이터 추가 방법

1. 논문을 Markdown 형식으로 변환
2. 이 폴더에 `.md` 파일로 저장
3. `build_index.py`를 실행하여 벡터 DB 구축

## 주의사항

- 파일명은 영문, 숫자, 하이픈(-)만 사용 권장
- 한글 파일명도 가능하지만 영문 권장
- 대용량 파일(10MB 이상)은 여러 파일로 분할 권장
