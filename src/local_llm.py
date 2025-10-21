from openai import OpenAI
from typing import Optional
import os

class LocalGPT:
    """LM Studio 서버를 통해 실행되는 gpt-oss-20b 모델"""

    def __init__(self, model_path: str = None, base_url: str = None):
        """
        Args:
            model_path: 모델 이름 (사용하지 않음, 호환성 유지용)
            base_url: LM Studio 서버 URL (기본: http://localhost:1234/v1)
        """
        self.base_url = base_url or os.getenv("LM_STUDIO_URL", "http://localhost:1234/v1")

        print(f"Connecting to LM Studio server at {self.base_url}")

        # OpenAI 클라이언트 초기화 (LM Studio는 OpenAI API 호환)
        self.client = OpenAI(
            base_url=self.base_url,
            api_key="lm-studio"  # LM Studio는 API 키가 필요 없지만 형식상 필요
        )

        print("[OK] LM Studio client initialized")

    def generate(
        self,
        prompt: str,
        max_length: int = 2048,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50
    ) -> str:
        """텍스트 생성"""

        try:
            # LM Studio API 호출
            completion = self.client.completions.create(
                model="local-model",  # LM Studio에서 로드된 모델
                prompt=prompt,
                max_tokens=max_length,
                temperature=temperature,
                top_p=top_p,
                stream=False
            )

            # 응답 추출
            response = completion.choices[0].text.strip()
            return response

        except Exception as e:
            error_msg = f"LM Studio API error: {str(e)}"
            print(f"Error: {error_msg}")
            return f"[Error] {error_msg}\n\nPlease make sure:\n1. LM Studio is running\n2. Model is loaded\n3. Server is started (default port: 1234)"

    def chat(self, message: str, system_prompt: Optional[str] = None) -> str:
        """채팅 형식으로 생성"""

        try:
            # 메시지 구성
            messages = []
            if system_prompt:
                messages.append({
                    "role": "system",
                    "content": system_prompt
                })
            messages.append({
                "role": "user",
                "content": message
            })

            # Chat API 호출
            completion = self.client.chat.completions.create(
                model="local-model",
                messages=messages,
                temperature=0.7,
                stream=False
            )

            response = completion.choices[0].message.content.strip()
            return response

        except Exception as e:
            error_msg = f"LM Studio API error: {str(e)}"
            print(f"Error: {error_msg}")
            return f"[Error] {error_msg}\n\nPlease make sure:\n1. LM Studio is running\n2. Model is loaded\n3. Server is started (default port: 1234)"

# 사용 예시
if __name__ == "__main__":
    print("\n" + "="*60)
    print("LM Studio Client Test")
    print("="*60)
    print("\nMake sure LM Studio is running with a model loaded!")
    print("="*60 + "\n")

    llm = LocalGPT()

    # 테스트
    print("Testing completion API...")
    response = llm.generate("What is Personal Health Agent? Answer in 2-3 sentences.")
    print(f"\nResponse:\n{response}")

    print("\n" + "="*60)
    print("Test complete!")
    print("="*60)
