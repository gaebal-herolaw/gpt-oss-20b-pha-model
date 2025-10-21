import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Optional
import os

class LocalGPT:
    """RunPod에서 직접 실행되는 gpt-oss-20b 모델"""

    def __init__(self, model_path: str = None):
        # 환경 변수 또는 기본값 사용
        self.model_path = model_path or os.getenv("LLM_MODEL", "gpt-oss-20b")

        # GPU 자동 감지
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"Loading LLM: {self.model_path}")
        print(f"Device: {self.device}")

        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"VRAM Available: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

        print("This may take a few minutes...")

        # 토크나이저 로드
        print("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            use_fast=True
        )

        # 패딩 토큰 설정
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # 모델 로드 설정
        print("Loading model...")
        load_kwargs = {
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
        }

        # GPU 사용 가능 시
        if torch.cuda.is_available():
            # GPU 메모리에 따라 설정 조정
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9

            if gpu_memory >= 40:  # A100 40GB 이상
                load_kwargs["torch_dtype"] = torch.float16
                load_kwargs["device_map"] = "auto"
            elif gpu_memory >= 24:  # RTX 3090/4090
                load_kwargs["torch_dtype"] = torch.float16
                load_kwargs["device_map"] = "auto"
                load_kwargs["max_memory"] = {0: "20GB"}
            else:  # 작은 GPU
                load_kwargs["torch_dtype"] = torch.float16
                load_kwargs["device_map"] = "auto"
                load_kwargs["load_in_8bit"] = True  # 양자화
        else:
            # CPU 모드
            load_kwargs["torch_dtype"] = torch.float32

        # 모델 로드
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            **load_kwargs
        )

        print(f"[OK] Model loaded on {self.device}")

        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated(0) / 1e9
            print(f"Model VRAM: {memory_allocated:.2f} GB")

    def generate(
        self,
        prompt: str,
        max_length: int = 2048,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50
    ) -> str:
        """텍스트 생성"""

        # 토큰화
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=8192,  # 컨텍스트 길이
            padding=True
        )

        # GPU로 이동
        if torch.cuda.is_available():
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # 생성
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_length,  # max_length 대신 max_new_tokens 사용
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=True  # 속도 향상
            )

        # 디코딩
        generated_text = self.tokenizer.decode(
            outputs[0],
            skip_special_tokens=True
        )

        # 프롬프트 제거하고 답변만 반환
        if generated_text.startswith(prompt):
            response = generated_text[len(prompt):].strip()
        else:
            response = generated_text.strip()

        return response

    def chat(self, message: str, system_prompt: Optional[str] = None) -> str:
        """채팅 형식으로 생성"""

        if system_prompt:
            prompt = f"{system_prompt}\n\nUser: {message}\n\nAssistant:"
        else:
            prompt = f"User: {message}\n\nAssistant:"

        return self.generate(prompt)

    def clear_cache(self):
        """GPU 캐시 정리"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("[OK] GPU cache cleared")

# 사용 예시
if __name__ == "__main__":
    print("\n" + "="*60)
    print("RunPod GPU Model Test")
    print("="*60)

    # 모델 로드
    llm = LocalGPT()

    # 테스트
    print("\nTesting generation...")
    response = llm.chat("What is Personal Health Agent? Answer in 2-3 sentences.")
    print(f"\nResponse:\n{response}")

    # 메모리 정리
    llm.clear_cache()

    print("\n" + "="*60)
    print("Test complete!")
    print("="*60)
