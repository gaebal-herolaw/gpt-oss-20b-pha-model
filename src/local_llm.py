import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Optional
from .config import Config

class LocalGPT:
    """로컬에서 실행되는 gpt-oss-20b 모델"""
    
    def __init__(self, model_path: str = None):
        self.model_path = model_path or Config.LLM_MODEL
        self.device = Config.DEVICE
        
        print(f"Loading LLM: {self.model_path}")
        print("This may take a few minutes...")
        
        # 토크나이저 로드
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True
        )
        
        # 모델 로드 (FP16으로 메모리 절약)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=Config.MODEL_DTYPE,
            device_map="auto",  # 자동 GPU 할당
            trust_remote_code=True
        )
        
        print(f"✓ Model loaded on {self.device}")
        
        if torch.cuda.is_available():
            print(f"Model VRAM: {self.model.get_memory_footprint() / 1e9:.2f} GB")
    
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
            max_length=Config.MAX_CONTEXT_LENGTH
        ).to(self.device)
        
        # 생성
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # 디코딩
        generated_text = self.tokenizer.decode(
            outputs[0],
            skip_special_tokens=True
        )
        
        # 프롬프트 제거하고 답변만 반환
        response = generated_text[len(prompt):].strip()
        
        return response
    
    def chat(self, message: str, system_prompt: Optional[str] = None) -> str:
        """채팅 형식으로 생성"""
        
        if system_prompt:
            prompt = f"{system_prompt}\n\nUser: {message}\n\nAssistant:"
        else:
            prompt = f"User: {message}\n\nAssistant:"
        
        return self.generate(prompt)

# 사용 예시
if __name__ == "__main__":
    llm = LocalGPT()
    
    # 테스트
    response = llm.chat("안녕하세요. Personal Health Agent에 대해 설명해주세요.")
    print(f"Response: {response}")
