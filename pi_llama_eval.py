import time
import psutil
import os
import csv
import numpy as np
from datasets import load_dataset
from llama_cpp import Llama

# 1. 경로 및 모델 설정 (라즈베리파이 환경)
MODEL_DIR = "/home/pi/win_models"
LLAMA_MODELS = [
    "llama3.2-3b-q5_k_m.gguf",
    "llama3.2-3b-q4_k_m.gguf",
    "llama3.2-3b-q3_k_m.gguf"
]
OUTPUT_FILE = "llama_performance_per_dataset.csv"

def get_memory_usage():
    """현재 프로세스의 RSS 메모리(MB) 반환"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)

def load_bench_prompts(num_samples=20):
    """각 데이터셋별로 프롬프트를 추출하여 딕셔너리로 반환"""
    print("데이터셋 4종 로드 및 프롬프트 추출 중...")
    prompts_dict = {}

    # 1. MMLU
    ds_mmlu = load_dataset("cais/mmlu", "college_computer_science", split="test").select(range(num_samples))
    prompts_dict["MMLU"] = [
        f"Question: {d['question']}\nA. {d['choices'][0]}\nB. {d['choices'][1]}\nC. {d['choices'][2]}\nD. {d['choices'][3]}\nAnswer with only the letter." 
        for d in ds_mmlu
    ]

    # 2. ARC
    ds_arc = load_dataset("ai2_arc", "ARC-Challenge", split="test").select(range(num_samples))
    prompts_dict["ARC"] = [
        f"Question: {d['question']}\n" + "\n".join([f"{l}. {t}" for l, t in zip(d['choices']['label'], d['choices']['text'])]) + "\nAnswer with only the letter/number."
        for d in ds_arc
    ]

    # 3. GSM8K
    ds_gsm = load_dataset("gsm8k", "main", split="test").select(range(num_samples))
    prompts_dict["GSM8K"] = [
        f"Question: {d['question']}\nLet's think step by step. End with 'The answer is [number]'" 
        for d in ds_gsm
    ]

    # 4. HumanEval
    ds_code = load_dataset("openai_humaneval", split="test").select(range(num_samples))
    prompts_dict["HumanEval"] = [
        f"Complete the Python function:\n```python\n{d['prompt']}\n```" 
        for d in ds_code
    ]

    return prompts_dict

# Llama 3 전용 프롬프트 포맷
def format_llama_prompt(prompt):
    return f"<|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"

def run_performance_bench(model_path, datasets_prompts):
    model_name = os.path.basename(model_path)
    print(f"\n========================================")
    print(f"[성능 분석 시작] 모델: {model_name}")
    print(f"========================================")
    
    start_load = time.perf_counter()
    
    # 라즈베리파이 5 맞춤형 설정 (CPU 멀티스레딩)
    llm = Llama(
        model_path=model_path,
        n_threads=4, # RPi5의 4코어 활용
        n_ctx=2048,
        verbose=False
    )
    load_time = time.perf_counter() - start_load
    
    # --- 추론 및 시간 측정 헬퍼 함수 ---
    def _measure_inference(prompt_text, max_tokens=128):
        prompt_formatted = format_llama_prompt(prompt_text)
        prompt_tokens = len(llm.tokenize(prompt_formatted.encode('utf-8')))
        
        start_time = time.perf_counter()
        
        # Llama 3 전용 종료 토큰 적용
        output = llm(
            prompt_formatted,
            max_tokens=max_tokens,
            stop=["<|eot_id|>"], 
            stream=True
        )
        
        first_token_time = None
        token_count = 0
        
        for chunk in output:
            if first_token_time is None:
                first_token_time = time.perf_counter()
            token_count += 1
            
        end_time = time.perf_counter()
        
        ttft = first_token_time - start_time if first_token_time else 0
        decode_time = end_time - first_token_time if first_token_time else 0
        
        prefill_tps = prompt_tokens / ttft if ttft > 0 else 0
        decode_tps = (token_count - 1) / decode_time if decode_time > 0 and token_count > 1 else 0
        
        return {
            "prompt_tokens": prompt_tokens,
            "ttft": ttft,
            "prefill_tps": prefill_tps,
            "decode_tps": decode_tps,
            "peak_mem": get_memory_usage()
        }

    # --- 1. 콜드 스타트 측정 ---
    print("  [워밍업] 메모리 매핑 및 콜드 스타트 지연 측정 중...")
    cold_result = _measure_inference("Hello. Reply with 'Hi' only.", max_tokens=10)
    print(f"  -> ❄️ 콜드 스타트 TTFT: {cold_result['ttft']:.3f}s (Decode: {cold_result['decode_tps']:.2f} TPS)\n")

    dataset_metrics = []

    # --- 2. 데이터셋별 본 테스트 진행 ---
    for dataset_name, prompts in datasets_prompts.items():
        print(f"  [{dataset_name}] 테스트 진행 중 (총 {len(prompts)}개)...")
        results = []
        
        for i, prompt in enumerate(prompts):
            res = _measure_inference(prompt)
            results.append(res)
            print(f"    - {i+1}번 | 프롬프트 토큰: {res['prompt_tokens']} | TTFT: {res['ttft']:.3f}s | Decode: {res['decode_tps']:.2f} TPS") 
            
        all_tokens_str = ", ".join([str(r['prompt_tokens']) for r in results])
        all_ttft_str = ", ".join([f"{r['ttft']:.3f}" for r in results])
        all_tps_str = ", ".join([f"{r['decode_tps']:.2f}" for r in results])
            
        metrics = {
            "Model": model_name,
            "Dataset": dataset_name,
            "Total Prompt Tokens": sum([r['prompt_tokens'] for r in results]),
            "Avg Prompt Tokens": np.mean([r['prompt_tokens'] for r in results]),
            "All Prompt Tokens": all_tokens_str,
            "Cold Start TTFT (s)": cold_result['ttft'] if dataset_name == "MMLU" else "-", # 콜드스타트는 첫 데이터셋에만 기록
            "Avg TTFT (s)": np.mean([r['ttft'] for r in results]),
            "All TTFTs (s)": all_ttft_str,
            "Avg Prefill TPS": np.mean([r['prefill_tps'] for r in results]),
            "Avg Decode TPS": np.mean([r['decode_tps'] for r in results]),
            "All Decode TPS": all_tps_str,
            "Peak Memory (MB)": np.max([r['peak_mem'] for r in results])
        }
        dataset_metrics.append(metrics)
        print(f"  => [{dataset_name}] 총 프롬프트 토큰: {metrics['Total Prompt Tokens']} | 평균 프롬프트 토큰: {metrics['Avg Prompt Tokens']:.1f} | 평균 TTFT: {metrics['Avg TTFT (s)']:.3f}s | 평균 Decode TPS: {metrics['Avg Decode TPS']:.2f}\n")

    return dataset_metrics

if __name__ == "__main__":
    datasets_prompts = load_bench_prompts(num_samples=20)
    all_summary = []

    for model_name in LLAMA_MODELS:
        full_path = os.path.join(MODEL_DIR, model_name)
        if os.path.exists(full_path):
            summary_list = run_performance_bench(full_path, datasets_prompts)
            all_summary.extend(summary_list)
        else:
            print(f"[경고] 파일을 찾을 수 없습니다: {full_path}")

    # CSV 파일로 결과 저장
    if all_summary:
        with open(OUTPUT_FILE, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=all_summary[0].keys())
            writer.writeheader()
            writer.writerows(all_summary)
        print(f"\n✅ 전체 벤치마크 완료! 결과가 '{OUTPUT_FILE}'에 저장되었습니다.")
