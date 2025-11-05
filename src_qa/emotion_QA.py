import sys
import os
import json
import numpy as np
import torch
import re
import glob
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

# ---------- 配置 ----------
model_config = {
    "model_name": "/data/zhangjingwei/LL-Doctor-qwen3-8b-Model",
    "embedding_model": "BAAI/bge-m3",
    "chunk_size": 512,
    "retrieved_count": 3,
    "max_length": 2048,
    "temperature": 0.7,
    "top_p": 0.9,
    "max_new_tokens": 200,
}

# ---------- 简化的检索器 ----------
class SimpleRetriever:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def chunk_text(self, text, chunk_size=512):
        if not text:
            return []
            
        sentences = re.split(r'[.!?。！？]', text)
        chunks = []
        current_chunk = ""
        
        for sent in sentences:
            sent = sent.strip()
            if not sent:
                continue
                
            if len(current_chunk.split()) + len(sent.split()) <= chunk_size:
                if current_chunk:
                    current_chunk += ". " + sent
                else:
                    current_chunk = sent
            else:
                if current_chunk:
                    chunks.append(current_chunk + ".")
                    current_chunk = sent
                    
        if current_chunk:
            chunks.append(current_chunk + ".")
            
        return chunks[:10]

    def retrieve_from_context(self, question, context, k=3):
        chunks = self.chunk_text(context)
        if not chunks:
            words = context.split()
            chunk_size = len(words) // k
            chunks = [' '.join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]
            chunks = chunks[:k]
            
        return chunks[:k]

# ---------- 最终版RAG系统 ----------
class EmotionQARAGFinal:
    def __init__(self, model_path):
        self.retriever = SimpleRetriever()
        self._load_model(model_path)
        
        self.stop_tokens = ["\n\n", "###", "Question:", "Answer:"]

    def _load_model(self, model_path):
        print(f"Loading model from {model_path}...")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path, 
                trust_remote_code=True, 
                local_files_only=True
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
            print("✓ Tokenizer loaded")
        except Exception as e:
            print(f"Tokenizer loading error: {e}")
            raise e
            
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
                local_files_only=True,
            )
            self.model.eval()
            print("✓ Model loaded successfully")
        except Exception as e:
            print(f"Model loading error: {e}")
            try:
                from transformers import Qwen2ForCausalLM
                self.model = Qwen2ForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    trust_remote_code=True,
                    local_files_only=True,
                )
                self.model.eval()
                print("✓ Model loaded successfully with Qwen2ForCausalLM")
            except Exception as e2:
                print(f"Qwen2ForCausalLM also failed: {e2}")
                raise e2

    def _build_prompt(self, problem, retrieved_chunks):
        # 加强英文回答的要求
        if retrieved_chunks:
            text = "\n".join(retrieved_chunks[:2])
            prompt = f"""Answer the following question in English based on the provided context.

Context: {text}

Question: {problem}

Answer in English:"""
        else:
            prompt = f"""Answer the following question in English.

Question: {problem}

Answer in English:"""
        return prompt

    def _generate_answer(self, prompt):
        try:
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=model_config["max_length"],
            )
            
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=model_config["max_new_tokens"],
                    do_sample=True,
                    temperature=model_config["temperature"],
                    top_p=model_config["top_p"],
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            
            full_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            answer = full_text[len(prompt):].strip()
            
            # 清理答案
            for stop in self.stop_tokens:
                if stop in answer:
                    answer = answer.split(stop)[0].strip()
            
            # 确保答案是英文 - 移除中文字符
            answer = self._ensure_english(answer)
                    
            return answer
            
        except Exception as e:
            print(f"Generation error: {e}")
            return "[Error generating answer]"

    def _ensure_english(self, text):
        """确保文本是英文，移除中文字符"""
        # 移除中文字符
        text = re.sub(r'[\u4e00-\u9fff]+', '', text)
        # 移除其他非英文字符（保留英文、数字、标点）
        text = re.sub(r'[^\x00-\x7F]+', ' ', text)
        return text.strip()

    def process_question(self, question_id, problem, context):
        retrieved = self.retriever.retrieve_from_context(
            problem, context, k=model_config["retrieved_count"]
        )
        
        prompt = self._build_prompt(problem, retrieved)
        answer = self._generate_answer(prompt)
        
        return {"id": question_id, "predicted_answer": answer}

# ---------- 主函数 ----------
def main():
    TEST_FILE = "/data/zhangjingwei/data/LongEmotion/Emotion QA.jsonl"
    OUTPUT_FILE = "results/emotion_qa_final_results.jsonl"
    
    if not os.path.exists(TEST_FILE):
        print(f"Error: Test file {TEST_FILE} does not exist!")
        return
        
    if not os.path.exists(model_config["model_name"]):
        print(f"Error: Model path {model_config['model_name']} does not exist!")
        return

    # 加载测试数据
    print("Loading test data...")
    with open(TEST_FILE, 'r', encoding='utf-8') as f:
        questions = [json.loads(line) for line in f]
    
    print(f"Loaded {len(questions)} questions")
    
    # 初始化RAG系统
    print("Initializing RAG system...")
    try:
        rag = EmotionQARAGFinal(model_config["model_name"])
    except Exception as e:
        print(f"Failed to initialize RAG system: {e}")
        return
    
    # 处理问题
    print("\nStarting processing...")
    results = []
    empty_count = 0
    
    for i, q in enumerate(tqdm(questions, desc="Processing")):
        problem = q.get("problem", "")
        context = q.get("context", "")
        
        result = rag.process_question(i, problem, context)
        results.append(result)
        
        # 检查答案
        if not result["predicted_answer"].strip() or result["predicted_answer"] == "[Error generating answer]":
            empty_count += 1
        
        # 每50条保存一次
        if (i + 1) % 50 == 0:
            with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
                for r in results:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            print(f"Progress: {i+1}/{len(questions)}, empty answers: {empty_count}")
    
    # 最终保存
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # 分析结果
    print(f"\n{'='*60}")
    print("FINAL RESULTS")
    print(f"{'='*60}")
    
    empty_answers = [r for r in results if not r["predicted_answer"].strip() or 
                    r["predicted_answer"] == "[Error generating answer]"]
    
    valid_answers = [r for r in results if r["predicted_answer"].strip() and 
                    r["predicted_answer"] != "[Error generating answer]"]
    
    print(f"Total questions: {len(results)}")
    print(f"Valid answers: {len(valid_answers)}")
    print(f"Empty/error answers: {len(empty_answers)}")
    print(f"Success rate: {len(valid_answers)/len(results)*100:.1f}%")
    
    # 检查是否有中文答案
    chinese_answers = []
    for r in valid_answers:
        if re.search(r'[\u4e00-\u9fff]', r["predicted_answer"]):
            chinese_answers.append(r)
    
    if chinese_answers:
        print(f"Chinese answers found: {len(chinese_answers)}")
        for i, r in enumerate(chinese_answers[:3]):
            print(f"  {i+1}. Q{r['id']}: {r['predicted_answer'][:100]}...")
    else:
        print("All answers are in English!")

if __name__ == "__main__":
    main()
