"""
LLM Summarizer Module for LSED
Handles summarization of social messages using LLMs via Ollama.

Based on: "Text is All You Need: LLM-enhanced Incremental Social Event Detection"
ACL 2025

The paper uses three LLMs:
- Meta's Llama3.1-8B
- Alibaba's Qwen2.5-7B
- Google's Gemma2-9B

Key prompting strategies:
1. "Summarize" performs better than "Paraphrase" for short texts
2. Add instruction to expand abbreviations
3. Enforce English response for multilingual datasets
"""

import logging
import time
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

logger = logging.getLogger(__name__)


@dataclass
class SummarizationResult:
    """Result of LLM summarization."""
    original_text: str
    summarized_text: str
    model: str
    success: bool
    error_message: Optional[str] = None
    processing_time: float = 0.0


class PromptTemplates:
    """
    Prompt templates for LLM summarization.
    
    Based on the paper's findings:
    - "Summarize" keyword works better than "Paraphrase" for short texts
    - Adding abbreviation expansion instruction improves results
    - Enforcing English response handles multilingual data
    """
    
    # Main summarize prompt (recommended by paper)
    SUMMARIZE = (
        "Summarize the following sentences: {text}. "
        "If you come across some abbreviations, expand them. "
        "Please respond in English."
    )
    
    # Alternative paraphrase prompt
    PARAPHRASE = (
        "Paraphrase the following sentences: {text}. "
        "If you come across some abbreviations, expand them. "
        "Please respond in English."
    )
    
    # Summarize without abbreviation expansion
    SUMMARIZE_SIMPLE = "Summarize the following sentences: {text}."
    
    # Summarize without language restriction
    SUMMARIZE_NO_LANG = (
        "Summarize the following sentences: {text}. "
        "If you come across some abbreviations, expand them."
    )
    
    @classmethod
    def get_prompt(cls, prompt_type: str) -> str:
        """Get prompt template by type."""
        prompts = {
            'summarize': cls.SUMMARIZE,
            'paraphrase': cls.PARAPHRASE,
            'summarize_simple': cls.SUMMARIZE_SIMPLE,
            'summarize_no_lang': cls.SUMMARIZE_NO_LANG
        }
        return prompts.get(prompt_type.lower(), cls.SUMMARIZE)


class OllamaClient:
    """Client for interacting with Ollama API."""
    
    def __init__(
        self, 
        host: str = "http://localhost:11434",
        timeout: int = 60
    ):
        """
        Initialize Ollama client.
        
        Args:
            host: Ollama API host URL
            timeout: Request timeout in seconds
        """
        self.host = host.rstrip('/')
        self.timeout = timeout
        self.generate_url = f"{self.host}/api/generate"
        
    def generate(
        self,
        model: str,
        prompt: str,
        temperature: float = 0.1,
        max_tokens: int = 256,
        **kwargs
    ) -> str:
        """
        Generate text using Ollama.
        
        Args:
            model: Model name
            prompt: Input prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            
        Returns:
            Generated text
        """
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens
            }
        }
        payload.update(kwargs)
        
        response = requests.post(
            self.generate_url,
            json=payload,
            timeout=self.timeout
        )
        response.raise_for_status()
        
        return response.json().get('response', '').strip()
    
    def is_available(self) -> bool:
        """Check if Ollama server is available."""
        try:
            response = requests.get(f"{self.host}/api/tags", timeout=5)
            return response.status_code == 200
        except Exception:
            return False
    
    def list_models(self) -> List[str]:
        """List available models."""
        try:
            response = requests.get(f"{self.host}/api/tags", timeout=10)
            response.raise_for_status()
            models = response.json().get('models', [])
            return [m['name'] for m in models]
        except Exception as e:
            logger.error(f"Failed to list models: {e}")
            return []


class MockLLMClient:
    """
    Mock LLM client for testing without actual LLM.
    Simply returns the original text with a prefix.
    """
    
    def generate(self, model: str, prompt: str, **kwargs) -> str:
        """Mock generation - extract and return enhanced text."""
        # Extract original text from prompt
        if "sentences:" in prompt.lower():
            text = prompt.split("sentences:")[-1].split(".")[0].strip()
            # Simple mock enhancement
            return f"[Summarized] {text}"
        return prompt
    
    def is_available(self) -> bool:
        return True
    
    def list_models(self) -> List[str]:
        return ["mock-model"]


class LLMSummarizer:
    """
    LLM-based summarizer for social messages.
    
    Uses LLMs to:
    1. Expand abbreviations
    2. Standardize informal expressions
    3. Add context from LLM's background knowledge
    """
    
    # Model name mappings
    MODEL_NAMES = {
        'llama3.1': 'llama3.1:8b',
        'qwen2.5': 'qwen2.5:7b',
        'gemma2': 'gemma2:9b'
    }
    
    def __init__(self, config: Dict[str, Any], use_mock: bool = False):
        """
        Initialize the LLM summarizer.
        
        Args:
            config: Configuration dictionary
            use_mock: Whether to use mock LLM client for testing
        """
        self.config = config
        self.llm_config = config.get('llm', {})
        self.prompt_config = config.get('prompts', {})
        
        # LLM settings
        self.model_key = self.llm_config.get('model', 'llama3.1')
        self.model_name = self.llm_config.get('model_names', self.MODEL_NAMES).get(
            self.model_key, 
            self.MODEL_NAMES.get(self.model_key, 'llama3.1:8b')
        )
        self.temperature = self.llm_config.get('temperature', 0.1)
        self.max_tokens = self.llm_config.get('max_tokens', 256)
        self.batch_size = self.llm_config.get('batch_size', 32)
        
        # Prompt settings
        self.prompt_type = self.prompt_config.get('prompt_type', 'summarize')
        self.prompt_template = self.prompt_config.get(
            self.prompt_type,
            PromptTemplates.get_prompt(self.prompt_type)
        )
        
        # Initialize client
        if use_mock:
            self.client = MockLLMClient()
            logger.info("Using mock LLM client")
        else:
            ollama_host = self.llm_config.get('ollama_host', 'http://localhost:11434')
            self.client = OllamaClient(host=ollama_host)
            
            if not self.client.is_available():
                logger.warning("Ollama server not available. Using mock client.")
                self.client = MockLLMClient()
        
        logger.info(f"LLM Summarizer initialized with model: {self.model_name}")
        logger.info(f"Prompt type: {self.prompt_type}")
    
    def create_prompt(self, text: str) -> str:
        """
        Create prompt from template.
        
        Args:
            text: Input text to summarize
            
        Returns:
            Formatted prompt
        """
        return self.prompt_template.format(text=text)
    
    def summarize_single(self, text: str) -> SummarizationResult:
        """
        Summarize a single text.
        
        Args:
            text: Input text
            
        Returns:
            SummarizationResult
        """
        start_time = time.time()
        
        try:
            prompt = self.create_prompt(text)
            
            response = self.client.generate(
                model=self.model_name,
                prompt=prompt,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            processing_time = time.time() - start_time
            
            return SummarizationResult(
                original_text=text,
                summarized_text=response,
                model=self.model_name,
                success=True,
                processing_time=processing_time
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Summarization failed: {e}")
            
            return SummarizationResult(
                original_text=text,
                summarized_text=text,  # Fallback to original
                model=self.model_name,
                success=False,
                error_message=str(e),
                processing_time=processing_time
            )
    
    def summarize_batch(
        self, 
        texts: List[str],
        num_workers: int = 4,
        show_progress: bool = True
    ) -> List[SummarizationResult]:
        """
        Summarize a batch of texts.
        
        Args:
            texts: List of input texts
            num_workers: Number of parallel workers
            show_progress: Whether to show progress bar
            
        Returns:
            List of SummarizationResult
        """
        results = []
        
        # For sequential processing (safer with LLM)
        if num_workers <= 1:
            iterator = tqdm(texts, desc="Summarizing") if show_progress else texts
            for text in iterator:
                result = self.summarize_single(text)
                results.append(result)
        else:
            # Parallel processing
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                future_to_text = {
                    executor.submit(self.summarize_single, text): i 
                    for i, text in enumerate(texts)
                }
                
                # Initialize results list with None
                results = [None] * len(texts)
                
                # Collect results
                iterator = as_completed(future_to_text)
                if show_progress:
                    iterator = tqdm(iterator, total=len(texts), desc="Summarizing")
                
                for future in iterator:
                    idx = future_to_text[future]
                    results[idx] = future.result()
        
        # Log statistics
        successful = sum(1 for r in results if r.success)
        total_time = sum(r.processing_time for r in results)
        
        logger.info(f"Summarization complete: {successful}/{len(results)} successful")
        logger.info(f"Total processing time: {total_time:.2f}s")
        
        return results
    
    def get_summarized_texts(
        self, 
        texts: List[str],
        **kwargs
    ) -> List[str]:
        """
        Get summarized texts (convenience method).
        
        Args:
            texts: List of input texts
            
        Returns:
            List of summarized texts
        """
        results = self.summarize_batch(texts, **kwargs)
        return [r.summarized_text for r in results]


class SummarizationCache:
    """
    Simple cache for summarization results to avoid redundant LLM calls.
    """
    
    def __init__(self, cache_file: Optional[str] = None):
        """
        Initialize cache.
        
        Args:
            cache_file: Path to cache file for persistence
        """
        self.cache: Dict[str, str] = {}
        self.cache_file = cache_file
        
        if cache_file:
            self._load_cache()
    
    def _load_cache(self):
        """Load cache from file."""
        import json
        from pathlib import Path
        
        if Path(self.cache_file).exists():
            try:
                with open(self.cache_file, 'r') as f:
                    self.cache = json.load(f)
                logger.info(f"Loaded {len(self.cache)} cached summaries")
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")
    
    def _save_cache(self):
        """Save cache to file."""
        import json
        from pathlib import Path
        
        if self.cache_file:
            Path(self.cache_file).parent.mkdir(parents=True, exist_ok=True)
            with open(self.cache_file, 'w') as f:
                json.dump(self.cache, f)
    
    def get(self, text: str) -> Optional[str]:
        """Get cached summary."""
        return self.cache.get(text)
    
    def set(self, text: str, summary: str):
        """Set cached summary."""
        self.cache[text] = summary
    
    def save(self):
        """Save cache to file."""
        self._save_cache()


if __name__ == "__main__":
    # Test the LLM summarizer
    import yaml
    
    # Load config
    with open("config/config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    # Test with mock client
    summarizer = LLMSummarizer(config, use_mock=True)
    
    # Test single summarization
    test_texts = [
        "@user123: OMG did u see the NFL game last night? TB won!",
        "Breaking: POTUS announces new policy on climate change #politics",
        "Just landed in NYC after a long flight from LAX"
    ]
    
    print("\nTesting LLM Summarizer:")
    print("="*60)
    
    for text in test_texts:
        result = summarizer.summarize_single(text)
        print(f"\nOriginal: {result.original_text}")
        print(f"Summarized: {result.summarized_text}")
        print(f"Success: {result.success}")
        print(f"Time: {result.processing_time:.3f}s")
    
    print("\n" + "="*60)
    print("Testing batch summarization:")
    
    results = summarizer.summarize_batch(test_texts, show_progress=True)
    
    for result in results:
        print(f"\n{result.original_text[:50]}...")
        print(f"  -> {result.summarized_text[:50]}...")