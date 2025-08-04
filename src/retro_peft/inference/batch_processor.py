"""
Batch processing utilities for efficient inference.

Handles large-scale batch processing with memory management,
progress tracking, and error handling.
"""

import time
import json
import os
from typing import Dict, List, Any, Optional, Iterator, Callable, Union
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import torch
from tqdm import tqdm

from .pipeline import RetrievalInferencePipeline


@dataclass
class BatchRequest:
    """Single batch request item"""
    id: str
    prompt: str
    retrieval_k: int = 5
    retrieval_enabled: bool = True
    max_length: Optional[int] = None
    temperature: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class BatchResult:
    """Batch processing result"""
    id: str
    prompt: str
    generated_text: str
    success: bool
    error: Optional[str] = None
    generation_time: float = 0.0
    retrieval_info: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None


class BatchProcessor:
    """
    Efficient batch processor for retrieval-augmented inference.
    
    Features:
    - Memory-efficient batch processing
    - Progress tracking and logging
    - Error handling and recovery
    - Results streaming and caching
    - Parallel processing support
    """
    
    def __init__(
        self,
        pipeline: RetrievalInferencePipeline,
        batch_size: int = 16,
        max_workers: int = 1,
        output_dir: Optional[str] = None,
        enable_progress_bar: bool = True,
        save_intermediate: bool = True,
        error_handling: str = "continue"  # "continue", "stop", "retry"
    ):
        """
        Initialize batch processor.
        
        Args:
            pipeline: Inference pipeline to use
            batch_size: Size of processing batches
            max_workers: Maximum number of parallel workers
            output_dir: Directory to save results
            enable_progress_bar: Whether to show progress bar
            save_intermediate: Whether to save intermediate results
            error_handling: How to handle errors ("continue", "stop", "retry")
        """
        self.pipeline = pipeline
        self.batch_size = batch_size
        self.max_workers = max_workers
        self.output_dir = output_dir
        self.enable_progress_bar = enable_progress_bar
        self.save_intermediate = save_intermediate
        self.error_handling = error_handling
        
        # Create output directory
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # Processing statistics
        self.stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "total_processing_time": 0.0,
            "average_processing_time": 0.0,
            "start_time": None,
            "end_time": None
        }
    
    def process_batch(
        self,
        requests: List[Union[BatchRequest, Dict[str, Any]]],
        return_retrieval_info: bool = False,
        resume_from: Optional[str] = None
    ) -> List[BatchResult]:
        """
        Process a batch of requests.
        
        Args:
            requests: List of batch requests
            return_retrieval_info: Whether to include retrieval information
            resume_from: Optional checkpoint file to resume from
            
        Returns:
            List of batch results
        """
        # Convert dict requests to BatchRequest objects
        batch_requests = []
        for req in requests:
            if isinstance(req, dict):
                batch_requests.append(BatchRequest(**req))
            else:
                batch_requests.append(req)
        
        self.stats["total_requests"] = len(batch_requests)
        self.stats["start_time"] = time.time()
        
        # Resume from checkpoint if specified
        completed_ids = set()
        results = []
        
        if resume_from and os.path.exists(resume_from):
            print(f"Resuming from checkpoint: {resume_from}")
            with open(resume_from, 'r') as f:
                checkpoint_results = [BatchResult(**item) for item in json.load(f)]
            
            results.extend(checkpoint_results)
            completed_ids = {result.id for result in checkpoint_results}
            self.stats["successful_requests"] = len(checkpoint_results)
        
        # Filter out already completed requests
        remaining_requests = [
            req for req in batch_requests 
            if req.id not in completed_ids
        ]
        
        if not remaining_requests:
            print("All requests already completed")
            return results
        
        print(f"Processing {len(remaining_requests)} requests "
              f"(batch_size={self.batch_size}, workers={self.max_workers})")
        
        # Process in batches
        if self.max_workers > 1:
            # Parallel processing
            results.extend(self._process_parallel(
                remaining_requests, return_retrieval_info
            ))
        else:
            # Sequential processing
            results.extend(self._process_sequential(
                remaining_requests, return_retrieval_info
            ))
        
        self.stats["end_time"] = time.time()
        self.stats["total_processing_time"] = (
            self.stats["end_time"] - self.stats["start_time"]
        )
        
        if self.stats["successful_requests"] > 0:
            self.stats["average_processing_time"] = (
                self.stats["total_processing_time"] / self.stats["successful_requests"]
            )
        
        # Save final results
        if self.output_dir:
            self._save_results(results, "final_results.json")
        
        self._print_summary()
        return results
    
    def _process_sequential(
        self, 
        requests: List[BatchRequest],
        return_retrieval_info: bool
    ) -> List[BatchResult]:
        """Process requests sequentially"""
        results = []
        
        # Create progress bar
        if self.enable_progress_bar:
            pbar = tqdm(total=len(requests), desc="Processing")
        
        for i in range(0, len(requests), self.batch_size):
            batch = requests[i:i + self.batch_size]
            batch_results = self._process_single_batch(batch, return_retrieval_info)
            results.extend(batch_results)
            
            # Update progress
            if self.enable_progress_bar:
                pbar.update(len(batch))
            
            # Save intermediate results
            if self.save_intermediate and self.output_dir:
                checkpoint_path = os.path.join(self.output_dir, f"checkpoint_{i}.json")
                self._save_results(results, checkpoint_path)
        
        if self.enable_progress_bar:
            pbar.close()
        
        return results
    
    def _process_parallel(
        self,
        requests: List[BatchRequest],
        return_retrieval_info: bool
    ) -> List[BatchResult]:
        """Process requests in parallel"""
        results = []
        
        # Split into batches
        batches = [
            requests[i:i + self.batch_size]
            for i in range(0, len(requests), self.batch_size)
        ]
        
        # Process batches in parallel
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all batches
            future_to_batch = {
                executor.submit(
                    self._process_single_batch, 
                    batch, 
                    return_retrieval_info
                ): batch_idx
                for batch_idx, batch in enumerate(batches)
            }
            
            # Create progress bar
            if self.enable_progress_bar:
                pbar = tqdm(total=len(requests), desc="Processing (parallel)")
            
            # Collect results as they complete
            batch_results = {}
            for future in as_completed(future_to_batch):
                batch_idx = future_to_batch[future]
                
                try:
                    batch_result = future.result()
                    batch_results[batch_idx] = batch_result
                    
                    # Update progress
                    if self.enable_progress_bar:
                        pbar.update(len(batch_result))
                    
                except Exception as e:
                    print(f"Batch {batch_idx} failed: {e}")
                    if self.error_handling == "stop":
                        raise
            
            if self.enable_progress_bar:
                pbar.close()
        
        # Combine results in original order
        for batch_idx in sorted(batch_results.keys()):
            results.extend(batch_results[batch_idx])
        
        return results
    
    def _process_single_batch(
        self,
        batch: List[BatchRequest],
        return_retrieval_info: bool
    ) -> List[BatchResult]:
        """Process a single batch of requests"""
        results = []
        
        for request in batch:
            start_time = time.time()
            
            try:
                # Generate response
                response = self.pipeline.generate(
                    prompt=request.prompt,
                    retrieval_k=request.retrieval_k,
                    retrieval_enabled=request.retrieval_enabled,
                    max_length=request.max_length,
                    temperature=request.temperature,
                    return_retrieval_info=return_retrieval_info
                )
                
                processing_time = time.time() - start_time
                
                # Create result
                result = BatchResult(
                    id=request.id,
                    prompt=request.prompt,
                    generated_text=response["generated_text"],
                    success=True,
                    generation_time=processing_time,
                    retrieval_info=response.get("retrieval_info") if return_retrieval_info else None,
                    metadata=request.metadata
                )
                
                self.stats["successful_requests"] += 1
                
            except Exception as e:
                processing_time = time.time() - start_time
                
                # Create error result
                result = BatchResult(
                    id=request.id,
                    prompt=request.prompt,
                    generated_text="",
                    success=False,
                    error=str(e),
                    generation_time=processing_time,
                    metadata=request.metadata
                )
                
                self.stats["failed_requests"] += 1
                
                # Handle error based on policy
                if self.error_handling == "stop":
                    raise
                elif self.error_handling == "retry":
                    # Simple retry logic (could be enhanced)
                    try:
                        retry_response = self.pipeline.generate(
                            prompt=request.prompt,
                            retrieval_k=request.retrieval_k,
                            retrieval_enabled=False,  # Retry without retrieval
                            max_length=request.max_length,
                            temperature=request.temperature
                        )
                        
                        result.generated_text = retry_response["generated_text"]
                        result.success = True
                        result.error = None
                        self.stats["successful_requests"] += 1
                        self.stats["failed_requests"] -= 1
                        
                    except Exception as retry_error:
                        result.error = f"Original: {e}, Retry: {retry_error}"
            
            results.append(result)
        
        return results
    
    def _save_results(self, results: List[BatchResult], filename: str):
        """Save results to file"""
        if not self.output_dir:
            return
        
        filepath = os.path.join(self.output_dir, filename)
        
        # Convert to serializable format
        serializable_results = []
        for result in results:
            result_dict = {
                "id": result.id,
                "prompt": result.prompt,
                "generated_text": result.generated_text,
                "success": result.success,
                "error": result.error,
                "generation_time": result.generation_time,
                "retrieval_info": result.retrieval_info,
                "metadata": result.metadata
            }
            serializable_results.append(result_dict)
        
        with open(filepath, 'w') as f:
            json.dump(serializable_results, f, indent=2)
    
    def _print_summary(self):
        """Print processing summary"""
        print("\n" + "="*50)
        print("BATCH PROCESSING SUMMARY")
        print("="*50)
        print(f"Total requests: {self.stats['total_requests']}")
        print(f"Successful: {self.stats['successful_requests']}")
        print(f"Failed: {self.stats['failed_requests']}")
        
        if self.stats['total_requests'] > 0:
            success_rate = (
                self.stats['successful_requests'] / self.stats['total_requests']
            ) * 100
            print(f"Success rate: {success_rate:.1f}%")
        
        print(f"Total time: {self.stats['total_processing_time']:.2f}s")
        
        if self.stats['successful_requests'] > 0:
            print(f"Average time per request: {self.stats['average_processing_time']:.3f}s")
            throughput = self.stats['successful_requests'] / self.stats['total_processing_time']
            print(f"Throughput: {throughput:.2f} requests/second")
        
        print("="*50)
    
    def process_file(
        self,
        input_file: str,
        output_file: Optional[str] = None,
        input_format: str = "jsonl",  # "jsonl", "json", "csv"
        prompt_field: str = "prompt",
        id_field: str = "id",
        **kwargs
    ) -> List[BatchResult]:
        """
        Process requests from file.
        
        Args:
            input_file: Path to input file
            output_file: Optional output file path
            input_format: Format of input file
            prompt_field: Field name for prompts
            id_field: Field name for IDs
            **kwargs: Additional arguments for process_batch
            
        Returns:
            List of batch results
        """
        # Load requests from file
        requests = self._load_requests_from_file(
            input_file, input_format, prompt_field, id_field
        )
        
        print(f"Loaded {len(requests)} requests from {input_file}")
        
        # Process requests
        results = self.process_batch(requests, **kwargs)
        
        # Save to output file if specified
        if output_file:
            output_format = output_file.split('.')[-1]
            self._save_results_to_file(results, output_file, output_format)
            print(f"Results saved to {output_file}")
        
        return results
    
    def _load_requests_from_file(
        self,
        input_file: str,
        input_format: str,
        prompt_field: str,
        id_field: str
    ) -> List[BatchRequest]:
        """Load requests from input file"""
        requests = []
        
        if input_format == "jsonl":
            with open(input_file, 'r') as f:
                for line_num, line in enumerate(f):
                    try:
                        data = json.loads(line.strip())
                        request = BatchRequest(
                            id=data.get(id_field, f"req_{line_num}"),
                            prompt=data.get(prompt_field, ""),
                            retrieval_k=data.get("retrieval_k", 5),
                            retrieval_enabled=data.get("retrieval_enabled", True),
                            max_length=data.get("max_length"),
                            temperature=data.get("temperature"),
                            metadata=data.get("metadata")
                        )
                        requests.append(request)
                    except json.JSONDecodeError as e:
                        print(f"Error parsing line {line_num}: {e}")
        
        elif input_format == "json":
            with open(input_file, 'r') as f:
                data_list = json.load(f)
                
                for i, data in enumerate(data_list):
                    request = BatchRequest(
                        id=data.get(id_field, f"req_{i}"),
                        prompt=data.get(prompt_field, ""),
                        retrieval_k=data.get("retrieval_k", 5),
                        retrieval_enabled=data.get("retrieval_enabled", True),
                        max_length=data.get("max_length"),
                        temperature=data.get("temperature"),
                        metadata=data.get("metadata")
                    )
                    requests.append(request)
        
        elif input_format == "csv":
            import csv
            with open(input_file, 'r') as f:
                reader = csv.DictReader(f)
                for i, row in enumerate(reader):
                    request = BatchRequest(
                        id=row.get(id_field, f"req_{i}"),
                        prompt=row.get(prompt_field, ""),
                        retrieval_k=int(row.get("retrieval_k", 5)),
                        retrieval_enabled=row.get("retrieval_enabled", "true").lower() == "true",
                        max_length=int(row["max_length"]) if row.get("max_length") else None,
                        temperature=float(row["temperature"]) if row.get("temperature") else None
                    )
                    requests.append(request)
        
        else:
            raise ValueError(f"Unsupported input format: {input_format}")
        
        return requests
    
    def _save_results_to_file(
        self,
        results: List[BatchResult],
        output_file: str,
        output_format: str
    ):
        """Save results to output file"""
        if output_format == "json":
            with open(output_file, 'w') as f:
                json.dump([
                    {
                        "id": r.id,
                        "prompt": r.prompt,
                        "generated_text": r.generated_text,
                        "success": r.success,
                        "error": r.error,
                        "generation_time": r.generation_time,
                        "retrieval_info": r.retrieval_info,
                        "metadata": r.metadata
                    }
                    for r in results
                ], f, indent=2)
        
        elif output_format == "jsonl":
            with open(output_file, 'w') as f:
                for r in results:
                    result_dict = {
                        "id": r.id,
                        "prompt": r.prompt,
                        "generated_text": r.generated_text,
                        "success": r.success,
                        "error": r.error,
                        "generation_time": r.generation_time,
                        "retrieval_info": r.retrieval_info,
                        "metadata": r.metadata
                    }
                    f.write(json.dumps(result_dict) + "\n")
        
        elif output_format == "csv":
            import csv
            with open(output_file, 'w', newline='') as f:
                fieldnames = [
                    "id", "prompt", "generated_text", "success", 
                    "error", "generation_time"
                ]
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                
                for r in results:
                    writer.writerow({
                        "id": r.id,
                        "prompt": r.prompt,
                        "generated_text": r.generated_text,
                        "success": r.success,
                        "error": r.error,
                        "generation_time": r.generation_time
                    })
    
    def get_stats(self) -> Dict[str, Any]:
        """Get processing statistics"""
        return self.stats.copy()