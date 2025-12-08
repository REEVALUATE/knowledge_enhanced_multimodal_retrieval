"""
Content Description Builder for ArtKB Knowledge Graph
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Optional
from PIL import Image
import torch
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from tqdm import tqdm
import multiprocessing as mp
from multiprocessing import Queue, Process
import time


class MultiGPUContentDescriptionBuilder:
    """
    Content description builder with multi-GPU support.
    Distributes image processing across multiple GPUs.
    """
    
    def __init__(self, model_name: str = "Salesforce/blip2-opt-2.7b", num_gpus: int = None):
        """
        Initialize the multi-GPU content description builder.
        
        Args:
            model_name: HuggingFace model identifier for BLIP-2
            num_gpus: Number of GPUs to use (None = all available)
        """
        self.model_name = model_name
        
        # Detect available GPUs
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available! Multi-GPU processing requires GPU.")
        
        self.num_gpus = num_gpus if num_gpus else torch.cuda.device_count()
        print(f"Found {torch.cuda.device_count()} GPUs, using {self.num_gpus}")
        
        for i in range(self.num_gpus):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    
    @staticmethod
    def worker_process(
        gpu_id: int,
        model_name: str,
        task_queue: Queue,
        result_queue: Queue,
        # metadata_folder: str
    ):
        """
        Worker process that runs on a single GPU.
        
        Args:
            gpu_id: GPU device ID
            model_name: Model name to load
            task_queue: Queue of (uuid, image_path) tasks
            result_queue: Queue for results
            metadata_folder: Path to metadata folder
        """
        # Set this process to use specific GPU
        torch.cuda.set_device(gpu_id)
        device = f'cuda:{gpu_id}'
        
        print(f"[GPU {gpu_id}] Loading model...")
        
        # Load model on this GPU
        processor = Blip2Processor.from_pretrained(model_name)
        model = Blip2ForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map={"": gpu_id}
        )
        model.eval()
        
        print(f"[GPU {gpu_id}] Model loaded, ready to process")
        
        processed = 0
        
        while True:
            try:
                # Get task from queue
                task = task_queue.get(timeout=1)
                
                if task is None:  # Poison pill
                    break
                
                uuid, image_path = task
                
                # Process image
                try:
                    # Load image
                    image = Image.open(image_path).convert('RGB')
                    
                    # Generate 5 descriptions with temperature sampling
                    descriptions = []
                    
                    # First description - greedy
                    inputs = processor(image, return_tensors="pt").to(device, torch.float16)
                    with torch.no_grad():
                        generated_ids = model.generate(
                            **inputs,
                            max_length=50,
                            min_length=10,
                            num_beams=5,
                            repetition_penalty=1.5,
                            length_penalty=1.0,
                            early_stopping=True
                        )
                    desc = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
                    descriptions.append(desc)
                    
                    # 4 more with temperature sampling for variety
                    for i in range(4):
                        inputs = processor(image, return_tensors="pt").to(device, torch.float16)
                        with torch.no_grad():
                            generated_ids = model.generate(
                                **inputs,
                                max_length=50,
                                min_length=10,
                                do_sample=True,
                                temperature=0.3 + i*0.1,
                                top_p=0.9,
                                repetition_penalty=1.5
                            )
                        desc = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
                        descriptions.append(desc)

                    
                    # Put result in queue
                    result_queue.put({
                        'uuid': uuid,
                        'descriptions': descriptions,
                        # 'object_type': object_type,
                        'success': True
                    })
                    
                    processed += 1
                    
                except Exception as e:
                    result_queue.put({
                        'uuid': uuid,
                        'descriptions': [""] * 5,
                        # 'object_type': None,
                        'success': False,
                        'error': str(e)
                    })
            
            except:
                continue
        
        print(f"[GPU {gpu_id}] Processed {processed} images, shutting down")
    
    def process_all_images(
        self,
        image_folder: str,
        # metadata_folder: str,
        output_folder: str
    ):
        """
        Process all images using multiple GPUs.
        
        Args:
            image_folder: Path to folder containing images
            metadata_folder: Path to metadata folder
            output_folder: Path to output folder
        """
        # Create output folder
        Path(output_folder).mkdir(parents=True, exist_ok=True)
        
        # Get uuids to be processed 
        image_uuids = set([f.split(".")[0] for f in os.listdir(image_folder) if f.endswith('.jpg')])
        print(f"Found {len(image_uuids)} images in image folder")
        content_uuids = set([f.split(".")[0] for f in os.listdir(output_folder) if f.endswith('.json')])
        print(f"Found {len(content_uuids)} already processed images in output folder")

        process_uuids = image_uuids - content_uuids
        print(f"Need to process {len(process_uuids)} images (skipping {len(image_uuids) - len(process_uuids)} already done)")

        # image_files = [f for f in os.listdir(image_folder) if f.endswith('.jpg')]
        image_files = [f"{uuid}.jpg" for uuid in process_uuids]
        print(f"Found {len(image_files)} images to process")
        
        # Filter out already processed
        tasks = []
        for img_file in image_files:
            uuid = os.path.splitext(img_file)[0]
            output_path = os.path.join(output_folder, f"{uuid}.json")
            
            if not os.path.exists(output_path):
                image_path = os.path.join(image_folder, img_file)
                tasks.append((uuid, image_path))
        
        print(f"Need to process {len(tasks)} images (skipping {len(image_files) - len(tasks)} already done)")
        
        if len(tasks) == 0:
            print("All images already processed!")
            return
        
        # Create queues
        task_queue = Queue(maxsize=self.num_gpus * 10)
        result_queue = Queue()
        
        # Start worker processes
        workers = []
        for gpu_id in range(self.num_gpus):
            p = Process(
                target=self.worker_process,
                args=(gpu_id, self.model_name, task_queue, result_queue)
            )
            p.start()
            workers.append(p)
        
        # Wait for models to load
        print("Waiting for models to load on all GPUs...")
        time.sleep(10)
        
        # Start result writer thread
        import threading
        
        def result_writer():
            processed = 0
            errors = 0
            
            pbar = tqdm(total=len(tasks), desc="Processing images")
            
            while processed + errors < len(tasks):
                try:
                    result = result_queue.get(timeout=1)
                    
                    if result['success']:
                        # Save result
                        output_data = {
                            "uuid": result['uuid'],
                            "content_descriptions": result['descriptions'],
                            "model_info": {
                                "model": self.model_name,
                                # "object_type": result['object_type'],
                                "prompts_used": []  # Not applicable for captioning mode
                            }
                        }
                        
                        output_path = os.path.join(output_folder, f"{result['uuid']}.json")
                        with open(output_path, 'w', encoding='utf-8') as f:
                            json.dump(output_data, f, indent=2, ensure_ascii=False)
                        
                        processed += 1
                    else:
                        errors += 1
                        print(f"Error processing {result['uuid']}: {result.get('error', 'Unknown')}")
                    
                    pbar.update(1)
                
                except:
                    continue
            
            pbar.close()
            
            print(f"\nProcessing complete!")
            print(f"  Successful: {processed}")
            print(f"  Errors: {errors}")
        
        writer_thread = threading.Thread(target=result_writer)
        writer_thread.start()
        
        # Feed tasks to queue
        print("Distributing tasks to GPUs...")
        for task in tasks:
            task_queue.put(task)
        
        # Send poison pills
        for _ in range(self.num_gpus):
            task_queue.put(None)
        
        # Wait for all workers to finish
        for p in workers:
            p.join()
        
        # Wait for writer to finish
        writer_thread.join()
        
        print("\nAll done!")


def main():
    import argparse

    mp.set_start_method('spawn', force=True)
    
    parser = argparse.ArgumentParser(description='Content Description Builder')
    parser.add_argument('--image_folder', type=str, default="ArtKB/images/", help='Path to image folder')
    parser.add_argument('--output_folder', type=str, default="ArtKB/texts/content_texts", help='Path to output folder')
    parser.add_argument('--model', type=str, default='Salesforce/blip2-opt-2.7b', 
                        help='Model name (default: blip2-opt-2.7b)')
    parser.add_argument('--num_gpus', type=int, default=None, 
                        help='Number of GPUs to use (default: all available)')
    
    args = parser.parse_args()
    
    builder = MultiGPUContentDescriptionBuilder(
        model_name=args.model,
        num_gpus=args.num_gpus
    )
    
    builder.process_all_images(
        image_folder=args.image_folder,
        output_folder=args.output_folder
    )


if __name__ == "__main__":
    main()