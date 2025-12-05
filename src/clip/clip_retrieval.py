from huggingface_hub import hf_hub_download
import importlib.util
from dotenv import load_dotenv
import os

from huggingface_hub import login
load_dotenv()
login(token=os.getenv("HF_TOKEN"))

class CLIPRetrieval():

    def __init__(self,model_name=None):
        load_dotenv()
        token = os.getenv("HF_TOKEN")
        py_file = hf_hub_download(
            repo_id="xuemduan/reevaluate-clip-retriever",
            filename="reevaluate_clip_retriever.py",
            token=token 
        )

        spec = importlib.util.spec_from_file_location("retriever", py_file)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        if model_name == None:
            self.retriever = module.CLIPRetriever.from_pretrained(
                "xuemduan/reevaluate-clip-retriever",
                local_embeddings_dir="data/embeddings",
                token=token 
            )
        else:
            self.retriever = module.CLIPRetriever.from_pretrained(
                "xuemduan/reevaluate-clip-retriever",
                model_name = model_name,
                local_embeddings_dir="data/embeddings",
                token=token 
            )
        
    def retrieval(self, query: str, alpha: float = 0.5):
        return self.retriever.search(query, alpha=alpha)
