from typing import Union, Set, List
from langchain_community.embeddings.huggingface import HuggingFaceBgeEmbeddings

class ReHuggingFaceBgeEmbeddings(HuggingFaceBgeEmbeddings):
    def embed_query(self, text: Union[str, Set[str]]) -> List[float]:
        if isinstance(text, set):
            text = " ".join(text)
        else:
            text = text.replace("\n", " ")
        return self.embed_documents([text])[0]

