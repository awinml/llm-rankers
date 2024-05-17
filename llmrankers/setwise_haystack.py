import logging
from typing import List, Optional, Dict, Any

from haystack import ComponentError, Document, component
from haystack.lazy_imports import LazyImport

logger = logging.getLogger(__name__)


with LazyImport(message="Run 'pip install git+https://github.com/awinml/llm-rankers.git'") as llm_ranker_import:
    from llmrankers.rankers import SearchResult
    from llmrankers.setwise import OpenAISetwiseLlmRanker, SetwiseLlmRanker, LlamaCPPSetwiseLlmRanker


@component
class SetwiseLLMRanker:
    """
    Implements the LLM based rankers using the Setwise method as proposed in
    "A Setwise Approach for Effective and Highly Efficient Zero-shot Ranking with Large Language Models"
    (https://arxiv.org/abs/2310.09497).
    """

    def __init__(
        self,
        model_name_or_path: str,
        tokenizer_name_or_path: Optional[str] = None,
        api_key: str = "",
        api_base_url: str = "",
        device: str = "cpu",
        num_child: int = 3,
        k: int = 10,
        scoring: str = "generation",
        method: str = "heapsort",
        num_permutation: int = 1,
        cache_dir: Optional[str] = None,
        ranker_type: str = "hugging_face",
        rate_limit: int = 30,
        rate_limit_window: int = 60,
        model_kwargs: Dict[str, Any] = {"n_ctx": 1024},
        generation_kwargs: Dict[str, Any] = {"max_tokens": 100, "temperature": 0.001},
    ):
        """
        Initialize a SetwiseLLMRanker.

        :param model:
            Local path or name of the model in Hugging Face's model hub, such as
            "google/flan-t5-large".
        :param tokenizer_name_or_path:
            Local path or name of the tokenizer in Hugging Face's model hub, such as
            "google/flan-t5-large".
        :param api_key:
            API key for the OpenAI API.
        :param api_base_url:
            Base URL for the OpenAI API.
        :param num_child:
            Number of children to use for the SetwiseLlmRanker.
        :param k:
            Number of results to return.
        :param scoring:
            Scoring method for the SetwiseLlmRanker.
        :param method:
            Ranking Method for the SetwiseLlmRanker.
        :param num_permutation:
            Number of permutations for the SetwiseLlmRanker.
        :param cache_dir:
            Cache directory for the SetwiseLlmRanker.
        :param ranker_type:
            Type of the SetwiseLlmRanker, should be one of ["hugging_face", "openai", "llamacpp"].
        :param rate_limit:
            Rate limit for the SetwiseLlmRanker.
        :param rate_limit_window:
            Rate limit window for the SetwiseLlmRanker.
        :param model_kwargs:
            Additional keyword arguments to pass to the model.
        :param generation_kwargs:
            Additional keyword arguments to pass to the generation function.
        """
        llm_ranker_import.check()

        # Raise error if the ranker type is not supported
        if ranker_type not in ["hugging_face", "openai", "llamacpp"]:
            err_msg = f"Unsupported ranker type: {ranker_type}. Please use 'hugging_face' or 'openai'"
            raise ValueError(err_msg)

        self.model_name_or_path = model_name_or_path
        self.tokenizer_name_or_path = tokenizer_name_or_path
        self.api_key = api_key
        self.api_base_url = api_base_url
        self.num_child = num_child
        self.k = k
        self.scoring = scoring
        self.method = method
        self.num_permutation = num_permutation
        self.cache_dir = cache_dir
        self.ranker_type = ranker_type
        self.device = device
        self.rate_limit = rate_limit
        self.rate_limit_window = rate_limit_window
        self.model_kwargs = model_kwargs
        self.generation_kwargs = generation_kwargs

    def warm_up(self):
        """
        Warm up the pair ranking model used for scoring the answers.
        """
        if self.ranker_type == "hugging_face":
            self.ranker = SetwiseLlmRanker(
                model_name_or_path=self.model_name_or_path,
                tokenizer_name_or_path=self.tokenizer_name_or_path,
                device=self.device,
                num_child=self.num_child,
                k=self.k,
                scoring=self.scoring,
                method=self.method,
                num_permutation=self.num_permutation,
                cache_dir=self.cache_dir,
            )
        elif self.ranker_type == "openai":
            self.ranker = OpenAISetwiseLlmRanker(
                model_name_or_path=self.model_name_or_path,
                api_key=self.api_key,
                api_base_url=self.api_base_url,
                num_child=self.num_child,
                k=self.k,
                method=self.method,
                rate_limit=self.rate_limit,
                rate_limit_window=self.rate_limit_window,
            )

        elif self.ranker_type == "llamacpp":
            self.ranker = LlamaCPPSetwiseLlmRanker(
                model_name_or_path=self.model_name_or_path,
                model_kwargs=self.model_kwargs,
                generation_kwargs=self.generation_kwargs,
                num_child=self.num_child,
                method=self.method,
                k=self.k,
            )

        else:
            err_msg = f"Unsupported ranker type: {self.ranker_type}. Please use 'hugging_face' or 'openai'"
            raise ValueError(err_msg)

    @component.output_types(documents=List[Document])
    def run(self, query: str, documents: List[Document]):
        """
        Returns a list of Documents ranked by their similarity to the given query.

        :param query:
            Query string.
        :param documents:
            List of Documents.
        """

        if not documents:
            return {"documents": []}

        if self.ranker is None:
            msg = "The component LLMBlenderRanker wasn't warmed up. Run 'warm_up()' before calling 'run()'."
            raise ComponentError(msg)

        search_results = [SearchResult(docid=doc.id, text=doc.content, score=0) for doc in documents]

        ranked_search_results = self.ranker.rerank(query, search_results)

        # Remap the search results to the original documents
        ranked_documents = []
        doc_dict = {doc.id: doc for doc in documents}

        for result in ranked_search_results:
            doc = doc_dict.get(result.docid)
            if doc:
                ranked_documents.append(doc)

        return {"documents": ranked_documents}
