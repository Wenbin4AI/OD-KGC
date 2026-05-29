from src.model.evidence_extractor import EvidenceExtractor
from src.model.ontology_filter import OntologyFilter
from src.model.compressor import EvidenceCompressor
from src.llm.client import LLMClient
from src.evaluation.evaluator import Evaluator
from src.data.kg_loader import KGLoader
from src.data.ontology_loader import OntologyLoader

def main(config_path):
    kg = KGLoader.load(config_path)
    ontology = OntologyLoader.load(config_path)
    queries = kg.get_queries()

    extractor = EvidenceExtractor(kg)
    filterer = OntologyFilter(ontology)
    compressor = EvidenceCompressor()
    llm = LLMClient()

    for q in queries:
        evidence = extractor.extract(q)
        filtered = filterer.apply(evidence, q)
        compressed = compressor.select(filtered)
        prediction = llm.predict(q, compressed)
        # 可选择立即评估或缓存结果

if __name__ == "__main__":
    main("configs/fb15k237.yaml")