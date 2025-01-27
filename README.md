RAG em PDFs com Deepseek-r1 e InMemoryVectorStore 📄🤖
Descrição

Este projeto implementa um sistema de RAG (Retrieval-Augmented Generation) para consultas inteligentes em documentos PDF. Ele utiliza Deepseek-r1 tanto para a geração de embeddings quanto como modelo de linguagem (LLM), proporcionando uma solução unificada e eficiente para análise de documentos. O armazenamento de embeddings é feito usando InMemoryVectorStore para rápida recuperação de informações.

O objetivo é oferecer uma ferramenta poderosa para buscar e processar informações em PDFs usando inteligência artificial.
Arquitetura

    Pré-processamento de PDFs:
        Os PDFs são extraídos e convertidos em texto.
        Embeddings semânticos são gerados com o Deepseek-r1 via OllamaEmbeddings.
    Armazenamento:
        Os embeddings são armazenados no InMemoryVectorStore.
    Consultas:
        O usuário faz perguntas relacionadas aos PDFs.
        O sistema recupera os trechos relevantes com base em similaridade semântica.
        O Deepseek-r1, como modelo de linguagem, gera respostas contextualizadas.

Tecnologias Utilizadas

    Deepseek-r1: Modelo usado para geração de embeddings e respostas em linguagem natural.
    InMemoryVectorStore: Armazenamento leve e rápido de embeddings para desenvolvimento local.
    Python: Linguagem principal para o desenvolvimento.
    LangChain: Para orquestrar o fluxo de trabalho do RAG.
    PDFPlumber: Para extração de texto de PDFs.