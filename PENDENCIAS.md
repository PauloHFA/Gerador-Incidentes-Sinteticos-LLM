# Pendências — Gerador-Incidentes-Sinteticos-LLM (`incidentforge`)

## 1. Implementado e Verificado Nesta Auditoria

- **Correções críticas no núcleo (`incidentforge/`):**
  - `generator.py`:
    - Corrigido `NameError` em `_generate_with_validation()` passando `service` para `_validate_incident()`.
    - Substituída a iteração duplicada por worker (`iter(tasks)`) por `asyncio.Queue` compartilhada para evitar processamento duplicado em concorrência.
    - Corrigido deadlock em `_background_writer()` com chamada obrigatória de `task_done()` e sentinela `"__stop__"`.
    - Conectado `PlannerConfig` e `FailureCatalog` ao `IncidentGenerator.initialize()`, além de chave única indexada por tarefa (`_task_key`) e proteção de estado com `asyncio.Lock`.
    - Integrado `MultiLayerValidator` (Schema + Semântico + Segurança/PII) e retries de transporte (`RetryConfig`) no fluxo de geração, registrando idioma, estilo e taxa de aprovação na 1ª tentativa.
  - `cli.py` & `config.py`:
    - Conectado `incidentforge generate` ao pipeline completo: configuração em camadas (`CLI > INCIDENTFORGE_* > incidentforge.toml > defaults`), `--count`, `--severities`, `--lang`, `--style`, `--architecture-profile`, `--failures-mode`, `--max-tokens-budget`, `--dry-run`, deduplicação e exportação real em `output_dir` via `export_incidents()`.
    - Corrigido `validate` para carregar a topologia uma única vez, aceitar arquivo `.jsonl` direto ou diretório, e respeitar `--schema-only`, `--semantic/--no-semantic` e `--security/--no-security`.
    - Adicionado subcomando `judge` com suporte ao template `incidentforge/prompts/judge_v1.j2`.
  - `retry.py`:
    - Corrigido `create_retry_policy()` para usar `retry_if_exception(is_retryable_exception)` (sem retry em `401/403/400`) e respeitar cabeçalho `Retry-After` (`_WaitWithRetryAfter`).
  - `client.py`:
    - Suporte a `INCIDENTFORGE_API_KEY` além de `OPENAI_API_KEY`, fallback automático quando `response_format` não é suportado, seleção de `HTTPXClient` via `use_httpx=True` e extração segura de JSON.
  - `planner.py`, `prompts.py`, `models.py`, `report.py`, `__init__.py`:
    - Cópia independente (`deep=True`) de `GenerationTask` em `_select_with_variation()` e filtro por `--severities`.
    - Persistência do template `incidentforge/prompts/judge_v1.j2` e sanitização anti-injeção para campos de topologia.
    - Campos opcionais `language` e `style` em `IncidentMetadata`, correção de `by_language`, métricas reais de validação/rejeição/1ª tentativa em `QualityReporter` e atualização de `__all__` em `__init__.py`.
- **Qualidade, limpeza e artefatos:**
  - Removidos ~60 arquivos temporários/resíduos da raiz (`fix_*.py`, `debug_*.py`, `page_*.txt`, `pdf_content.txt`, `test_output*`, `test_report/`, `tests/test_failures_fixed.py`).
  - Adicionados `LICENSE` (MIT), `.env.example` (valores `[REDACTED]`), `Dockerfile`, `configs/incidentforge.toml`, `.github/codeql-config.yml` e schemas JSON atualizados em `schema/`.
  - Corrigidos `pyproject.toml` (`[project.scripts]`, `[tool.ruff.lint]`, `[tool.mypy]`) e `.github/workflows/ci.yml` (`main`/`master`).
  - Validação executada: `ruff check .` (0 erros), `ruff format --check .` (31 arquivos formatados), `mypy incidentforge` (0 erros em 15 arquivos) e `pytest -v` (**142/142** testes passando, **90,39%** de cobertura).

---

## 2. Melhorias Opcionais Futuras

- Implementar geração de *planos de experimento de caos* como segundo tipo de documento (item 7 da seção 13 da especificação, caso o escopo evolua além de post-mortems).
- Adicionar comando CLI ou script dedicado para gerar um golden dataset de 500–1.000 incidentes com benchmark de latência publicado no `README.md`.
- Publicar documentação estática com `mkdocs-material` e configurar publicação automatizada no PyPI / Hugging Face Hub.

---

## 3. O que Depende do Usuário / Infraestrutura Externa

- **Credenciais de LLM:** definir `INCIDENTFORGE_API_KEY` (ou `OPENAI_API_KEY`) ou endpoint local compatível (`vLLM` / `Ollama` via `--base-url`) para executar gerações e avaliações `judge` reais fora de mocks.
- **Docker & Busca Vetorial (`examples/vector_search`):** executar em ambiente com Docker daemon ativo para validar o `Dockerfile` e subir `ChromaDB` / `PostgreSQL + pgvector`.
- **Renomeação / Branch no GitHub:** decidir se deseja renomear o repositório/pasta para o padrão descritivo em português (`<Tipo>-<Dominio>-<Qualificador>`) e se deseja alinhar a branch `master` para `main`.
- **Commit / Push:** autorizar o `git commit` e `git push` das alterações atuais para `origin/master`.
