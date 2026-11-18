# ⚡ New-Gen-Local-AI (`nglai`)

> **Engenharia de Software Agêntica 100% Local com Modelos 30B–35B MoE Comprimidos (Triple-Stack Compression + Arquitetura Compound AI)**

O **New-Gen-Local-AI** replica localmente a experiência completa de agentes de engenharia de software de última geração (planejamento arquitetural, leitura e edição cirúrgica multi-arquivo com diff, execução segura de terminal, indexação AST do repositório inteiro e auto-correção), rodando **modelos da classe 30B–35B (`Qwen3.6-35B-A3B` e `Qwen3-Coder-30B-A3B`)** mesmo em PCs com **8 GB de VRAM + 16 GB de RAM**!

---

## 🧠 Como Funciona a Compressão Triple-Stack + Compound AI

1. **Camada 1 — Pesos Dinâmicos (`Unsloth Dynamic 2.0` / `UD-IQ3_XXS` & `UD-Q3_K_XL`):**
   Quantização mista guiada por divergência KL que comprime blocos tolerantes para ~2–3 bits enquanto mantém camadas críticas de **Atenção e Tool Calling (JSON)** em alta precisão (6/8-bit), reduzindo modelos de ~65 GB (BF16) para **~12.8 a 13.8 GB**.
2. **Camada 2 — Memória de Contexto em Tempo Real (`Google TurboQuant` `turbo3` / `q4_0` KV Cache):**
   Comprime o KV Cache de execução em **4x a 6x**, viabilizando janelas reais de **32K a 64K+ tokens** gastando menos de ~0.7 GB de VRAM.
3. **Camada 3 — Tensor Split Cirúrgico (`GPU VRAM + System RAM`):**
   Aloca **100% das camadas de Atenção + KV Cache + primeiros blocos MoE na placa de vídeo** (ex: ~6.6 GB na RTX 3050 8GB) e distribui apenas os blocos restantes de Experts MoE (`-ot` / `ffn_*_exps=CPU`) na memória RAM (~6.8 GB), ativando apenas **~3B de parâmetros por token**!
4. **Arquitetura Compound AI (Nível Fronteira):**
   * **🗺️ AST Repo-Map Indexer:** Analisa classes, funções, assinaturas e imports do repositório inteiro para que o modelo entenda projetos complexos gastando 10x menos tokens.
   * **🧠 Deep Thinking (`<think>`) + Auto-Verificador:** O agente raciocina antes de tocar no código, valida erros de sintaxe (`SyntaxError` / JSON) no ato da edição e roda testes no terminal para se autocorrigir.
   * **💾 Hot SSD (256 GB) ↔ Cold HDD (1 TB) Vault:** Mantém sua biblioteca pesada de modelos no **HDD de 1 TB** e promove automaticamente para o **SSD rápido** apenas o modelo ativo no momento!

---

## 🖥️ Interfaces Inclusas

* **CLI Interativa (`nglai` / `nglai chat`):** Terminal rico com diffs coloridos, blocos de raciocínio `<think>`, aprovação de comandos destrutivos e *slash commands* (`/hardware`, `/models`, `/use`, `/map`, `/memory`, `/think`, `/ide`).
* **Web Workspace estilo Antigravity (`nglai web`):** Painel local em `http://127.0.0.1:7860` com árvore de arquivos, visualizador de código e Unified Diffs ao vivo, chat WebSocket com cards de ferramentas e barra de telemetria VRAM/RAM/TurboQuant.
* **Gateway OpenAI-Compatible (`/v1`):** Plugue diretamente no **VS Code (Roo Code / Cline / Continue)** ou **OpenCode** usando `http://127.0.0.1:7860/v1` (com injeção automática opcional do AST Repo-Map via header `X-NGLAI-RepoMap: true`).

---

## 🚀 Guia Rápido de Instalação (Windows — Ex: `RTX 3050 8GB + 16GB RAM + SSD + HDD`)

> **Isolamento Zero-Disco-C:** O launcher `nglai.bat` isola automaticamente ambientes `.venv`, cache do `pip` e downloads do HuggingFace dentro da pasta do projeto.

### 1. Clonar o Repositório
```powershell
git clone git@github.com:IsLSteven/New-Gen-Local-AI.git
cd New-Gen-Local-AI
```

### 2. Instalar o Motor CUDA (`llama-server` / `TurboQuant`)
```powershell
.\scripts\setup_engine.ps1
# Opcional (compilar fork nativo llama-cpp-turboquant com CUDA Toolkit + CMake):
# .\scripts\setup_engine.ps1 -BuildTurboQuantFromSource
```

### 3. Configurar Cofre Híbrido SSD (256 GB) + HDD (1 TB)
```powershell
# Aponte o Hot Cache para uma pasta no seu SSD e o Cold Vault para o HDD de 1 TB:
.\nglai.bat vault set-paths --hot-ssd "C:\NGLAI_HotSSD" --cold-hdd "D:\NGLAI_VaultHDD"
```

### 4. Conferir o Diagnóstico de Hardware & Tensor Split Cirúrgico
```powershell
.\nglai.bat hardware
```

### 5. Baixar o Modelo Principal Comprimido (35B ou 30B MoE)
```powershell
# Opção 1 (Padrão Recomendado — Atenção Linear Híbrida + Deep Thinking, ~13.1 GB):
.\nglai.bat vault pull qwen3.6-35b-a3b-ud-iq3

# Opção 2 (Especialista Puro em Repositórios 256K ctx, ~12.8 GB):
.\nglai.bat vault pull qwen3-coder-30b-a3b-ud-iq3
```

### 6. Iniciar e Trabalhar!
```powershell
# Iniciar o motor comprimido em segundo plano:
.\nglai.bat engine start

# Opção A — Abrir o Painel Web estilo Antigravity + Gateway /v1 (http://127.0.0.1:7860):
.\nglai.bat web --workspace "D:\MeuProjeto"

# Opção B — Usar direto no Terminal Interativo:
.\nglai.bat chat --workspace "D:\MeuProjeto"
```

---

## 🧪 Rodando a Suíte de Testes Automatizados

```powershell
.\.venv\Scripts\pytest.exe -v
```

---

## 📄 Licença

Distribuído sob a licença **Apache 2.0**. Veja [LICENSE](LICENSE) para detalhes.
