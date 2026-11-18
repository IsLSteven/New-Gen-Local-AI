# ⚡ New-Gen-Local-AI (`nglai`)

> **Agente Autônomo de Engenharia de Software 100% Local** — Roda modelos de código de classe **30B–35B MoE** em hardware doméstico combinando **Compressão Triple-Stack** (*Unsloth Dynamic 2.0 + KV Cache Quantization / TurboQuant + Surgical GPU/RAM Tensor Split*) e **Arquitetura Compound AI** (*AST Repo-Map + Deep Thinking + Auto-Verify*).

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-3776AB.svg)](https://www.python.org/downloads/)
[![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20Linux%20%7C%20macOS-success.svg)](#)

---

## ✨ O Que Este Projeto Faz?

Diferente de um simples chat local, o **New-Gen-Local-AI** funciona como um **Engenheiro de Software Agêntico completo** (no estilo *Antigravity / Claude Code / OpenCode*), rodando inteiramente na sua máquina, sem mensalidades e sem enviar seu código para a nuvem:

- 📂 **Lê, cria e edita múltiplos arquivos cirurgicamente** (`read_file`, `write_file`, `edit_file`) exibindo *Unified Diffs* coloridos em tempo real.
- 🗺️ **Enxerga o repositório inteiro (`AST Repo-Map`):** Indexa classes, funções, assinaturas e *imports* (Python, TypeScript/JS, Rust, Go, C#, Java, C++) para entender projetos grandes gastando **10x menos tokens**.
- 🧠 **Raciocínio Profundo (`<think>`) + Auto-Verificação:** Planeja antes de tocar no código, valida erros de sintaxe (`SyntaxError` / JSON) no exato instante da edição e executa testes no terminal (`run_command`) para corrigir os próprios bugs.
- 🗜️ **Triple-Stack Compression Engine:**
  1. **Pesos Dinâmicos (`Unsloth Dynamic 2.0`):** Comprime modelos gigantes de ~65 GB para **~11.4 a 12.8 GB** protegendo camadas de Atenção e *Tool Calling* em alta precisão.
  2. **Compressão de Memória KV Cache (`TurboQuant` / `q4_0` + FlashAttention):** Reduz o uso de memória de conversas e arquivos longos em **4x a 6x**, habilitando 16K–64K+ de contexto sem estourar a placa de vídeo.
  3. **Tensor Split Cirúrgico (`GPU VRAM + System RAM`):** Divide automaticamente camadas de Atenção na GPU e blocos *Mixture-of-Experts* (`MoE`) na RAM.
- 💾 **Cofre Híbrido SSD ↔ HDD (`Hot/Cold Vault`) & Zero-Disco-C:** Guarda sua coleção pesada de modelos em um HD secundário e mantém no SSD rápido apenas o modelo em uso — sem jogar lixo na pasta `%USERPROFILE%\.cache` do Windows!
- 🖥️ **3 Formas de Usar:** **Painel Web Local** (estilo IDE/Workspace), **Terminal Interativo (CLI)** e **Gateway `/v1` compatível com OpenAI** (para plugar no VS Code com *Roo Code*, *Cline* ou *Continue*).

---

## 📊 Qual Modelo Escolher Para o Seu Computador?

O sistema detecta sua GPU/VRAM e RAM automaticamente, mas você pode escolher qualquer modelo do catálogo integrado:

| Seu Hardware | Modelo Recomendado (`ID`) | Tamanho | Como Roda |
| :--- | :--- | :---: | :--- |
| **Qualquer PC / Teste Rápido** *(4GB+ VRAM ou só CPU)* | `qwen2.5-coder-1.5b-draft` | **1.12 GB** | ⚡ Instantâneo (~60+ tok/s)! Ideal para testar toda a ferramenta em 1 minuto. |
| **GPU 4GB VRAM** *(ex: GTX 1650 / GTX 1050 Ti)* | `qwen3-4b-ud-q4kxl` | **2.37 GB** | 🔥 100% GPU com Unsloth Dynamic 2.0, *Deep Thinking* (`<think>`) e *Tool Calling*. |
| **GPU 6GB–8GB VRAM** *(foco em velocidade máxima)* | `qwen3-8b-q4km` | **5.03 GB** | 🚀 100% na GPU (~40–55 tok/s) com modo *Thinking* e *Tool Calling*. |
| **GPU 8GB VRAM** *(foco em código denso 14B)* | `qwen2.5-coder-14b-ud-iq3m` | **7.34 GB** | 🔥 Especialista 14B comprimido para caber em 8GB VRAM. |
| **GPU 8GB VRAM + 16GB RAM** *(ex: RTX 3050/3060/4060)* | **`qwen3.6-35b-a3b-ud-iq3`** *(ou `qwen3-coder-30b-a3b-ud-iq3`)* | **12.30 GB** *(11.97 GB)* | 🏆 **Cérebro 30B–35B MoE!** Divide ~6.5 GB na GPU + ~6.6 GB na RAM (ativa só 3B/token). |
| **Notebook 4GB VRAM + 32GB RAM** *(ex: GTX 1650 + 32GB)* | `qwen3.6-35b-a3b-ud-iq3` | **12.30 GB** | 🧠 Atenção na GPU (~3 GB) + Experts MoE nos 32 GB de RAM. |
| **Workstation 32GB+ RAM ou GPU 16–24GB+** | `qwen3-coder-30b-a3b-ud-q4kxl` | **16.45 GB** | 💎 Qualidade máxima 4-bit XL praticamente idêntica ao BF16 original. |

---

## 🚀 Passo a Passo: Como Baixar e Utilizar (Do Zero)

### Pré-requisitos
1. **Git** instalado ([baixar Git](https://git-scm.com/downloads)).
2. **Python 3.11 ou superior** ([baixar Python](https://www.python.org/downloads/)) — ⚠️ *No Windows, marque a caixinha **"Add python.exe to PATH"** durante a instalação!*
3. *(Opcional, mas recomendado)* Driver atualizado da sua placa de vídeo (NVIDIA, AMD ou Intel). Se não tiver placa dedicada, o projeto instala automaticamente o modo CPU!

---

### Opção A — Modo Automático em 1 Comando (Recomendado!)

#### 1. Clone o repositório
Abra o terminal (**PowerShell / CMD** no Windows, ou **Bash / Zsh** no Linux/macOS) em um disco com bom espaço livre e rode:

```bash
git clone https://github.com/IsLSteven/New-Gen-Local-AI.git
cd New-Gen-Local-AI
```

#### 2. Inicie com o Launcher Automático

**No Windows:**
```powershell
.\start.bat
```
*(Ou dê dois cliques no arquivo `start.bat` na pasta do projeto!)*

**No Linux ou macOS:**
```bash
chmod +x start.sh nglai.sh
./start.sh
```

**O que acontece automaticamente na primeira vez?**
1. Cria um ambiente Python isolado (`.venv`) sem sujar seu sistema operacional.
2. Detecta sua placa de vídeo e baixa sozinho o motor oficial `llama-server` ideal (**CUDA** para NVIDIA, **Vulkan** para AMD/Intel, **Metal** para Mac M1/M2/M3/M4 ou **CPU**).
3. Abre o servidor local em **`http://127.0.0.1:7860`**.

#### 3. Baixe o Modelo e Comece a Usar pelo Navegador!
1. Abra **`http://127.0.0.1:7860`** no seu navegador.
2. No canto superior direito, escolha o modelo desejado na lista (para testar em 1 minuto, escolha `Qwen2.5-Coder-1.5B`; para poder máximo em projetos complexos, escolha `Qwen3.6-35B-A3B` ou `Qwen3-Coder-30B-A3B`).
3. Clique em **"⬇ Baixar Modelo"** e acompanhe a barra de progresso em tempo real.
4. Assim que concluir, clique em **"Iniciar Motor"** (o status ficará verde: `Engine Online`).
5. Pronto! Digite qualquer pedido no chat à direita (ex: *"Analise a estrutura deste projeto e crie um módulo de autenticação com testes"*).

---

### Opção B — Passo a Passo Completo pelo Terminal (CLI Avançada)

Se preferir controlar cada etapa via linha de comando ou usar o agente direto no terminal (estilo *Claude Code*):

```powershell
# 1. Ver diagnóstico completo da sua máquina e cálculo do Tensor Split:
.\nglai.bat hardware

# 2. (Opcional — Para quem tem SSD pequeno + HD grande de 1TB)
# Guarda todos os modelos no HD e mantém no SSD apenas o modelo ativo:
.\nglai.bat vault set-paths --hot-ssd "C:\NGLAI_HotSSD" --cold-hdd "D:\NGLAI_VaultHDD"

# 3. Instalar o motor de inferência (auto / cuda / vulkan / cpu):
.\nglai.bat engine install --backend auto

# 4. Ver catálogo de modelos e baixar o escolhido (suporta retomada se a internet cair!):
.\nglai.bat vault list
.\nglai.bat vault pull qwen3.6-35b-a3b-ud-iq3

# 5. Iniciar o motor em segundo plano:
.\nglai.bat engine start

# 6A. Abrir o Agente Interativo no Terminal apontando para qualquer pasta de projeto:
.\nglai.bat chat --workspace "C:\Caminho\Do\SeuProjeto"

# 6B. OU abrir o Painel Web + Gateway /v1 apontando para seu projeto:
.\nglai.bat web --workspace "C:\Caminho\Do\SeuProjeto"
```

*(Em Linux/macOS, basta substituir `.\nglai.bat` por `./nglai.sh`).*

#### Comandos Úteis Dentro do Chat no Terminal (`nglai chat`)
- `/hardware` — Exibe uso de VRAM/RAM e flags do Tensor Split.
- `/models` — Lista modelos disponíveis e mostra quais estão no SSD ou HDD.
- `/use <id>` — Troca o modelo ativo (ex: `/use qwen3-coder-30b-a3b-ud-iq3`).
- `/map` — Mostra o **AST Repo-Map** gerado para o seu projeto.
- `/memory <regra>` — Salva uma convenção permanente do projeto em `.nglai/memory.json`.
- `/think` — Liga/desliga o modo de raciocínio profundo `<think>`.
- `/ide` — Mostra o JSON pronto para conectar no VS Code.
- `/clear` / `/sair` — Limpa a sessão ou encerra.

---

## 🔌 Como Usar Dentro do VS Code (Roo Code, Cline, Continue) ou OpenCode

Com o `start.bat` ou `nglai web` rodando, você ganha automaticamente uma **API Local compatível com OpenAI** enriquecida com nosso **AST Repo-Map**:

- **API Provider:** `OpenAI Compatible`
- **Base URL:** `http://127.0.0.1:7860/v1`
- **API Key:** `nglai-local` *(qualquer valor não-vazio)*
- **Model ID:** `qwen3.6-35b-a3b-ud-iq3` *(ou o ID do modelo ativo)*
- **Custom Header (Opcional — injeta visão do repositório inteiro):** `X-NGLAI-RepoMap: true`

Você também pode gerar arquivos de configuração prontos rodando:
```powershell
.\.venv\Scripts\python.exe scripts\export_ide_configs.py
```

---

## 🛠️ Solução de Problemas (FAQ)

- **Meu Disco `C:` tem pouco espaço livre. Vai encher meu Windows?**
  Não! O `nglai` possui isolamento **Zero-Disco-C**: se você clonar o projeto em `D:\New-Gen-Local-AI`, absolutamente tudo (`.venv`, cache do `pip`, cache do HuggingFace, binários e modelos `.gguf`) ficará contido dentro de `D:\New-Gen-Local-AI`.
- **O download de um modelo grande parou na metade. Perdi tudo?**
  Não! O downloader usa *HTTP Range Resume* (`.gguf.part`). Basta rodar `nglai vault pull <id>` (ou clicar em *Baixar Modelo* novamente) que ele continua exatamente de onde parou.
- **Como ativar os kernels experimentais `turbo3` do fork `llama-cpp-turboquant`?**
  Por padrão, o motor oficial já usa quantização de KV Cache `q4_0` + FlashAttention pronta para uso. Caso tenha *CUDA Toolkit* + *CMake* instalados e queira compilar o fork experimental a partir do código-fonte no Windows, execute:
  ```powershell
  .\scripts\setup_engine.ps1 -BuildTurboQuantFromSource
  ```

---

## 🧪 Rodando os Testes Automatizados

```powershell
.\.venv\Scripts\pytest.exe -v
```

---

## 📄 Licença

Distribuído sob a licença **Apache 2.0**. Consulte [LICENSE](LICENSE) para mais informações.
