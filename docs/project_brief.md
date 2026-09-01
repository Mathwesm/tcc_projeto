# Projeto "Centauro-Lite" — Briefing técnico

> Este documento é o **prompt/spec de referência** do projeto: consolida tudo que foi
> decidido até agora (contexto, dados, modelo, método, hardware, métricas, riscos) para
> guiar tanto quem for escrever o código quanto qualquer assistente de IA que for
> ajudar na implementação. Sempre que uma decisão técnica mudar, atualize este
> arquivo primeiro — ele é a fonte da verdade do projeto, não a apresentação em
> PowerPoint (que é só o material de proposta/banca).

## 1. Objetivo

Treinar um modelo alternativo ao **Centaur** (Binz et al., 2025, *Nature*) para prever
o comportamento humano em tarefas psicológicas, usando um modelo open-source menor
(**Qwen3**) e ajuste fino (fine-tuning) com métodos diferentes dos usados no paper
original — tudo isso rodando em hardware **muito** mais modesto do que o do estudo
original.

**Meta de desempenho:** chegar o mais perto possível do valor de referência do
Centaur na métrica de log-verossimilhança negativa (NLL = **0,44**), idealmente
empatando ou superando; na pior hipótese, ficar um pouco abaixo — mas sempre
superando claramente o baseline sem ajuste (Llama-base = **0,58** no paper original;
o baseline real do nosso projeto é o **Qwen3 sem fine-tuning**, que também deve ser
medido). Não tem problema científico nenhum em ficar abaixo do Centaur — o ponto do
TCC é mostrar o quão perto dá pra chegar gastando uma fração dos recursos.

## 2. Contexto (recapitulando o que já pesquisamos)

- **Centaur**: fine-tuning do Llama 3.1 70B com QLoRA (rank 8, ~0,15% dos parâmetros
  treináveis) sobre o dataset Psych-101. Treino: ~5 dias em 1x A100 80GB. Biblioteca:
  `unsloth`. Métrica principal: log-verossimilhança negativa média sobre os tokens de
  escolha humana (menor = melhor). Resultado: NLL 0,44 (Centaur) vs. 0,58
  (Llama-base) vs. 0,56 (modelos cognitivos especializados, média de 14 modelos).
- **Psych-101**: `marcelbinz/Psych-101` no Hugging Face. 60.092 linhas (1 por
  participante), 160 experimentos, 10.681.650 escolhas. Campos: `text` (prompt
  completo em linguagem natural, com o formato de trial-by-trial), `experiment`
  (formato `autor+ano/arquivo.csv`), `participant` (id). As escolhas humanas em cada
  trial são marcadas com `<<` e `>>` no meio do texto — é sobre esses tokens que a
  função de perda é calculada, o resto é só contexto (loss masking).
- Existe também `marcelbinz/Psych-101-test`, um conjunto de teste separado
  disponibilizado pelos autores — vale conferir se ele cobre os domínios que vamos
  usar antes de decidir a fonte definitiva do split de avaliação.

## 3. Restrição de hardware (o ponto de partida real do projeto)

- GPU disponível para os testes locais: **notebook com RTX 3060, ~6-8GB de VRAM**
  (a confirmar o valor exato antes de fixar os hiperparâmetros de memória).
- Isso descarta fine-tuning full ou LoRA em modelos grandes localmente. A rota
  viável é **QLoRA em 4-bit** sobre um modelo pequeno da família **Qwen3**:
  - `Qwen3-0.6B` — mais seguro, cabe com folga, bom para iterar rápido.
  - `Qwen3-1.7B` — alvo principal recomendado: ainda cabe confortavelmente em
    6-8GB com QLoRA 4-bit + `unsloth` (a documentação do Unsloth confirma que até o
    Qwen3-14B cabe numa T4 de 16GB com QLoRA — um 1.7B em 6-8GB tem folga real).
  - `Qwen3-4B` — só como stretch goal, se sobrar VRAM com `max_seq_length` curto e
    batch size 1; testar por último.
- Biblioteca de fine-tuning: **`unsloth`** — mesma biblioteca usada no paper do
  Centaur, o que também é bom argumento pro TCC (comparação mais direta de método).

## 4. Decisões técnicas fixadas

| Decisão | Valor | Por quê |
|---|---|---|
| Modelo base | Qwen3-1.7B (alvo principal) | Cabe em 6-8GB com QLoRA 4-bit, é a geração atual da família Qwen |
| Método | QLoRA 4-bit, rank 8 | Espelha o Centaur (rank 8) pra comparação justa; variações de rank entram depois como experimento adicional |
| Biblioteca | `unsloth` + `transformers` + `peft` + `bitsandbytes` | Mesma stack usada no Centaur, eficiente em VRAM baixa |
| Dataset | `marcelbinz/Psych-101` (subconjunto inicial) | O mesmo do Centaur — garante comparabilidade |
| Escopo inicial | 2-3 experimentos específicos (ver seção 6) | Validar o pipeline antes de escalar para os 160 experimentos |
| Split | 90/10 por **participante** (não por linha solta) | Mesma metodologia do Centaur; evita vazamento de dados do mesmo participante entre treino e validação |
| Métrica | NLL média sobre os tokens `<<...>>` (masked loss) | Mesma métrica do paper — é o que permite comparar os números diretamente |
| Métricas de referência | Centaur = 0,44 · Llama-base = 0,58 (do paper) · Qwen3-base = a medir | Balizas fixas pra saber onde o projeto está |

## 5. Estrutura de projeto sugerida

```
centauro-lite/
├── docs/
│   └── project_brief.md        <- este arquivo
├── requirements.txt
├── README.md
├── src/
│   ├── 01_explore_data.py      <- EDA: baixa o Psych-101, cataloga os 160 experimentos
│   ├── 02_prepare_dataset.py   <- filtra experimentos-alvo, faz split 90/10, aplica o masking <<>>
│   ├── 03_train_qlora.py       <- fine-tuning QLoRA do Qwen3 com unsloth
│   └── 04_evaluate.py          <- calcula NLL no baseline e no modelo treinado, compara com o Centaur
├── data/
│   ├── experiment_catalog.csv  <- gerado pelo 01 (lista dos 160 experimentos)
│   └── processed/              <- gerado pelo 02 (train/ e val/)
└── outputs/
    ├── qwen3-centauro-lite-adapter/  <- adapter LoRA treinado (gerado pelo 03)
    └── eval_results.json             <- resultado da avaliação (gerado pelo 04)
```

## 6. Como um cientista/engenheiro de dados tocaria isso — roadmap

1. **EDA primeiro, sem exceção** (`01_explore_data.py`): antes de treinar qualquer
   coisa, listar os 160 experimentos reais do Psych-101 e escolher, com base no que
   existe de fato (não em suposição), os 2-3 experimentos que vão representar os
   domínios do estudo de caso da apresentação (decisão sob risco / aprendizado /
   categorização). **Isso ainda está em aberto** — já confirmamos exemplos reais para
   "decisão sob risco" (`peterson2021using`) e "categorização"
   (`badham2017deficits`), mas o experimento de "aprendizado por reforço" precisa
   ser escolhido a partir do catálogo gerado por este script antes de seguir.
2. **Baseline antes de treinar** (`04_evaluate.py` rodado no modelo sem ajuste):
   medir o NLL do Qwen3 "cru" nesse subconjunto — isso vira o novo "Llama-base" de
   comparação, específico do nosso setup.
3. **Preparar os dados** (`02_prepare_dataset.py`): aplicar o masking dos tokens
   `<<...>>`, dividir 90/10 por participante, salvar em disco (pra não repetir esse
   processamento a cada treino).
4. **Treinar** (`03_train_qlora.py`): primeira rodada com os hiperparâmetros
   default do script (espelhando o Centaur: rank 8, lr 5e-5, batch efetivo ~32 via
   gradient accumulation). Rodar poucas épocas primeiro (1) pra validar que o
   pipeline funciona de ponta a ponta antes de qualquer ajuste fino de
   hiperparâmetro.
5. **Avaliar** (`04_evaluate.py` de novo, agora com o adapter treinado): comparar o
   NLL do modelo treinado contra o baseline (passo 2) e contra os valores fixos do
   Centaur/Llama-base do paper.
6. **Iterar**: só depois que o pipeline básico funciona ponta a ponta é que faz
   sentido testar variações (rank do LoRA, learning rate, mais épocas, mais
   experimentos) — isso é justamente o "método alternativo" que diferencia este
   trabalho do Centaur.
7. **Escalar** (fora do escopo inicial): depois de validado no subconjunto, avaliar
   se dá pra rodar em nuvem (Colab/Kaggle com GPU maior, ou serviço pago) para
   cobrir mais experimentos do Psych-101 completo, se o tempo do TCC permitir.

## 7. Riscos já mapeados (herdados da apresentação, agora com ação associada)

- **VRAM insuficiente mesmo em 4-bit** → reduzir `max_seq_length`, usar
  `Qwen3-0.6B` em vez de `1.7B`, batch size 1 com mais gradient accumulation.
- **Prazo do TCC** → escopo inicial deliberadamente pequeno (2-3 experimentos, não
  os 160), documentado como decisão consciente, não corte de última hora.
- **Viés do dataset / comparação indireta (tokenizador diferente)** → já
  documentado na apresentação; não afeta o código, é uma ressalva a manter na
  discussão dos resultados.
- **Overfitting num subconjunto pequeno** → monitorar a loss de validação a cada
  época, não só a de treino; manter o split por participante (nunca por linha) pra
  não vazar dado do mesmo histórico entre treino e validação.

## 8. O que ainda precisa de decisão humana (não travar sozinho)

- Confirmar a VRAM exata da RTX 3060 (6GB ou 8GB muda o teto de `max_seq_length` e
  batch size).
- Escolher o experimento real de "aprendizado por reforço" a partir do catálogo do
  passo 1.
- Decidir se o treino roda 100% local ou se, quando a VRAM não bastar, migra pontos
  específicos (ex.: uma rodada maior) para uma GPU de nuvem gratuita (Colab/Kaggle).
