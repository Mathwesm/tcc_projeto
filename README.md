# Centauro-Lite

Fine-tuning QLoRA de modelos pequenos sobre o **Psych-101** para prever comportamento
humano em tarefas psicologicas, como alternativa ao **Centaur** (Binz et al., 2025,
*Nature*) — que usa Llama 3.1 70B e cinco dias de A100 80GB. Aqui: Qwen3-1.7B em 4-bit
num notebook com RTX 3060 de 6GB.

A pergunta do trabalho nao e "conseguimos bater 0,44". E **quanto do desempenho do
Centaur sobrevive quando o modelo encolhe 40x e o hardware 100x** — e o valor da
resposta depende inteiramente de a comparacao ser honesta.

Documento de referencia completo: [`docs/project_brief.md`](docs/project_brief.md).

## Stack

Python 3.12 · Poetry · loguru · pydantic · typer · datasets · transformers · Ruff · mypy · pytest

## Pre-requisitos

So a instalacao do Poetry difere entre sistemas. Do `poetry install` em diante os
comandos sao identicos nos tres.

<details>
<summary><b>Windows</b></summary>

```powershell
python -m pip install --user pipx
python -m pipx ensurepath
pipx install poetry
```

Abra um terminal novo depois do `ensurepath`. Defina `PYTHONUTF8=1` no ambiente: o
padrao do Windows e `cp1252` e corrompe as transcricoes do Psych-101.
</details>

<details>
<summary><b>macOS</b></summary>

```bash
brew install pipx
pipx ensurepath
pipx install poetry
```
</details>

<details>
<summary><b>Linux</b></summary>

```bash
sudo apt install pipx        # ou: python3 -m pip install --user pipx
pipx ensurepath
pipx install poetry
```
</details>

## Setup

```bash
git clone https://github.com/Mathwesm/tcc_projeto.git
cd tcc_projeto
poetry install
poetry run pre-commit install
cp .env.example .env   # no Windows: copy .env.example .env
```

O `.env` so e necessario para o alerta de falha no Telegram; o pipeline roda sem ele.

## Como rodar

Toda etapa e um subcomando que le a **mesma** configuracao de
[`configs/default.yaml`](configs/default.yaml):

| Comando | O que faz | Saida |
|---|---|---|
| `poetry run poe eda` | Cataloga os experimentos do Psych-101 | `data/interim/experiment_catalog.csv` |
| `poetry run poe prepare` | Filtra, balanceia, divide e fatia em janelas | `data/processed/dataset/` e `splits.json` |
| `poetry run poe evaluate` | Mede o NLL sobre as escolhas humanas | `outputs/eval_results.json` |
| `poetry run poe train` | Fine-tuning QLoRA | `outputs/adapter/` |
| `poetry run poe sweep` | Treina e avalia toda `configs/sweep/` | `outputs/<run>/` |
| `poetry run poe report` | Tabela de ablacao e graficos | `outputs/report.txt`, `outputs/figures/` |

As duas ultimas exigem GPU e `unsloth`, que nao sao instalados localmente — rode-as
no Kaggle, pelo notebook em [`notebooks/kaggle_train.ipynb`](notebooks/kaggle_train.ipynb).

Para rodar uma variacao sem tocar no default:

```bash
poetry run python -m centauro_lite eda --config configs/rank32.yaml
```

Atalhos de desenvolvimento: `poe gate` (portao completo), `poe test`, `poe lint`.

## Por que uma CLI e nao scripts numerados

A primeira versao do projeto era `01_explore_data.py` … `04_evaluate.py`, e cada um
carregava sua propria copia de `max_seq_length`. Quando essas copias divergem, nada
falha: o dataset e tokenizado num comprimento, o modelo treina em outro, e o NLL sai
errado sem uma linha de erro na tela.

`configs/default.yaml` e a unica fonte, e chave desconhecida no YAML e **rejeitada** —
um `max_seq_lenght` com typo para a execucao em vez de usar o default em silencio.

## O que a EDA ja mostrou

Rodada em 2026-09-01 com `max_seq_length = 2048`:

- **Os totais batem exatamente com os numeros publicados**: 60.092 participantes e
  10.681.650 escolhas. Isso valida a deteccao dos marcadores `<<...>>` contra a fonte —
  se o regex estivesse errado, a contagem nao fecharia.
- A coluna `experiment` tem **76 valores distintos**, nao os "160 experimentos" do
  paper. Como participantes e escolhas fecham, nao ha dado faltando: o paper conta
  experimentos por um criterio mais fino do que um arquivo por linha. Vale registrar a
  diferenca na metodologia em vez de repetir "160" sem checar.
- **Apenas 2 dos 76 experimentos cabem inteiros em 2048 tokens.** Os outros 74 precisam
  ser fatiados.

A coluna decisiva do catalogo e **`fits_in_window_pct`**: a fracao de participantes cuja
transcricao inteira cabe numa janela. Cada linha do Psych-101 e a sessao completa de um
participante — media de ~4,2 mil tokens e maximo de 57 mil, contra os 32.768 de contexto
que o Centaur usa. Truncar em 2048 nao perde "um pedaco do fim": perde tudo menos os
**primeiros trials**, justamente a fase em que a pessoa ainda nao aprendeu a tarefa e o
comportamento e menos previsivel. O NLL sobe por vies de amostragem e deixa de
significar qualquer coisa.

Por isso as transcricoes sao **fatiadas em janelas**, nao truncadas — e `windows_mean`
diz em quantas.

## O que o `prepare` produziu

Rodada em 2026-09-01 com `max_choices_per_domain = 30000`:

| Dominio | Experimento | Escolhas apos balanceamento |
|---|---|---|
| `risky_choice` | `peterson2021using` | 30.055 |
| `categorization` | `badham2017deficits` | 29.776 |
| `reinforcement_learning` | `kool2016when` (exp1+exp2) | 30.091 |

Antes do balanceamento a proporcao era de 37 para 1 a favor do `peterson2021using`, o
que faria o treino ser ~92% um experimento so — e a tese de especializacao
multi-dominio ficaria sem sentido.

Resultado: 572 participantes de treino (1.546 janelas) e 62 de validacao (169 janelas).

**A verificacao que importa**: 89.922 escolhas selecionadas contra 89.922 tokens
pontuados, diferenca **zero**. Nenhuma escolha foi perdida no fatiamento nem contada
duas vezes. Nenhum participante aparece dos dois lados do split, e os quatro arquivos
de experimento estao representados na validacao.

A ordem das operacoes e o ponto todo: **filtrar → balancear → dividir por participante
→ so entao fatiar em janelas**. Fatiar antes de dividir colocaria pedacos da mesma
sessao nos dois lados do split, e a validacao passaria a medir memorizacao em vez de
generalizacao.

## Onde o treino roda

O pipeline de dados (`eda`, `prepare`) roda no notebook local, em CPU. O treino e a
avaliacao rodam no **Kaggle**: a RTX 3060 Laptop tem 6 GB e o Windows come parte disso,
e o `unsloth` depende do Triton, cujo suporte a Windows nativo e fragil.

O Kaggle da ~30 horas semanais de GPU de 16 GB com cota publicada, ao contrario do
Colab gratuito, que nao informa quanto resta.

**Escolha `GPU T4 x2`, nao a P100.** O unsloth exige CUDA capability 7.0 ou maior: a T4
e 7.5 e a P100, sendo Pascal, e 6.0. Nao e uma questao de desempenho — abaixo de 7.0 o
unsloth nao carrega. O notebook restringe a execucao a uma das duas T4, porque o unsloth
nao lida bem com multiplas GPUs e um modelo de 1.7B em 4-bit sobra numa so. O notebook pronto esta em
[`notebooks/kaggle_train.ipynb`](notebooks/kaggle_train.ipynb): ele clona este
repositorio, instala o unsloth, e roda baseline → treino → avaliacao → Minitaur.

**O split nao e re-sorteado la.** `data/processed/splits.json` e versionado no
repositorio e o `prepare` o le por padrao (`--reuse-splits`). Uma seed sozinha nao fixa
um split: ela fixa uma permutacao da ordem de linhas que a biblioteca produziu naquele
momento, e essa ordem muda entre versoes. O manifesto e o que permite provar que o
Kaggle avaliou os mesmos participantes que ficaram de fora aqui.

## O primeiro resultado

Rodada em 2026-09-01 numa T4 do Kaggle, 44 minutos, 1 epoca, 49 passos:

| | NLL | vs. baseline |
|---|---|---|
| Qwen3-1.7B sem treino | 0,9240 | — |
| Qwen3-1.7B + QLoRA rank 8 | **0,6421** | **−30,5%** |

O Minitaur-8B foi medido na mesma rodada e deu 1,0595, **mas esse numero e invalido**
e foi descartado: ele leu o dataset tokenizado para o Qwen3. Ver "Vocabularios nao se
misturam" abaixo.

Melhorou nos tres dominios, nao so na media: `peterson2021using` −40,7%,
`kool2016when` −28,6% e −27,1%, `badham2017deficits` −19,8%. Se o ganho viesse de um
experimento so, seria o desbalanceamento voltando pela janela.

A avaliacao contou 8.700 tokens contra os 8.704 da preparacao. A diferenca de 4 sao
janelas que comecam exatamente numa escolha: o primeiro token de cada janela e
descartado pelo modelo causal, que nao tem contexto anterior para prever a partir dele.
E o comportamento que `scored_token_count` implementa, e a conta fechar com a explicacao
e o que confirma que o denominador da metrica esta certo.

## A varredura de ablacao

49 passos e quase nenhum treino, entao `configs/sweep/` contem nove configuracoes que
variam **uma coisa por vez**: epocas (3, 5), rank do LoRA (16, 32, 64), learning rate
(1e-4, 2e-4), so as camadas de atencao, e uma combinacao das que funcionarem.

Uma variavel por linha nao e capricho. Numa tabela onde duas coisas mudaram juntas,
nenhuma diferenca pode ser atribuida a nenhuma das duas.

```bash
poetry run poe sweep     # roda tudo que ainda nao tem resultado
poetry run poe report    # tabela e graficos
```

As configuracoes ficam em dois grupos porque uma sessao em lote do Kaggle morre em 12
horas e a varredura inteira passa disso — uma execucao cortada no teto perde o que nao
terminou:

| grupo | rodadas | estimativa |
|---|---|---|
| `configs/sweep/group_a` | `epochs3`, `rank32`, `lr1e4`, `attention_only`, `best_guess` | ~6h |
| `configs/sweep/group_b` | `epochs5`, `rank16`, `rank64`, `lr2e4` | ~6h |
| `configs/sweep/group_c` | `best_guess_v2`, `lr4e4`, `epochs8` | ~5h |

```bash
poetry run python -m centauro_lite sweep --configs configs/sweep/group_a
```

No Kaggle, use **Save Version -> Save & Run All (Commit)**, nunca Run All: a sessao
interativa e derrubada apos 1 hora sem interacao, enquanto a execucao em lote roda no
servidor sem depender do navegador.

Cada rodada acontece num processo separado: carregar e descartar modelos quantizados
repetidamente no mesmo processo fragmenta a VRAM, e numa placa de 16 GB e a quarta
rodada que morre. Rodadas ja medidas sao puladas, entao uma sessao que cai no meio
continua de onde parou.

### Vocabularios nao se misturam

O dataset preparado guarda **ids de token**, e um id so significa alguma coisa dentro de
um vocabulario. Os mesmos ids do nosso dataset, decodificados pelos dois tokenizadores:

| tokenizador | texto |
|---|---|
| Qwen3 (o que gravou) | `You will be shown several examples of geometric objects.` |
| Llama 3.1 (Minitaur) | `askear be.awtnav ptr of Laden_len == guard Am is to.path` |

Um modelo lendo os ids de outro ve ruido, devolve um numero plausivel e **nao levanta
erro**. Foi exatamente o que aconteceu na primeira medicao do Minitaur.

Por isso nao existe um `--model` no `evaluate`. Avaliar outro modelo exige um arquivo de
configuracao proprio, e preparar os dados com ele:

```bash
poetry run python -m centauro_lite prepare  --config configs/minitaur.yaml
poetry run python -m centauro_lite evaluate --config configs/minitaur.yaml --label minitaur-8b
```

Isso produz uma segunda tokenizacao dos **mesmos participantes** — e por isso existem
dois fingerprints:

- **`split_fingerprint`** identifica *quais pessoas* entram, e ignora tokenizador e
  tamanho de janela. Default e Minitaur compartilham `2379f5392193`: mesmos participantes
  de validacao, exatamente.
- **`data_fingerprint`** identifica *como o texto virou tokens*. Default `8c70b49e3cf5`,
  Minitaur `13256cacdbb0` — datasets separados em disco, sem se sobrescrever.

Se os dois mudassem juntos, dois resultados difeririam por dois motivos ao mesmo tempo e
nenhum poderia ser atribuido ao modelo.

Ressalva que fica na discussao do TCC: mesmo feito assim, NLL por token nunca e
perfeitamente comparavel entre tokenizadores, porque cada um corta o texto em pedacos
diferentes. Dar a cada modelo o seu vocabulario e o piso, nao a solucao.

### A protecao do fingerprint

`data_fingerprint` e o hash de tudo que muda o dataset preparado: experimentos, tamanho
da janela, stride, split, seed e tokenizador. Ele fica gravado no manifesto, e `train` e
`evaluate` se **recusam a rodar** se o dataset em disco nao corresponder a configuracao.

A falha que isso evita e silenciosa. Aumente `max_seq_length` num arquivo da varredura e
o dataset em disco continua tokenizado no tamanho antigo: o treino roda, a avaliacao
roda, e o NLL descreve janelas de um tamanho que a configuracao diz que nao e aquele.
Nada mais na pilha percebe.

Pelo mesmo motivo, o relatorio so calcula a melhora contra o baseline **da mesma
configuracao de dados**. Comparar uma rodada de 4096 tokens com um baseline de 2048
creditaria ao fine-tuning um ganho que veio do contexto extra.

## Resultado da ablacao

Medido em 2026-09-02, T4 do Kaggle, mesmos 62 participantes retidos em todas as linhas.

| configuracao | NLL | vs. baseline |
|---|---|---|
| `best_guess` (rank 32, 3 ep, lr 1e-4) | **0,5388** | −41,7% |
| `epochs5` | 0,5517 | −40,3% |
| `epochs3` | 0,5679 | −38,5% |
| `rank64` | 0,5825 | −37,0% |
| `lr2e4` | 0,5836 | −36,8% |
| `rank32` | 0,5892 | −36,2% |
| `lr1e4` | 0,6000 | −35,1% |
| `rank16` | 0,6077 | −34,2% |
| referencia (rank 8, 1 ep, lr 5e-5) | 0,6421 | −30,5% |
| **`minitaur-8b`** | **0,6748** | — |
| `attention_only` | 0,7029 | −23,9% |
| baseline sem ajuste | 0,9240 | — |

**O modelo de 1,7B superou o Minitaur-8B em 20%**, medido pelo mesmo codigo, sobre os
mesmos participantes retidos. O Minitaur e a versao de 8 bilhoes do Centaur, publicada
pelos proprios autores com a mesma receita, e foi treinado nos 160 experimentos — a
comparacao mede especializacao contra escala dentro do dominio, e nao qualidade de
modelo em abstrato.

Ressalva que acompanha a linha do Minitaur: ela carrega outro `data_fingerprint`, porque
foi necessariamente medida sobre a tokenizacao dele. Repare nos 8.701 tokens contra
8.700 — o Llama fatiou uma escolha em um token a mais. NLL por token nao e rigorosamente
comparavel entre tokenizadores; uma diferenca de 20%, no entanto, e grande demais para
ser explicada por isso.

### As tres curvas

Cada eixo foi variado isoladamente, com todo o resto fixo:

| epocas | NLL | | rank | NLL | | learning rate | NLL |
|---|---|---|---|---|---|---|---|
| 1 | 0,6421 | | 8 | 0,6421 | | 5e-5 | 0,6421 |
| 3 | 0,5679 | | 16 | 0,6077 | | 1e-4 | 0,6000 |
| 5 | 0,5517 | | 32 | 0,5892 | | 2e-4 | 0,5836 |
| | | | 64 | 0,5825 | | | |

O **rank saturou**: cada dobra rende metade da anterior (0,034 → 0,018 → 0,007). As
outras duas ainda nao pararam — em particular, previa-se que 2e-4 fosse excessivo e ele
melhorou o resultado, de modo que o teto da taxa de aprendizado permanece por localizar.

Dai o `group_c`: a combinacao das melhores opcoes individuais nunca foi executada,
porque o `best_guess` foi montado antes de os resultados individuais existirem.

## Os quatro pontos de comparacao

O 0,44 do Centaur foi medido nos 160 experimentos completos, com outro tokenizador.
Comparar um NLL medido em tres experimentos com aquele numero e comparar denominadores
diferentes. Por isso a comparacao central e outra:

| Ponto | Como obter | Papel |
|---|---|---|
| Qwen3-1.7B cru | `evaluate` sem adapter | piso — de onde partimos |
| Qwen3-1.7B + QLoRA | `evaluate --adapter ...` | o resultado do trabalho |
| **Minitaur-8B** | `prepare` + `evaluate` com `configs/minitaur.yaml` | **teto e comparacao justa** |
| Centaur 70B = 0,44 | valor fixo do paper | baliza, comparacao declaradamente indireta |

O Minitaur e a versao 8B do Centaur publicada pelos proprios autores, com a mesma
receita. Rodado pelo mesmo codigo, no mesmo split e com a mesma metrica, ele transforma
"chegamos perto de um numero de outro setup" em uma comparacao controlada.

## Estrutura

```
configs/default.yaml       # fonte unica de configuracao
src/centauro_lite/
├── cli.py                 # uma etapa do pipeline por subcomando
├── config.py              # settings de ambiente (.env), validadas no boot
├── core/
│   ├── catalog.py         # EDA: inventario dos experimentos
│   ├── chunking.py        # fatiamento em janelas, com snapping de fronteira
│   ├── masking.py         # localizacao das escolhas humanas <<...>>
│   ├── metrics.py         # agregacao do NLL, ponderada por token
│   ├── reporting.py       # tabela de ablacao e comparabilidade entre rodadas
│   ├── sampling.py        # balanceamento por dominio
│   └── splits.py          # split por participante e manifesto em disco
├── models/
│   └── pipeline_config.py # schema da configuracao do pipeline
├── services/
│   ├── modeling.py        # unsloth: carregar, adaptar, treinar (unica parte que usa GPU)
│   └── notifier.py        # alerta de falha no Telegram
configs/sweep/             # uma configuracao por rodada da ablacao
notebooks/
├── kaggle_train.ipynb     # uma rodada, ponta a ponta
└── kaggle_sweep.ipynb     # a varredura de ablacao
└── utils/logger.py
tests/                     # cobre masking, config e catalogo
data/                      # raw/ interim/ processed/ (fora do git)
```

## Testes

```bash
poetry run pytest
```

**Cobre**: exclusao dos delimitadores `<<`/`>>` da perda, tokens que cruzam a fronteira
de uma escolha, transcricoes malformadas e multi-linha, rejeicao de chave desconhecida
e de stride maior que a janela na configuracao, a aritmetica de janelas do catalogo, a
preservacao de toda escolha no fatiamento (e de nenhuma duas vezes, mesmo com
sobreposicao), o balanceamento entre dominios e entre experimentos do mesmo dominio, e
a ausencia de vazamento de participante entre treino e validacao, e a agregacao do
NLL (ponderacao por token, exclusao da posicao 0 apos o shift causal, e recusa em
reportar 0,0 quando nada foi pontuado).

**Nao cobre**: `services/modeling.py` e o laco de GPU do `evaluate`. Sao encanamento
para o unsloth, sem logica propria, e nao ha GPU no ambiente onde a suite roda. A
aritmetica que eles alimentam esta testada em `core/metrics.py` — foi justamente para
isso que ela vive separada do codigo de GPU.

**Nao foi executado ainda**: nenhuma rodada de treino aconteceu. O codigo de `train` e
`evaluate` foi escrito contra a documentacao do unsloth e nao contra uma GPU real, entao
espere ajustes na primeira execucao.

## Portao de qualidade

Roda sozinho no pre-commit e no CI. Manualmente, um comando so:

```bash
poetry run poe gate
```

Executa `ruff format` → `ruff check` → `mypy src/` → `pytest`, na mesma ordem do CI.
O linter e quem impoe as regras do projeto: `T20` proibe `print()`, `DTZ` proibe
`datetime` ingenuo, `PTH` proibe `os.path`, `S` pega segredo hardcoded.

## Referencias

- [Binz et al. (2025), *A foundation model to predict and capture human cognition*, Nature](https://www.nature.com/articles/s41586-025-09215-4)
- [Dataset Psych-101](https://huggingface.co/datasets/marcelbinz/Psych-101)
- [Llama-3.1-Minitaur-8B](https://huggingface.co/marcelbinz/Llama-3.1-Minitaur-8B) — versao 8B do Centaur, ponto de comparacao central deste trabalho
- [Repositorio do Centaur 70B](https://github.com/marcelbinz/Llama-3.1-Centaur-70B)
