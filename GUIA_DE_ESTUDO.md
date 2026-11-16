# Guia de estudo: Projeto U2 (Ajuste de Curvas)

Este guia é para você **aprender do zero** o que o projeto faz. Ele não é para entregar.

O projeto existe em dois formatos, **com o mesmo código**:
- `projeto_u2.py`: o script, que é o que você vai mostrar ao professor;
- `Projeto_U2_Ajuste_de_Curvas.ipynb`: o notebook, com o mesmo código dividido em células, os textos explicativos, os gráficos já prontos e as conclusões completas.

No `.py`, cada parte começa com um comentário `# ==================== 4.3 ... ====================`, com o mesmo número de seção do notebook. Assim, tudo o que este guia diz sobre a "Seção 4.3" vale para os dois arquivos.

**Ordem de leitura sugerida:**
1. Parte 1 (a história do projeto), 5 minutos.
2. Parte 2 (os conceitos), com calma. É a parte mais importante.
3. Parte 3 (o notebook seção por seção), com o notebook aberto ao lado.
4. Parte 4 (resultados) e Parte 5 (resumo para a apresentação).
5. Depois, o arquivo `PERGUNTAS_DO_PROFESSOR.md`, para treinar as respostas.

---

## Parte 1. A história do projeto, em linguagem simples

Existe um sistema físico (pode ser um motor, um circuito, um tanque...). A gente **mexe nele** com um sinal de entrada `u` e **mede** duas coisas que ele produz, as saídas `x` e `y`. A entrada é uma onda quadrada: fica em +1 por 1 segundo, depois em −1 por 1 segundo, e assim por diante. As medições têm **ruído**: os pontos ficam "espalhados" em volta do valor verdadeiro, como uma balança que oscila.

O professor quer um **modelo matemático**: uma fórmula que, sabendo o que aconteceu antes, diga o que o sistema vai fazer agora. Ele já definiu o formato da fórmula:

```
y[n] = f( y[n−2], y[n−1], u[n], u[n−1] )
```

Em palavras: *o valor da saída agora depende dos dois valores anteriores da saída e dos valores atual e anterior da entrada.* O `n` é o número da amostra (o "instante"). `y[n−1]` é a amostra imediatamente anterior, 0,02 s antes.

O projeto descobre a função `f` de dois jeitos:
- **RBF:** `f` é uma soma de "sininhos" (gaussianas). É uma função flexível, que pode fazer curvas.
- **MMQ:** `f` é uma conta linear simples: `a1·y[n−1] + a2·y[n−2] + b0·u[n] + b1·u[n−1]`.

Depois mede o quanto cada modelo acerta (EMQ e Pearson) e compara os dois.

**Resultado em uma frase:** os dois acertam praticamente igual ao prever um passo à frente. Quando o modelo precisa andar sozinho (simulação livre), a RBF fica um pouco melhor na saída `y`. Na saída `x`, o ruído é tão grande que os dois modelos ficam fracos.

---

## Parte 2. Os conceitos, do zero

### 2.1 Sinal, amostra e ruído
- **Sinal:** uma grandeza que muda com o tempo (a saída `x`, por exemplo).
- **Amostra:** o valor do sinal medido num instante. Aqui há uma amostra a cada 0,02 s, de 0 a 30 s, ou seja, 1501 amostras.
- **Ruído:** a parte aleatória da medição. Ele faz os pontos "pularem" em volta do valor verdadeiro. Nenhum modelo consegue prever o ruído, porque ele é aleatório. Por isso o erro nunca chega a zero.

### 2.2 Modelo e ajuste de curvas
Um **modelo** é uma fórmula com alguns números ajustáveis (os **parâmetros** ou **pesos**). **Ajustar** o modelo é escolher esses números para a fórmula ficar o mais perto possível dos dados. É o mesmo que você viu nas aulas com a reta `Y = β0 + β1·X`: os parâmetros β0 e β1 são escolhidos para a reta passar "pelo meio" dos pontos.

### 2.3 Método dos Mínimos Quadrados (MMQ)
"Passar pelo meio dos pontos" precisa de uma definição exata. O MMQ define assim: **escolha os parâmetros que tornam a soma dos erros ao quadrado a menor possível.** Erro = valor medido − valor do modelo. Elevamos ao quadrado para o erro negativo não cancelar o positivo e para punir mais os erros grandes.

### 2.4 Matriz de projeto e equação normal
Quando o modelo é uma soma de "ingredientes" multiplicados por pesos, dá para escrever tudo de uma vez com matrizes:

```
valores previstos = A · w
```

- `A` é a **matriz de projeto**: uma **linha para cada amostra** e uma **coluna para cada ingrediente** do modelo.
- `w` é o vetor de pesos.

Os pesos que minimizam o erro quadrático são a solução de um sistema linear chamado **equação normal**:

```
(Aᵀ A) · w = Aᵀ · alvo
```

**De onde vem?** O erro quadrático é uma função dos pesos com formato de "tigela" (uma parábola em várias dimensões). O ponto mais baixo da tigela é onde a derivada (o gradiente) vale zero. Derivando e igualando a zero, sai exatamente essa equação.

**Por que `Aᵀ A`?** A matriz `A` tem 1499 linhas e poucas colunas, então `A · w = alvo` tem mais equações que incógnitas e não tem solução exata (sistema **sobredeterminado**). Multiplicando por `Aᵀ`, o sistema vira quadrado (tamanho colunas × colunas) e passa a ter solução: a de menor erro quadrático.

No notebook, a função `equacao_normal(A, alvo)` faz isso com `np.linalg.solve`, que resolve o sistema diretamente, sem calcular a inversa. É mais rápido e mais preciso do que `inv`, e o resultado é o mesmo da fórmula `w = (AᵀA)⁻¹ Aᵀ alvo`.

**Ponto-chave:** o projeto usa a **mesma** função `equacao_normal` para a RBF e para o MMQ. A única diferença entre os dois é o que vai nas colunas da matriz.

### 2.5 Regressores (as entradas do modelo)
**Regressores** são os valores que entram no modelo para ele fazer a previsão. Aqui são 4: `s[n−1]`, `s[n−2]`, `u[n]` e `u[n−1]` (onde `s` é `x` ou `y`). O **alvo** é o valor que queremos prever, `s[n]`. Cada instante `n` vira uma linha: 4 regressores e 1 alvo.

### 2.6 Gaussiana e RBF
Uma **gaussiana** é uma função em forma de sino:

```
φ(z) = exp( −distância(z, c)² / (2σ²) )
```

- Vale **1** quando a entrada `z` está exatamente no **centro** `c`.
- Vai caindo até quase **0** quando `z` se afasta do centro.
- **σ (sigma)** é a **largura** do sino. σ grande = sino largo, que "enxerga" longe. σ pequeno = sino estreito.

Uma **RBF** (Função de Base Radial) soma vários sinos, cada um com seu centro e seu peso:

```
previsão = w0 + w1·φ1(z) + w2·φ2(z) + ... + wM·φM(z)
```

Imagine cada gaussiana como uma **lanterna** apontada para um ponto (o centro). O peso diz quanto aquela lanterna "puxa" a previsão para cima ou para baixo. Onde há várias lanternas, a soma delas desenha uma curva. Onde não há nenhuma lanterna, o modelo não sabe o que fazer.

No nosso projeto, a entrada `z` tem **4 números** (os 4 regressores), então cada centro também tem 4 números, e a distância é calculada nas 4 dimensões: soma das 4 diferenças ao quadrado.

**Por que a RBF usa a mesma equação normal do MMQ?** Porque, depois de escolher centros e σ, cada gaussiana vira só uma **coluna de números** na matriz. O modelo continua sendo "matriz × pesos". Ele é **não linear nas entradas** (por causa do sino), mas **linear nos pesos**. Isso basta para usar a equação normal.

### 2.7 Validação, subajuste e sobreajuste
- **Subajuste (underfitting):** modelo simples demais. Erra nos dados de ajuste e em dados novos.
- **Sobreajuste (overfitting):** modelo flexível demais. Passa a "decorar" o ruído: erra pouco nos dados de ajuste, mas erra mais em dados novos.
- **Validação:** separamos uma parte dos dados que o modelo **não vê** durante o ajuste e medimos o erro nela. É assim que escolhemos as configurações (número de centros e σ) sem cair no sobreajuste.
- **Parcimônia:** entre modelos com desempenho parecido, preferir o mais simples. Essa regra está no material do mini projeto da disciplina: *"escolha o modelo mais simples cujo erro de validação esteja próximo do mínimo"*.

### 2.8 EMQ (Erro Médio Quadrático)
```
EMQ = média de (medido − previsto)²
```
Quanto menor, melhor. A raiz do EMQ (√EMQ) é o "erro típico", na mesma unidade do sinal. Exemplo: EMQ = 0,0075 → √EMQ ≈ 0,087, ou seja, o modelo erra tipicamente uns 0,09 para cima ou para baixo.

### 2.9 Correlação de Pearson (r)
Mede se dois sinais **sobem e descem juntos**. Vai de −1 a +1:
- r = 1: andam juntos perfeitamente (relação linear perfeita);
- r = 0: não têm relação linear;
- r = −1: um sobe quando o outro desce.

**r²** é a fração da variação dos dados que o modelo explica. r = 0,954 → r² ≈ 0,91 → o modelo explica 91% da variação.

**EMQ e r medem coisas diferentes:** o EMQ mede o **tamanho** do erro; o r mede se a **forma** foi capturada. Por isso o enunciado pede os dois.

### 2.10 Regressão linear entre real e ajustado
Fazemos um gráfico com o valor **ajustado** no eixo horizontal e o **medido** no vertical, e passamos uma reta pelos pontos (a mesma regressão linear da aula, com o sistema 2×2). Se o modelo for bom e não tiver tendência de errar sempre para o mesmo lado (**viés**), a reta fica perto de `medido = 0 + 1·ajustado`: **inclinação 1 e intercepto 0**.

### 2.11 Predição um passo à frente × simulação livre
- **Um passo à frente:** para prever `s[n]`, o modelo recebe os valores **medidos** de `s[n−1]` e `s[n−2]`. É como prever o tempo de amanhã sabendo o de hoje.
- **Simulação livre:** o modelo recebe **só a entrada `u`** e usa as **próprias previsões** como "passado". É como prever o tempo do mês inteiro sem olhar pela janela. É bem mais difícil, porque os erros se acumulam. Esse teste mostra se o modelo realmente aprendeu como o sistema funciona, que é a justificativa do projeto ("um modelo que simule a dinâmica do sistema").

---

## Parte 3. O notebook, seção por seção

Abra o `projeto_u2.py` (ou o notebook) e acompanhe. No notebook, cada seção tem um texto explicativo antes do código.

**Mapa de linhas do `projeto_u2.py`**, para apontar na tela enquanto explica:

| Linha | O que é |
|---|---|
| 58 | `emq`: Erro Médio Quadrático |
| 62 | `pearson`: correlação de Pearson |
| 67 | `regressao_linear`: reta pelo sistema 2×2 da aula |
| 74 | `equacao_normal`: resolve (AᵀA)·w = Aᵀ·alvo. **Serve para a RBF e para o MMQ** |
| 84 | `montar_regressores`: as 4 entradas [s[n−1], s[n−2], u[n], u[n−1]] e o alvo s[n] |
| 99–104 | `matriz_phi`: matriz das gaussianas. Linha 102 = distância ao quadrado; linha 103 = fórmula da gaussiana |
| 106 | `escolher_centros`: onde ficam os centros |
| 122 | sorteio da validação (20%, semente fixa) |
| 126–127 | números de centros e valores de σ testados |
| 139–140 | regra da parcimônia (limite de 2% e escolha do mais simples) |
| 164–166 | ajuste final da RBF (centros e pesos) |
| 219 | `grafico_regressao`: regressão, Pearson e EMQ de uma vez |
| 247 | MMQ: `theta_x = equacao_normal(Z_x, alvo_x)` |
| 290 | `simular_livre`: o modelo usa as próprias previsões como passado |
| fim do arquivo | resumo das conclusões (em comentário) |

### Introdução
Diz o objetivo, descreve as colunas do arquivo, mostra o modelo do enunciado e tem uma **tabela que liga cada item da avaliação à seção do notebook**. Use essa tabela na apresentação.

A primeira célula de código importa as bibliotecas (`csv` para ler o arquivo, `numpy` para contas com vetores e matrizes, `matplotlib` para gráficos) e define as cores dos gráficos.

### Seção 1. Leitura dos dados
- O arquivo usa vírgula como separador decimal (`"0,02"`). Por isso cada valor passa por `valor.replace(",", ".")` antes de virar número com `float(...)`.
- `dados[:, 0]` significa "todas as linhas, coluna 0". Assim separamos `t`, `u`, `x` e `y`.
- O gráfico mostra a entrada (onda quadrada) e as duas saídas com ruído. Repare que `x` é muito mais ruidosa que `y`.

### Seção 2. Ferramentas
Quatro funções curtas, escritas à mão (o professor gosta de ver que você não usou função pronta):

| Função | O que faz | Linha principal |
|---|---|---|
| `emq(real, ajustado)` | média dos erros ao quadrado | `np.sum(erro ** 2) / len(erro)` |
| `pearson(a, b)` | fórmula do r | subtrai as médias, multiplica e divide pelas raízes |
| `regressao_linear(xr, yr)` | reta α + β·x pelo sistema 2×2 da aula | `np.linalg.solve(A, B)` |
| `equacao_normal(A, alvo)` | resolve (AᵀA)·w = Aᵀ·alvo | `np.linalg.solve(A.T @ A, A.T @ alvo)` |

O `@` é a multiplicação de matrizes em Python, e `A.T` é a transposta de A.

No fim da célula há uma **conferência**: o Pearson feito à mão é comparado com o do NumPy (`np.corrcoef`). Os dois dão 0,993371, o que mostra que a função está certa.

### Seção 3. Montando os regressores
A função `montar_regressores(s, u)` cria a matriz `Z` (1499 linhas × 4 colunas) e o vetor `alvo`. O truque são as **fatias**:

| Fatia | Significa | Por quê |
|---|---|---|
| `s[2:]` | s[n] (alvo) | do índice 2 até o fim |
| `s[1:-1]` | s[n−1] | um atrás |
| `s[:-2]` | s[n−2] | dois atrás |
| `u[2:]` | u[n] | mesmo instante do alvo |
| `u[1:-1]` | u[n−1] | um atrás |

Todas têm 1499 elementos, então a linha k de cada uma fala do mesmo instante. Começa em n = 2 porque o primeiro alvo precisa de dois valores anteriores. A célula imprime a primeira linha e confere com os valores originais, para você ver que o alinhamento está certo.

### Seção 4. RBF (vale as Etapas 01, 02 e 03)

**4.1 e 4.2: as funções da RBF**

`matriz_phi(Z, centros, sigma)` monta a matriz Φ:
- começa com uma matriz de 1s (`np.ones`): a coluna 0 fica com 1s e serve para o peso w0 (o termo constante);
- para cada centro `j`, calcula a distância ao quadrado de **todas** as linhas até esse centro (`np.sum((Z - centros[j]) ** 2, axis=1)`) e aplica a fórmula da gaussiana. Isso vira a coluna `j + 1`.

`escolher_centros(Z, por_estado, por_troca=3)` decide onde ficam os centros. **Esta é a decisão mais importante do projeto** e a que o professor mais cobra.
1. A entrada `(u[n], u[n−1])` só tem 4 combinações: (+1, +1) e (−1, −1), com a entrada parada (≈ 735 linhas cada), e (+1, −1) e (−1, +1), nos instantes de troca (só 14 e 15 linhas).
2. Para cada combinação, pega as linhas daquela combinação, **ordena pelo valor de s[n−1]** e escolhe linhas **igualmente espaçadas** nessa ordem. Assim os centros vão do menor ao maior valor da saída.
3. Os centros são **linhas dos próprios dados**, então ficam automaticamente onde os dados estão, na mesma escala.
4. Usa `por_estado` centros para cada combinação parada e 3 para cada combinação de troca (há poucas linhas de troca).

**Por que separar por combinação de `u`?** Porque `u` só vale +1 ou −1. Um centro com u = +1 fica a uma distância enorme de qualquer linha com u = −1, e a gaussiana dele vale praticamente zero lá. Se uma combinação ficasse sem centro, o modelo não teria nenhuma "lanterna" para ela.

**4.3: escolha do número de centros e de σ (validação)**
- Sorteia 20% das linhas para validação, com semente fixa (`default_rng(0)`), então o sorteio é sempre o mesmo.
- Testa 4 quantidades de centros (5, 10, 15 e 20 por estado, ou seja, M = 16, 26, 36 e 46) e 6 valores de σ (0,1 a 0,75). Para cada combinação, ajusta no treino e mede o EMQ na validação.
- **Regra da parcimônia:** calcula `limite = 1,02 × menor erro` e escolhe o **primeiro** modelo que fica abaixo desse limite, olhando primeiro o menor número de centros e depois o menor σ. É isso que `np.argwhere(tabela <= limite)[0]` faz: `argwhere` lista as posições na ordem (linha 0 primeiro, e dentro dela coluna 0 primeiro), e a linha 0 é a de menos centros.
- **Resultado:** x → 16 centros e σ = 0,2; y → 16 centros e σ = 0,5.

**Por que preferir σ pequeno?** Com σ grande, todas as gaussianas ficam parecidas, as colunas de Φ ficam quase iguais, a matriz ΦᵀΦ fica quase impossível de resolver com precisão (**mal condicionada**) e os pesos ficam enormes e instáveis.

**Por que y precisou de σ maior que x?** y varia numa faixa maior (−0,78 a 0,69, contra −0,42 a 0,41 de x). Com só 5 centros por estado espalhados nessa faixa, os centros de y ficam mais distantes entre si, e o sino precisa ser mais largo para os vizinhos se sobreporem. É a regra prática do material da disciplina: σ da ordem do espaçamento entre centros.

**4.4: ajuste final e parâmetros livres**
- Com as configurações escolhidas, escolhe os centros usando **todas** as linhas e calcula os pesos pela equação normal. A validação só serviu para escolher as configurações; para os pesos finais, usar todos os dados dá a estimativa mais precisa.
- Imprime: número de gaussianas (16), σ, número de pesos (17 = w0 + 16), tamanho da matriz Φ (1499 × 17), w0 e alguns centros com seus pesos. **Esses são os parâmetros livres do modelo RBF** que o enunciado pede.
- O gráfico mostra os centros (triângulos) em cima da nuvem de dados, no plano (s[n−1], s[n−2]). É a resposta visual para "onde estão os centros?".

**4.5: Etapas 01 e 02**
Gráfico do medido (pontos) com o ajustado pela RBF (linha), e o EMQ de cada saída. A linha é serrilhada porque recebe os valores medidos (com ruído) como entrada.

**4.6: Etapa 03**
A função `grafico_regressao` faz as três coisas de uma vez: regressão linear medido × ajustado, Pearson e EMQ, e desenha o gráfico com a faixa ±√EMQ. O resultado da RBF é inclinação 1 e intercepto 0 exatos. Isso não é coincidência: acontece sempre que o modelo tem termo constante (w0) e foi ajustado pelo MMQ.

### Seção 5. MMQ (vale a Etapa 04)
- O modelo é `s[n] = a1·s[n−1] + a2·s[n−2] + b0·u[n] + b1·u[n−1]`. A matriz de projeto é a própria `Z`: os 4 regressores entram **direto** nas colunas, sem gaussianas.
- `theta_x = equacao_normal(Z_x, alvo_x)` calcula os 4 parâmetros. **Esses são os parâmetros livres do MMQ.**
- `a1` e `a2` medem quanto o passado da saída influencia o presente (a "memória" do sistema). `b0` e `b1` medem o efeito da entrada.
- Depois repete os gráficos, o EMQ, a regressão e o Pearson, igual à RBF.
- A inclinação e o intercepto saem só **quase** 1 e **quase** 0, porque o modelo do enunciado não tem termo constante.

### Seção 6. Simulação livre
A função `simular_livre(prever, u, s_medido)`:
1. copia só as duas primeiras amostras medidas;
2. daí em diante, para cada n, monta `z = [s_sim[n−1], s_sim[n−2], u[n], u[n−1]]` usando as **próprias previsões** e calcula `s_sim[n]`.

`prever_rbf` e `prever_mmq` criam a função de previsão de cada modelo, para a mesma `simular_livre` servir aos dois. No gráfico:
- **y:** os dois modelos reproduzem bem a forma da resposta, e a RBF acerta um pouco melhor os picos;
- **x:** os dois viram praticamente uma onda quadrada pequena. Os modelos não aprenderam a forma da resposta de x.

### Seção 7. Tabela comparativa
Junta todos os números numa tabela.

### Seção 8. Conclusões
São 7 conclusões. Leia todas e entenda cada uma. Elas valem 2 pontos.

---

## Parte 4. Resultados (os números que você precisa saber)

| Modelo | Saída | EMQ | Pearson r |
|---|---|---|---|
| RBF, um passo à frente | x | 0,0122 | 0,637 |
| RBF, um passo à frente | y | 0,0075 | 0,954 |
| MMQ, um passo à frente | x | 0,0123 | 0,636 |
| MMQ, um passo à frente | y | 0,0075 | 0,954 |
| RBF, simulação livre | x | 0,0150 | 0,528 |
| RBF, simulação livre | y | 0,0185 | 0,914 |
| MMQ, simulação livre | x | 0,0150 | 0,524 |
| MMQ, simulação livre | y | 0,0197 | 0,878 |

**RBF:** 16 gaussianas por saída (5 + 5 para a entrada parada, 3 + 3 para as trocas), σ = 0,2 em x e 0,5 em y, 17 pesos.

**MMQ:**

| Saída | a1 | a2 | b0 | b1 |
|---|---|---|---|---|
| x | 0,2984 | 0,2118 | −0,0216 | −0,0150 |
| y | 0,4876 | 0,4649 | −0,0159 | 0,0426 |

**Como ler isso:**
1. Um passo à frente: **empate**. A relação é praticamente linear, então as gaussianas não acrescentam nada.
2. Simulação livre em y: **RBF um pouco melhor** (erro menor e r maior).
3. x é sempre pior que y por causa do **ruído**, que em x é quase do tamanho do sinal.
4. Em x, a memória estimada é baixa (a1 + a2 = 0,51) porque o passado medido tem muito ruído. Por isso a simulação de x vira uma onda quadrada.

---

## Parte 5. Resumo de 1 minuto para abrir a apresentação

> "O sistema tem uma entrada `u`, que é uma onda quadrada, e duas saídas medidas com ruído, `x` e `y`. O enunciado pede o modelo `s[n] = f(s[n−1], s[n−2], u[n], u[n−1])`. Fiz esse modelo de duas formas.
> Com **RBF**, f é uma soma de 16 gaussianas. Cada centro é um ponto de 4 dimensões tirado dos próprios dados, e escolhi o número de centros e o σ por validação.
> Com **MMQ**, f é linear, com 4 parâmetros.
> Nos dois casos os pesos saem da mesma equação normal. Avaliei com EMQ, regressão linear entre medido e ajustado, e Pearson.
> Prevendo um passo à frente, os dois empatam, o que mostra que a relação é praticamente linear. Na simulação livre, em que o modelo só recebe a entrada, a RBF ficou um pouco melhor em y. Em x o ruído é tão grande que os dois ficam fracos."

---

## Parte 6. Como rodar

1. Deixe `projeto_u2.py`, `Projeto_U2_Ajuste_de_Curvas.ipynb` e `dados_projeto_02.csv` na mesma pasta (já estão).
2. **Script:** rode `python projeto_u2.py` nessa pasta. Os resultados aparecem no terminal e os gráficos abrem **um de cada vez**: feche a janela de um gráfico para o programa seguir para o próximo. O programa inteiro leva poucos segundos (sem contar o tempo com as janelas abertas).
3. **Notebook:** abra no Jupyter ou no VS Code e use **"Run All"** (executar tudo).
4. Precisa de Python com `numpy` e `matplotlib` (instale com `pip install numpy matplotlib`). Em qualquer computador os números saem iguais, porque o sorteio da validação usa semente fixa.
