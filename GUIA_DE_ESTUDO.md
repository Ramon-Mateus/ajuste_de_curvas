# Guia de estudo: Projeto U2 (Ajuste de Curvas)

Este guia é **para você estudar**, não para entregar. Ele explica o raciocínio por trás do notebook `Projeto_U2_Ajuste_de_Curvas.ipynb` (e do script `projeto_u2.py`, que tem o mesmo código) e traz as perguntas que o professor provavelmente vai fazer, já com as respostas. As perguntas específicas sobre gaussianas, centros e σ estão no `PERGUNTAS_SOBRE_RBF.md`.

---

## 1. O projeto em 30 segundos (para abrir a apresentação)

> "O sistema tem uma entrada `u` (onda quadrada) e duas saídas medidas com ruído, `x` e `y`. O enunciado pede o modelo `s[n] = f(s[n−1], s[n−2], u[n], u[n−1])`. Fiz esse modelo de duas formas:
> **(1) RBF:** f é uma soma de 26 gaussianas, cada uma com centro num ponto de 4 dimensões tirado dos próprios dados.
> **(2) MMQ:** f é linear, `s[n] = a1·s[n−1] + a2·s[n−2] + b0·u[n] + b1·u[n−1]`.
> Os pesos dos dois saem da mesma equação normal. Avaliei com EMQ, regressão linear (medido × ajustado) e Pearson. Na predição um passo à frente os dois empatam, o que mostra que a relação é praticamente linear. Na simulação livre, em que o modelo só recebe a entrada, a RBF foi melhor em y (erro 28% menor)."

Antes disso, como preparação, mostro a RBF com uma entrada só (o tempo): ela filtra o ruído e serve de referência visual.

## 2. Estrutura do notebook

| Seção | O que faz | Item da avaliação |
|---|---|---|
| 1 | Lê o CSV e mostra os sinais | — |
| 2 | Funções: EMQ, Pearson, regressão, matriz RBF, equação normal | — |
| 3 | **Preparação:** RBF com 1 entrada (tempo) = filtro e referência | — |
| 4 | **RBF dinâmica** com 4 entradas (fórmula do enunciado) | Etapas 01, 02, 03 (RBF, EMQ, regressão, Pearson) |
| 5 | **MMQ** (modelo linear, mesma fórmula) | Etapa 04 (repetir com MMQ) |
| 6 | Simulação livre dos dois modelos | complemento das conclusões |
| 7 | Tabela comparativa | — |
| 8 | Conclusões | Conclusões |

## 3. Resultados principais (decore estes números)

| Modelo | Saída | EMQ | Pearson r |
|---|---|---|---|
| RBF dinâmica (um passo à frente) | x | 0,0120 | 0,648 |
| RBF dinâmica (um passo à frente) | y | 0,0076 | 0,954 |
| MMQ (um passo à frente) | x | 0,0123 | 0,636 |
| MMQ (um passo à frente) | y | 0,0075 | 0,954 |
| RBF dinâmica (simulação livre) | x | 0,0149 | 0,528 |
| RBF dinâmica (simulação livre) | y | 0,0141 | 0,920 |
| MMQ (simulação livre) | x | 0,0150 | 0,524 |
| MMQ (simulação livre) | y | 0,0197 | 0,878 |
| Referência: RBF no tempo (filtro) | x | 0,0098 | 0,725 |
| Referência: RBF no tempo (filtro) | y | 0,0054 | 0,968 |

- **RBF dinâmica:** 26 centros por saída, σ = 0,2 (x) e σ = 0,3 (y), 27 pesos (w₀ + 26).
- **MMQ:** x → a1 = 0,298, a2 = 0,212, b0 = −0,022, b1 = −0,015. y → a1 = 0,488, a2 = 0,465, b0 = −0,016, b1 = 0,043.
- **RBF no tempo:** 40 gaussianas (x) e 50 (y), centros de 0 a 30 s.

## 4. Mapa do código (`projeto_u2.py`)

| Linha(s) | O que faz |
|---|---|
| 72–77 | `matriz_phi`: monta a matriz das gaussianas. **A mesma função serve para 1 entrada (tempo) e 4 entradas.** Linha 75 = distância ao quadrado; linha 76 = gaussiana |
| 79–80 | `equacao_normal`: resolve (Aᵀ A) w = Aᵀ alvo. **A mesma função calcula os pesos da RBF no tempo, da RBF dinâmica e do MMQ** |
| 87–94 | RBF no tempo: centros de 0 a 30 s (linha 88), σ = espaçamento (linha 89) |
| 96–115 | Escolha do número de gaussianas da RBF no tempo (validação 1 de cada 5) |
| 151–154 | `montar_regressores`: cada linha z[n] = [s[n−1], s[n−2], u[n], u[n−1]] e alvo s[n] |
| 156–164 | `escolher_centros`: os 26 centros da RBF dinâmica |
| 177–192 | Escolha de σ da RBF dinâmica (validação com 20% sorteado) |
| 203–210 | Ajuste final da RBF dinâmica (centros, pesos, previsão) |
| 238–257 | Etapas 01 e 02 (gráfico e EMQ da RBF) |
| 261–284 | Etapa 03 (regressão e Pearson) |
| 288 | MMQ: `theta_x = equacao_normal(Z_x, alvo_x)` |
| 330–336 | `simular_livre`: o modelo usa as próprias previsões como passado |

**Truque de fatiamento** (pergunta comum): para `n = 2 … N−1`,
- `s[2:]` é s[n] (alvo)
- `s[1:-1]` é s[n−1]
- `s[:-2]` é s[n−2]

Todos têm 1499 elementos, então cada linha fica alinhada no mesmo instante n.

## 5. Perguntas prováveis do professor e respostas

### Sobre o modelo e o enunciado

**P: Qual modelo você implementou?**
O do enunciado: s[n] = f(s[n−1], s[n−2], u[n], u[n−1]), para x e para y. Com RBF, f é uma soma de gaussianas; com MMQ, f é linear.

**P: Para que serve a Seção 3 (RBF no tempo)?**
É uma preparação. Mostra a RBF no caso mais simples, com uma entrada, onde dá para desenhar os centros e a soma dos sinos. Ela também filtra o ruído e dá uma curva de referência, a linha preta tracejada dos gráficos. Ela não é o modelo dinâmico, porque não usa a entrada u e só descreve estes 30 s.

**P: O que é o modelo do MMQ fisicamente?**
Uma equação de diferenças de 2ª ordem (modelo ARX). É a versão discreta de uma equação diferencial de 2ª ordem, como um massa-mola-amortecedor ou um circuito RLC. a1 e a2 são a "memória/inércia" e b0, b1 o efeito da entrada.

### Sobre a matemática

**P: Por que o modelo RBF é "linear" se a gaussiana não é?**
Ele é não linear na entrada, mas **linear nos pesos w**: a saída é Φ·w. Por isso os pesos saem da equação normal, sem iterações, igual ao MMQ.

**P: Deduza a equação normal.**
Queremos minimizar J(w) = (1/N)·‖Φw − s‖². O gradiente é ∇J = (2/N)·Φᵀ(Φw − s). Igualando a zero: Φᵀ Φ w = Φᵀ s. Como J é uma parábola (quadrática convexa), esse ponto é o mínimo.

**P: Qual a diferença entre a matriz da RBF e a do MMQ?**
No MMQ, a matriz tem os 4 regressores direto nas colunas (1499 × 4). Na RBF, cada coluna é uma gaussiana aplicada aos 4 regressores, mais uma coluna de 1s (1499 × 27). A equação normal é a mesma.

**P: Por que `np.linalg.solve` e não `np.linalg.inv`?**
`solve` resolve o sistema direto. É mais rápido e mais preciso do que calcular a inversa e multiplicar. O resultado é o mesmo da fórmula w = (Φᵀ Φ)⁻¹ Φᵀ s.

**P: Por que `solve` funciona se a matriz é 1499 × 27 (não quadrada)?**
Porque não resolvo Φ w = s, que não tem solução exata (há mais equações que incógnitas). Resolvo Φᵀ Φ w = Φᵀ s, e Φᵀ Φ é quadrada (27 × 27).

### Sobre as métricas

**P: O que é o EMQ e qual o valor "bom"?**
É a média dos erros ao quadrado. Com dados ruidosos, o melhor possível é mais ou menos a **variância do ruído**. A RBF no tempo mostra esse limite (≈ 0,0098 em x e 0,0054 em y). Os modelos dinâmicos ficam um pouco acima porque usam valores passados ruidosos como entrada.

**P: O que significa o r de Pearson?**
Mede o quanto dois sinais variam juntos de forma linear (−1 a 1). r² é a fração da variação explicada: 0,954² ≈ 0,91, ou seja, a RBF explica 91% da variação de y.

**P: Por que a regressão deu inclinação 1 e intercepto 0 exatos na RBF?**
É uma propriedade do MMQ com termo constante (w₀): o erro fica ortogonal à coluna de 1s e à curva ajustada, então a melhor reta medido × ajustado é `0 + 1·ajustado`. O ajuste não tem viés. O modelo MMQ do enunciado não tem termo constante, por isso deu "quase" 1 e "quase" 0.

**P: Para que serve a faixa ±√EMQ?**
Mostra o erro típico em torno da reta. A maioria dos pontos fica dentro dela.

**P: Por que x tem r tão menor que y?**
Relação sinal/ruído. Em x, o ruído (≈ 0,11) é do tamanho do sinal (≈ 0,1 a 0,25). Em y, o sinal (≈ 0,4 a 0,7) é bem maior que o ruído. Não é falha do método.

### Sobre os resultados

**P: Por que a RBF não ganhou do MMQ na predição um passo à frente?**
Porque, com esse ruído, a relação entre o passado e o próximo valor é praticamente linear. As gaussianas dão flexibilidade para curvas, mas não há curvatura relevante para capturar. Na validação também empatam (x: 0,0131 contra 0,0133; y: 0,0066 contra 0,0067). Conclusão: para esse uso, o modelo linear com 4 parâmetros basta.

**P: Então para que serve a RBF?**
Na simulação livre de y ela foi melhor: EMQ 0,0141 contra 0,0197 (28% menor) e r 0,920 contra 0,878. Ela reproduz melhor a forma dos picos. Ou seja, a parte não linear importa quando o modelo precisa andar sozinho.

**P: Qual a diferença entre predição um passo à frente e simulação livre?**
Na predição um passo à frente, o modelo recebe os valores **medidos** de s[n−1] e s[n−2] e só prevê o próximo. Na simulação livre, recebe só a entrada u e usa as **próprias previsões** como passado. A simulação livre é mais difícil, porque os erros se acumulam.

**P: Por que as curvas um passo à frente são serrilhadas?**
Porque elas usam os valores medidos (com ruído) de s[n−1] e s[n−2] como entrada; parte do ruído passa para a previsão.

**P: Por que a simulação de x vira uma onda quadrada?**
Porque os parâmetros ficaram enviesados pelo ruído nos regressores: o ajuste "desconfia" do passado ruidoso e diminui a memória (a1 = 0,30 e a2 = 0,21, quando o esperado para um sinal suave seria a1 ≈ 2 e a2 ≈ −1, porque s[n] − 2s[n−1] + s[n−2] ≈ Δt²·s″ ≈ 0). Sem memória, o modelo só segue a entrada. A RBF sofre o mesmo efeito, porque também foi ajustada com regressores ruidosos.

**P: Como melhoraria?**
Ajustar os modelos a partir do sinal filtrado, aumentar a ordem (mais atrasos), ou usar métodos próprios para ruído nos regressores (variáveis instrumentais).

**P: Por que não usou funções complexas (Fourier)?**
O enunciado pede RBF **ou** funções complexas. Escolhi RBF porque é o método dos exemplos da unidade (Exemplo_RBF e o mini projeto da turbina).

**P: Por que separou validação e depois ajustou com todos os dados?**
A validação serve só para escolher os hiperparâmetros (M, σ). Depois de escolhidos, usar todos os dados dá a estimativa mais precisa dos pesos.

## 6. Conceitos que você precisa saber dizer em uma frase

- **Regressores:** as entradas do modelo, aqui s[n−1], s[n−2], u[n], u[n−1].
- **Subajuste (underfitting):** modelo simples demais; erra no treino e na validação.
- **Sobreajuste (overfitting):** modelo flexível demais; acerta o treino (copia o ruído) e erra na validação.
- **Matriz de projeto (Φ ou Z):** cada linha é uma amostra, cada coluna é uma função de base/regressor.
- **Sistema sobredeterminado:** mais equações (1499) que incógnitas; o MMQ acha a solução que minimiza o erro quadrático.
- **Número de condição de Φᵀ Φ:** mede a sensibilidade do sistema; muito grande = pesos instáveis.
- **Parcimônia:** entre modelos com desempenho parecido, escolher o mais simples.

## 7. Como rodar

1. Deixe `Projeto_U2_Ajuste_de_Curvas.ipynb`, `projeto_u2.py` e `dados_projeto_02.csv` na mesma pasta (já estão).
2. **Notebook:** abra no Jupyter/VS Code e use "Run All".
3. **Script:** rode `python projeto_u2.py`. Os resultados aparecem no terminal e os gráficos abrem um de cada vez: feche a janela de um gráfico para o programa seguir para o próximo. Os comentários `# ==== ... ====` marcam as mesmas seções do notebook.
4. Precisa apenas de `numpy` e `matplotlib`. A validação usa sorteio com semente fixa, então os números saem sempre iguais.
