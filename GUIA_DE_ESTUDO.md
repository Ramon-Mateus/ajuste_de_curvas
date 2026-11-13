# Guia de estudo: Projeto U2 (Ajuste de Curvas)

Este guia é **para você estudar**, não para entregar. Ele explica o raciocínio por trás do notebook `Projeto_U2_Ajuste_de_Curvas.ipynb` e traz as perguntas que o professor provavelmente vai fazer, já com as respostas.

---

## 1. O projeto em 30 segundos (para abrir a apresentação)

> "Recebemos uma entrada `u` (onda quadrada) e duas saídas medidas com ruído, `x` e `y`, de um sistema físico. Fiz dois modelos.
> **(1) RBF:** uma soma de gaussianas espalhadas no tempo, que filtra o ruído e devolve a curva suave do sinal.
> **(2) MMQ:** o modelo dinâmico do enunciado, `y[n] = a1·y[n-1] + a2·y[n-2] + b0·u[n] + b1·u[n-1]`, com 4 parâmetros calculados pela equação normal.
> Avaliei os dois com EMQ, regressão linear (medido × ajustado) e correlação de Pearson. A RBF teve o melhor ajuste. O MMQ chegou perto com só 4 parâmetros, mas o ruído nos dados enviesa os parâmetros dele."

## 2. Resultados principais (decore estes números)

| Modelo | Saída | EMQ | Pearson r |
|---|---|---|---|
| RBF (M = 40) | x | 0,0097 | 0,725 |
| RBF (M = 50) | y | 0,0053 | 0,968 |
| MMQ (um passo à frente) | x | 0,0123 | 0,636 |
| MMQ (um passo à frente) | y | 0,0075 | 0,954 |
| MMQ (simulação livre) | x | 0,0150 | 0,524 |
| MMQ (simulação livre) | y | 0,0197 | 0,878 |

Parâmetros do MMQ: x → a1 = 0,298, a2 = 0,212, b0 = −0,022, b1 = −0,015. y → a1 = 0,488, a2 = 0,465, b0 = −0,016, b1 = 0,043.

## 3. Mapa do código (o que cada bloco faz)

| Bloco do notebook | O que faz | Função/linha-chave |
|---|---|---|
| Leitura | Lê o CSV, troca vírgula decimal por ponto | `float(valor.replace(",", "."))` |
| Ferramentas | EMQ, Pearson e reta de regressão, tudo pela fórmula | `emq`, `pearson`, `regressao_linear` |
| RBF: matriz | Monta Φ: coluna de 1s + uma gaussiana por centro | `matriz_phi` |
| RBF: pesos | Resolve a equação normal (Φᵀ Φ) w = Φᵀ s | `np.linalg.solve(Phi.T @ Phi, Phi.T @ sinal)` |
| RBF: escolha de M | Esconde 1 de cada 5 pontos e testa M de 20 a 200 | `validacao = (np.arange(N) % 5 == 2)` |
| MMQ: regressores | Cada linha = [s[n-1], s[n-2], u[n], u[n-1]] | `montar_regressores` |
| MMQ: parâmetros | Resolve (Aᵀ A) θ = Aᵀ b | `ajustar_mmq` |
| Simulação livre | O modelo usa as próprias previsões como passado | `simular_livre` |

**Truque de fatiamento** (pergunta comum): para `n = 2 … N-1`,
- `s[2:]` é s[n] (alvo)
- `s[1:-1]` é s[n-1]
- `s[:-2]` é s[n-2]

Todos têm o mesmo tamanho (N − 2 = 1499), então cada linha fica alinhada no mesmo instante n.

## 4. Perguntas prováveis do professor e respostas

**P: O que é uma função de base radial?**
É uma função que só depende da distância a um centro. A gaussiana `φ(t) = exp(−(t−c)²/(2σ²))` vale 1 no centro `c` e cai para 0 longe dele. Somando várias, cada uma com um peso, formamos qualquer curva suave.

**P: Por que o modelo RBF é "linear" se a gaussiana não é?**
Ele é não linear em `t`, mas **linear nos pesos `w`**: a saída é `Φ·w`. Por isso dá para achar os pesos com a mesma equação normal do MMQ, sem iterações.

**P: Quais são os parâmetros livres da RBF?**
O número de gaussianas M, os centros c_j, a largura σ e os pesos w_0…w_M. Centros e σ eu escolhi (centros igualmente espaçados; σ = espaçamento entre centros). M veio da validação. Os pesos vieram da equação normal.

**P: Por que σ igual ao espaçamento entre centros?**
É a regra prática da aula. Com σ muito pequeno, as gaussianas não se sobrepõem e a curva fica "ondulada" entre centros. Com σ muito grande, todas ficam parecidas e Φᵀ Φ fica quase singular (pesos enormes e instáveis). Com σ ≈ espaçamento, os sinos vizinhos se sobrepõem na medida certa.

**P: Como você escolheu M = 40 e M = 50?**
Por validação: escondi 1 de cada 5 amostras, ajustei com o resto e medi o erro nas escondidas. Com poucas gaussianas o erro é alto (subajuste). Com muitas, ele volta a subir (sobreajuste, a curva copia o ruído). Como a curva ficou quase plana perto do mínimo, peguei o **menor M** com erro até 2% acima do mínimo (regra da parcimônia).

**P: Por que não usar 200 gaussianas, se o erro de treino fica menor?**
Porque erro de treino menor não quer dizer modelo melhor. Com 200 gaussianas a curva passa a seguir o ruído (dá para ver o x(t) ondulado). O erro de validação, que mede o desempenho em dados não vistos, piora.

**P: Deduza a equação normal.**
Queremos minimizar J(w) = (1/N)·‖Φw − s‖². O gradiente é ∇J = (2/N)·Φᵀ(Φw − s). Igualando a zero: Φᵀ Φ w = Φᵀ s. Como J é uma parábola (quadrática convexa), esse ponto é o mínimo global.

**P: Por que `np.linalg.solve` e não `np.linalg.inv`?**
`solve` resolve o sistema diretamente (eliminação/fatoração LU). É mais rápido e mais preciso numericamente do que calcular a inversa e depois multiplicar. O resultado é o mesmo da fórmula w = (Φᵀ Φ)⁻¹ Φᵀ s.

**P: O que é o EMQ e qual o valor "bom"?**
É a média dos erros ao quadrado. Como os dados têm ruído, o melhor EMQ possível é mais ou menos a **variância do ruído**. Nossa RBF ficou nesse nível (√EMQ ≈ 0,10 em x e 0,07 em y), então ela removeu o ruído sem copiá-lo.

**P: O que significa o r de Pearson?**
Mede o quanto dois sinais variam juntos de forma linear (de −1 a 1). r² é a fração da variação dos dados explicada pelo modelo: 0,968² ≈ 0,94, ou seja, o ajuste RBF explica 94% da variação de y.

**P: Por que a regressão deu exatamente inclinação 1 e intercepto 0 na RBF?**
É uma propriedade do MMQ com termo constante: o erro (resíduo) fica ortogonal à coluna de 1s e à curva ajustada. Isso força a melhor reta entre medido e ajustado a ser `medido = 0 + 1·ajustado`. Significa que o ajuste **não tem viés**. No MMQ dinâmico o modelo não tem termo constante, por isso deu "quase" 1 (0,9995) e "quase" 0 (0,0022).

**P: Para que serve a faixa ±√EMQ no gráfico de regressão?**
Mostra o erro típico em torno da reta. A maioria dos pontos deve ficar dentro dela. Em x a faixa é larga (muito ruído); em y é estreita.

**P: Por que x tem r tão menor que y?**
Por causa da relação sinal/ruído. Em x o ruído (≈ 0,10) é do tamanho do próprio sinal (≈ 0,1 a 0,25); em y o sinal (≈ 0,4 a 0,7) é bem maior que o ruído (≈ 0,07). Mesmo a curva verdadeira de x teria correlação moderada com os pontos medidos. Não é falha do método.

**P: O que é o modelo do MMQ fisicamente?**
Uma equação de diferenças de 2ª ordem (modelo ARX). É a versão discreta de uma equação diferencial de 2ª ordem, como um massa-mola-amortecedor ou um circuito RLC. a1 e a2 representam a "memória/inércia" (como a saída depende do próprio passado); b0 e b1 representam o efeito da entrada.

**P: Qual a diferença entre predição um passo à frente e simulação livre?**
Na predição um passo à frente, o modelo recebe os valores **medidos** de y[n-1] e y[n-2] e só prevê o próximo. Na simulação livre, ele recebe só a entrada u e usa as **próprias previsões** como passado. A simulação livre é o teste mais difícil, porque os erros se acumulam.

**P: Por que a curva do MMQ ficou serrilhada e a da RBF suave?**
Porque o MMQ usa os valores medidos (com ruído) de y[n-1] e y[n-2] como entrada; parte do ruído passa para a previsão. A RBF é uma função suave do tempo, então não consegue acompanhar o ruído.

**P: Por que a simulação livre de x ficou ruim (parece uma onda quadrada)?**
Porque os parâmetros a1 e a2 estimados ficaram enviesados (menores do que deveriam). Quando os regressores (y[n-1], y[n-2]) têm ruído, o MMQ "desconfia" deles e diminui o peso do passado. Sem "memória" suficiente, o modelo só copia a entrada (por isso a onda quadrada). Para um sinal suave amostrado a cada 0,02 s, esperaríamos a1 ≈ 2 e a2 ≈ −1, porque s[n] − 2s[n−1] + s[n−2] ≈ Δt²·s″ é quase zero.

**P: Então a RBF é melhor que o MMQ?**
Em erro, sim. Mas elas fazem coisas diferentes. A RBF tem 41–51 parâmetros e **descreve** estes 30 s; ela não sabe o que aconteceria com outra entrada. O modelo MMQ tem 4 parâmetros e **relaciona entrada e saída**; ele pode prever a resposta a uma entrada nova. Mais parâmetros dão mais flexibilidade, mas não explicam o sistema.

**P: Como melhoraria o modelo dinâmico?**
Identificar o modelo a partir do sinal já filtrado pela RBF, testar ordens maiores (mais atrasos), ou usar métodos próprios para ruído nos regressores (por exemplo, variáveis instrumentais). Outra opção é usar RBF no espaço dos regressores (modelo NARX não linear).

**P: Por que não usou funções complexas (Fourier)?**
O enunciado pede RBF **ou** funções complexas. Escolhi RBF porque é o método trabalhado nos exemplos da unidade (Exemplo_RBF e o mini projeto da turbina) e porque se adapta bem ao transitório do início do sinal (os primeiros segundos de x), que uma série de Fourier periódica representaria pior.

**P: Por que separou treino e validação só para escolher M e depois ajustou com todos os dados?**
A validação serve só para escolher o hiperparâmetro M. Depois de escolhido, usar todos os dados dá a estimativa mais precisa dos pesos.

## 5. Conceitos que você precisa saber dizer em uma frase

- **Subajuste (underfitting):** modelo simples demais; erra no treino e na validação.
- **Sobreajuste (overfitting):** modelo flexível demais; acerta o treino (copia o ruído) e erra na validação.
- **Matriz de projeto (Φ ou A):** cada linha é uma amostra, cada coluna é uma função de base/regressor.
- **Sistema sobredeterminado:** mais equações (1499) que incógnitas (4); não tem solução exata, o MMQ acha a que minimiza o erro quadrático.
- **Número de condição de Φᵀ Φ:** mede o quanto o sistema é sensível a erros; muito grande = pesos instáveis.

## 6. Como rodar

1. Deixe `Projeto_U2_Ajuste_de_Curvas.ipynb` e `dados_projeto_02.csv` na mesma pasta (já estão).
2. Abra o notebook no Jupyter/VS Code e use "Run All".
3. Precisa apenas de `numpy` e `matplotlib`. Não há aleatoriedade, então os números saem sempre iguais.

**Versão script (`projeto_u2.py`):** é o mesmo código do notebook, só sem as explicações (elas ficam no notebook e neste guia). Na pasta do projeto, rode `python projeto_u2.py`. Os resultados aparecem no terminal e os gráficos abrem um de cada vez: feche a janela de um gráfico para o programa seguir para o próximo. Os comentários `# ==== ... ====` marcam as mesmas seções do notebook (Etapa 01, 02, 03, 04...), para você achar cada parte rápido.
