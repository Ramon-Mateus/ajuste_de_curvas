# Perguntas prováveis do professor (e como responder)

Leia o `GUIA_DE_ESTUDO.md` antes deste arquivo: aqui os conceitos aparecem sem explicação do zero.

**Dica de postura:** responda **primeiro com o número**, depois explique o porquê. Exemplo: "16 gaussianas por saída. Testei 16, 26, 36 e 46 na validação e escolhi a mais simples com erro próximo do mínimo."

Algumas respostas trazem um **trecho de código para rodar na hora**. Use de um destes jeitos:
- **no `projeto_u2.py`:** cole no **fim** do arquivo e rode de novo (feche as janelas dos gráficos até chegar ao final);
- **no notebook:** cole numa célula nova no **fim**, depois de rodar tudo ("Run All"). Todos os números deste arquivo foram calculados com o seu código e os seus dados.

---

## 1. Sobre o enunciado e o modelo

**"Qual modelo você implementou?"**
> "O do enunciado: s[n] = f(s[n−1], s[n−2], u[n], u[n−1]), para x e para y. Na RBF, f é uma soma de gaussianas; no MMQ, f é linear: a1·s[n−1] + a2·s[n−2] + b0·u[n] + b1·u[n−1]."

**"Onde está cada item da avaliação?"**
> Mostre a tabela da introdução do notebook: RBF na Seção 4 (Etapas 01 a 03), MMQ na Seção 5 (Etapa 04), conclusões na Seção 8.

**"Quais são os parâmetros livres?"**
> "No MMQ, a1, a2, b0 e b1 (Seção 5.1). Na RBF, os 17 pesos w0 a w16, calculados pela equação normal, e também os centros e o σ, escolhidos por validação (Seção 4.4)."

**"O que o modelo MMQ representa fisicamente?"**
> "É uma equação de diferenças de 2ª ordem, a versão discreta de uma equação diferencial de 2ª ordem, como um massa-mola-amortecedor. a1 e a2 são a memória do sistema, e b0 e b1 o efeito da entrada."

**"Por que RBF e não funções complexas?"**
> "O enunciado pede RBF **ou** funções complexas. Escolhi RBF porque é o método dos exemplos da unidade (o Exemplo_RBF e o mini projeto da turbina)."

---

## 2. Sobre as gaussianas: centros, quantidade e σ

Na aula de dúvidas, o professor perguntou a um colega: *"Onde estão os centros? Por que está em 10 se os valores são a partir de 1400?"* A regra por trás da pergunta:

> **Os centros têm de ficar no mesmo intervalo e na mesma unidade da variável que entra na gaussiana.**

A gaussiana só "liga" (vale perto de 1) quando a entrada está perto do centro. Se todos os centros ficam longe dos dados, todas as gaussianas valem ≈ 0, e o modelo vira uma constante (só sobra o w0). **No seu projeto isso não acontece:** os centros são linhas dos próprios dados.

**"Quantas gaussianas você está usando?" / "Quantos centros?"**
É a mesma pergunta: cada gaussiana tem um centro.
> "16 gaussianas por saída: 5 para quando a entrada está parada em +1, 5 para quando está parada em −1, e 3 para cada tipo de troca da entrada (de −1 para +1 e de +1 para −1). Com o w0, são 17 pesos. Testei 5, 10, 15 e 20 centros por estado na validação. Escolhi o modelo mais simples com erro até 2% acima do menor erro, que é a regra de parcimônia do material da disciplina."

Mostre a tabela impressa na Seção 4.3. Para x, por exemplo:

| Centros por estado (M) | EMQ de validação com σ = 0,2 |
|---|---|
| **5 (M = 16)** | **0,01331** |
| 10 (M = 26) | 0,01308 (o menor de todos) |
| 15 (M = 36) | 0,01346 |
| 20 (M = 46) | 0,01348 |

> "O menor erro foi com 26 centros, mas 16 centros erra só 1,8% a mais, com 10 pesos a menos. Pela parcimônia, fiquei com 16."

**"Por que só 3 centros para as trocas?"**
> "Porque há poucas linhas de troca nos dados: 14 com (u[n], u[n−1]) = (+1, −1) e 15 com (−1, +1), contra cerca de 735 de cada estado parado."

**"Onde estão os centros?"**
> "Cada centro é um vetor de 4 números, no mesmo espaço das entradas do modelo: [s[n−1], s[n−2], u[n], u[n−1]]. Para cada combinação de (u[n], u[n−1]), ordenei as linhas dos dados pelo valor de s[n−1] e peguei linhas igualmente espaçadas nessa ordem. Assim os centros cobrem do menor ao maior valor da saída."

Mostre o gráfico da Seção 4.4 (triângulos sobre a nuvem de pontos). Para ver os centros em números:

```python
print("Centros de x  [x[n-1], x[n-2], u[n], u[n-1]]:")
print(np.round(modelo_rbf["x"][0], 3))
print("sigma de x =", modelo_rbf["x"][1])
```

**"Por que os centros estão nessa faixa?"** (a versão da pergunta do colega para o seu projeto)
> "Porque é onde os dados estão: x vai de −0,42 a 0,41, y de −0,78 a 0,69, e u só vale −1 ou +1. Como os centros são linhas dos dados, ficam automaticamente nessa faixa e na mesma unidade."

Se ele quiser ver o que acontece com centros fora da faixa (empurrando s em +10):

```python
centros, sigma, _ = modelo_rbf["y"]
longe = centros.copy()
longe[:, 0] += 10
longe[:, 1] += 10
Phi = matriz_phi(Z_y, longe, sigma)
w = np.linalg.lstsq(Phi, alvo_y, rcond=None)[0]   # lstsq: as colunas viraram ~0 e solve falharia
print("EMQ com centros longe:", round(emq(alvo_y, Phi @ w), 5), "| variância de y:", round(np.var(alvo_y), 5))
```
Resultado: EMQ = 0,0838, igual à variância de y. Ou seja, o modelo vira só a média e não explica nada (o EMQ normal é 0,0075).

**"Qual σ você usou? Por quê?"**
> "σ = 0,2 para x e 0,5 para y. Testei de 0,1 a 0,75 na validação e, junto com o número de centros, escolhi o modelo mais simples com erro até 2% acima do mínimo, preferindo σ menor."

**"Por que y tem σ maior que x?"**
> "y varia numa faixa maior que x. Com 5 centros por estado espalhados nessa faixa, os centros de y ficam mais distantes entre si, e o sino precisa ser mais largo para os vizinhos se sobreporem. É a regra do material: σ da ordem do espaçamento entre centros."

**"E se mudar o σ?"** Rode:

```python
centros = modelo_rbf["x"][0]
for sigma in [0.1, 0.2, 0.3, 0.5, 1.0, 2.0]:
    Phi = matriz_phi(Z_x, centros, sigma)
    w = equacao_normal(Phi, alvo_x)
    print(f"sigma = {sigma}: EMQ = {emq(alvo_x, Phi @ w):.5f} | maior peso = {np.abs(w).max():.3g}"
          f" | condição de ΦᵀΦ = {np.linalg.cond(Phi.T @ Phi):.0e}")
```

| σ | EMQ (x) | Maior peso | Condição de ΦᵀΦ |
|---|---|---|---|
| 0,1 | 0,01354 | 0,39 | ~10³ |
| **0,2** | **0,01222** | 0,25 | ~10⁴ |
| 0,5 | 0,01211 | 6,3 | ~10⁶ |
| 1,0 | 0,01209 | 213 | ~10⁹ |
| 2,0 | 0,01208 | 7 080 | ~10¹³ |

> "Com σ pequeno, cada gaussiana enxerga uma região pequena, sobram 'buracos' entre os centros e o erro sobe. Com σ grande, o erro quase não muda, mas as gaussianas ficam parecidas, a matriz ΦᵀΦ fica quase singular (o número de condição explode) e os pesos ficam enormes e instáveis. Por isso preferi o menor σ que já dá um erro próximo do mínimo."

**"O que é o número de condição?"**
> "Mede o quanto a solução do sistema é sensível a pequenas mudanças nos dados. Número de condição grande = pesos instáveis, que mudariam muito com um pouco de ruído diferente."

**"Qual a unidade de σ?"**
> "A mesma das entradas do modelo: a unidade da saída e da entrada, que aqui estão na mesma escala, de −1 a 1."

**"Por que não padronizou (z-score) os dados?"**
> "Porque as 4 entradas já estão numa escala parecida: s entre −0,8 e 0,7, u entre −1 e 1. Na aula de regressão múltipla a padronização era necessária porque os sensores tinham escalas muito diferentes (vibração, temperatura, ruído). Aqui não é o caso."

**"Por que tirar os centros dos dados e não usar uma grade regular?"**
> "Os dados ficam concentrados perto da diagonal s[n−1] ≈ s[n−2], porque amostras vizinhas são quase iguais (gráfico da Seção 4.4). Uma grade poria muitos centros onde não há dados. Essas gaussianas não teriam informação para calcular o peso e a matriz ficaria mal condicionada."

**"Por que não um centro em cada ponto dos dados?"**
> "Seriam 1499 centros para 1499 equações: a curva passaria exatamente por todos os pontos, copiando o ruído (sobreajuste), e a matriz ficaria quase singular."

**"Uma gaussiana com u = +1 influencia linhas com u = −1?"**
> "Praticamente não. A diferença entre +1 e −1 é 2 em cada uma das duas coordenadas de u, então a distância ao quadrado é pelo menos 8. Com σ = 0,5 a gaussiana vale cerca de 10⁻⁷, e com σ = 0,2, cerca de 10⁻⁴⁴. Na prática o modelo funciona como modelos separados para cada combinação de u. Por isso cada combinação precisa ter seus próprios centros."

**"E se você tirasse os centros de troca?"** Rode:

```python
centros, sigma, _ = modelo_rbf["y"]
troca = Z_y[:, 2] != Z_y[:, 3]                  # linhas em que a entrada trocou
for nome, C in [("com centros de troca", centros), ("sem centros de troca", centros[:10])]:
    Phi = matriz_phi(Z_y, C, sigma)
    previsto = Phi @ equacao_normal(Phi, alvo_y)
    print(f"{nome}: EMQ total = {emq(alvo_y, previsto):.5f} | EMQ só nas trocas = {emq(alvo_y[troca], previsto[troca]):.4f}")
```
Resultado em y: sem os centros de troca, o erro nas linhas de troca sobe de 0,0057 para 0,22, e o EMQ total sobe de 0,0075 para 0,0117. **Lição:** os centros precisam cobrir toda a região dos dados, inclusive as situações raras.

**"Quantas gaussianas influenciam cada previsão?"**
> "Poucas. Em média, cerca de 4 gaussianas valem mais que 0,1 em cada instante. As distantes valem praticamente zero."

**"A gaussiana tem 2σ² no denominador. Por quê?"**
> "É a forma do material da disciplina: φ = exp(−‖z − c‖² / (2σ²)). Com o 2, σ tem o significado de desvio padrão. Sem o 2 mudaria só a escala de σ, não o método."

**"Por que alguns pesos são negativos?"**
> "Cada peso diz quanto aquela gaussiana empurra a previsão para cima (positivo) ou para baixo (negativo). O que importa é a soma de todas, não cada peso sozinho."

---

## 3. Sobre a matemática

**"Por que a RBF é 'linear' se a gaussiana não é?"**
> "Ela é não linear nas entradas, mas linear nos pesos: a saída é Φ·w. Depois de escolher centros e σ, cada gaussiana vira uma coluna de números. Por isso os pesos saem da equação normal, sem iterações, igual ao MMQ."

**"Deduza a equação normal."**
> "Quero minimizar J(w) = (1/N)·‖Φw − s‖². O gradiente é ∇J = (2/N)·Φᵀ(Φw − s). Igualando a zero, sai ΦᵀΦ·w = Φᵀs. Como J tem formato de tigela (quadrática convexa), esse ponto é o mínimo."

**"Qual a diferença entre a matriz da RBF e a do MMQ?"**
> "No MMQ, os 4 regressores entram direto nas colunas (1499 × 4). Na RBF, cada coluna é uma gaussiana aplicada aos 4 regressores, mais uma coluna de 1s (1499 × 17). A equação normal é a mesma."

**"Por que `np.linalg.solve` e não `inv`?"**
> "solve resolve o sistema direto. É mais rápido e mais preciso do que calcular a inversa e multiplicar, e o resultado é o mesmo de w = (ΦᵀΦ)⁻¹Φᵀs. O material do mini projeto também diz para nunca calcular a inversa explicitamente."

**"No mini projeto a aula usou `pinv`. Por que você usou `solve`?"**
> "As duas dão o mesmo resultado quando ΦᵀΦ é bem condicionada, que é o caso aqui, porque escolhi σ pequeno justamente para isso. pinv seria mais seguro se a matriz fosse quase singular."

**"Por que `solve` funciona se Φ é 1499 × 17 (não quadrada)?"**
> "Porque não resolvo Φw = s, que tem mais equações que incógnitas e não tem solução exata. Resolvo ΦᵀΦ·w = Φᵀs, e ΦᵀΦ é quadrada (17 × 17)."

**"Por que não usou gradiente descendente?"**
> "A equação normal dá a solução exata numa única conta. O gradiente descendente chegaria ao mesmo resultado por iterações, como o mini projeto mostrou. Para um sistema 17 × 17, a equação normal é mais simples e mais rápida."

**"Explique o fatiamento `s[1:-1]`, `s[:-2]`, `s[2:]`."**
> "Para n de 2 a 1500: s[2:] é s[n], s[1:-1] é s[n−1] e s[:-2] é s[n−2]. As três têm 1499 elementos, então cada linha fica alinhada no mesmo instante. A Seção 3 imprime a primeira linha para conferir."

---

## 4. Sobre a validação

**"Por que separou validação?"**
> "Para escolher o número de centros e o σ sem sobreajuste. Se eu escolhesse pelo erro nos dados de ajuste, o modelo com mais gaussianas sempre ganharia, porque copia o ruído."

**"Por que sorteio e não '1 de cada 5'?"**
> "A entrada troca a cada 50 amostras, que é múltiplo de 5. Com '1 de cada 5', as linhas de troca cairiam todas no mesmo grupo, e a validação poderia ficar sem nenhuma troca. O sorteio de 20% mistura as trocas entre treino e validação. A semente é fixa, então o resultado é sempre igual."

**"Por que, depois de validar, ajustou com todos os dados?"**
> "A validação serve só para escolher as configurações (número de centros e σ). Depois de escolhidas, usar todos os dados dá a estimativa mais precisa dos pesos."

**"O que é a regra dos 2%?"**
> "É a parcimônia: entre modelos com erro praticamente igual, fico com o mais simples. Considero 'praticamente igual' até 2% acima do menor erro. O material do mini projeto diz para escolher o modelo mais simples cujo erro de validação esteja próximo do mínimo."

---

## 5. Sobre as métricas

**"O que é o EMQ e qual valor é bom?"**
> "É a média dos erros ao quadrado. Com dados ruidosos, nunca chega a zero: o mínimo possível é mais ou menos a variância do ruído. √EMQ dá o erro típico: em y, cerca de 0,09."

**"O que significa o r de Pearson?"**
> "Mede o quanto dois sinais variam juntos de forma linear, de −1 a 1. r² é a fração da variação explicada: 0,954² ≈ 0,91, ou seja, o modelo explica 91% da variação de y."

**"Por que a regressão da RBF deu inclinação 1 e intercepto 0 exatos?"**
> "É uma propriedade do MMQ com termo constante (w0): o erro que sobra fica sem correlação com a coluna de 1s e com a curva ajustada, então a melhor reta medido × ajustado é 0 + 1·ajustado. O ajuste não tem viés. O MMQ do enunciado não tem termo constante, por isso deu só quase 1 e quase 0."

**"Para que serve a faixa ±√EMQ no gráfico?"**
> "Mostra o erro típico em volta da reta. A maioria dos pontos fica dentro dela."

**"Por que x tem r tão menor que y?"**
> "Relação sinal/ruído. Em x, o ruído é quase do tamanho da variação do próprio sinal. Em y, o sinal varia muito mais que o ruído. Não é falha do método: nenhuma curva consegue acompanhar de perto pontos tão espalhados."

---

## 6. Sobre os resultados e as conclusões

**"Por que a RBF não ganhou do MMQ prevendo um passo à frente?"**
> "Porque, com esse ruído, a relação entre o passado e o próximo valor é praticamente linear. As gaussianas dão flexibilidade para fazer curvas, mas não há curvatura relevante para capturar. Na validação também empatam: em x, 0,01331 na RBF contra 0,01334 no MMQ; em y, 0,00676 contra 0,00671. Para esse uso, o modelo linear com 4 parâmetros basta."

**"Então para que serve a RBF?"**
> "Na simulação livre de y ela foi um pouco melhor: EMQ 0,0185 contra 0,0197, e r 0,914 contra 0,878. Ela reproduz melhor a forma dos picos. Ou seja, a parte não linear ajuda um pouco quando o modelo precisa andar sozinho."

**"Qual a diferença entre um passo à frente e simulação livre?"**
> "Um passo à frente recebe os valores medidos de s[n−1] e s[n−2] e só prevê o próximo. A simulação livre recebe só a entrada u e usa as próprias previsões como passado, então os erros se acumulam. É o teste que mostra se o modelo simula o sistema, que é a justificativa do projeto."

**"Por que a curva ajustada é serrilhada?"**
> "Porque ela usa os valores medidos (com ruído) de s[n−1] e s[n−2] como entrada, e parte desse ruído passa para a previsão."

**"Por que a simulação de x virou uma onda quadrada?"**
> "Porque a memória estimada ficou baixa: a1 + a2 = 0,51. Os valores passados medidos têm muito ruído, então o MMQ 'confia pouco' neles e diminui a1 e a2. Com pouca memória, o modelo só copia a entrada. A RBF sofre o mesmo efeito, porque também foi ajustada com regressores ruidosos."

Se ele duvidar, mostre que com um sinal suavizado a memória aparece:

```python
janela = np.ones(25) / 25                          # média móvel de 25 amostras (0,5 s)
for nome, s in [("x", x), ("y", y)]:
    s_suave = np.convolve(s, janela, mode="same")
    Z_s, alvo_s = montar_regressores(s_suave, u)
    print(nome, "a1, a2, b0, b1 com sinal suavizado:", np.round(equacao_normal(Z_s, alvo_s), 3))
```
Resultado: x → a1 ≈ 1,48 e a2 ≈ −0,48; y → a1 ≈ 1,72 e a2 ≈ −0,72. Com menos ruído, a1 + a2 fica perto de 1, o que significa memória forte. Isso confirma que o ruído é o que "apaga" a memória.

**"Como melhoraria o modelo?"**
> "Ajustaria a partir de um sinal filtrado, aumentaria a ordem do modelo (mais valores passados) ou usaria métodos próprios para ruído nos regressores."

**"Qual modelo você recomendaria?"**
> "Para prever um passo à frente, o MMQ: mesmo desempenho com 4 parâmetros em vez de 17. Para simular o sistema só a partir da entrada, a RBF, que foi um pouco melhor em y."

---

## 7. Se não souber responder

Diga o que você **sabe** e como **testaria**. Exemplo: "Não testei isso, mas daria para verificar mudando os valores de σ na Seção 4.3 (linha 127 do `projeto_u2.py`) e comparando o EMQ de validação." Isso mostra que você entende o método, mesmo sem a resposta pronta.
