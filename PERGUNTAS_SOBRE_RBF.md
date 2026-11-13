# Perguntas do professor sobre as gaussianas (RBF): como responder

Este guia parte das perguntas que o professor fez ao colega na aula de dúvidas e mostra como respondê-las **com os números do seu projeto**. No fim, há outras perguntas que ele pode fazer na mesma linha. Todos os números foram calculados com o seu código e os seus dados.

O projeto tem **duas** RBFs. Saiba qual é qual:

| | RBF dinâmica (Seção 4) | RBF no tempo (Seção 3) |
|---|---|---|
| Papel | **O modelo pedido no enunciado** (vale nota) | Preparação/referência (filtro) |
| Entrada da gaussiana | vetor de 4 números: [s[n−1], s[n−2], u[n], u[n−1]] | 1 número: o tempo t |
| Número de gaussianas | 26 (para x e para y) | 40 (x) e 50 (y) |
| Centros | 26 linhas dos próprios dados | de 0 a 30 s, igualmente espaçados |
| σ | 0,2 (x) e 0,3 (y), por validação | 0,77 s (x) e 0,61 s (y) = espaçamento |

Quando o professor perguntar "quantas gaussianas?", responda primeiro sobre a **RBF dinâmica**, que é o modelo do enunciado.

---

## 1. O que o professor estava cobrando

> "Onde estão os centros? Por que está em 10 se os valores são a partir de 1400?"

A regra por trás dessa pergunta é:

> **Os centros têm que ficar no mesmo intervalo e na mesma unidade da variável que entra na gaussiana.**

A gaussiana φ(z) = exp(−‖z − c‖² / (2σ²)) só "liga" (vale perto de 1) quando a entrada **z** está perto do centro **c**. Pelo que você descreveu, o colega deixou os centros perto de 10 enquanto a entrada tinha valores na casa de 1400. Nenhum dado fica perto de nenhum centro, todas as gaussianas valem praticamente zero, e o modelo vira uma constante (só sobra o w₀).

**No seu projeto isso não acontece:** os centros da RBF dinâmica são **linhas dos próprios dados**, então estão, por construção, onde os dados estão.

---

## 2. Respostas prontas para as três perguntas

### "Quantas gaussianas você está usando?" / "Quantos centros escolheu?"

São a mesma pergunta: **cada gaussiana tem um centro**.

> "No modelo dinâmico uso **26 gaussianas para cada saída**: 10 para quando a entrada está parada em +1, 10 para quando está parada em −1, e 3 para cada tipo de troca da entrada (de −1 para +1 e de +1 para −1). Somando o w₀, são 27 pesos. Escolhi 10 por estado testando 5, 10, 15 e 20 na validação: 10 deu o menor erro nas duas saídas."

| Centros por estado | Total M | EMQ de validação (x) | EMQ de validação (y) |
|---|---|---|---|
| 5 | 16 | 0,01331 | 0,00689 |
| **10** | **26** | **0,01308** | **0,00664** |
| 15 | 36 | 0,01346 | 0,00673 |
| 20 | 46 | 0,01348 | 0,00689 |

**Por que só 3 para as trocas?** Porque há poucas linhas de troca nos dados: 14 linhas com (u[n], u[n−1]) = (+1, −1) e 15 com (−1, +1), contra ≈ 735 de cada estado parado.

### "Onde estão os centros?"

> "Cada centro é um **vetor de 4 números**, no mesmo espaço das entradas do modelo: [s[n−1], s[n−2], u[n], u[n−1]]. Eu peguei os centros dos próprios dados: para cada combinação de (u[n], u[n−1]), ordenei as linhas pelo valor de s[n−1] e peguei linhas igualmente espaçadas nessa ordem. Assim os centros cobrem desde o menor até o maior valor da saída. O gráfico da Seção 4.4 mostra os centros (triângulos) em cima da nuvem de dados."

Exemplos (saída do código):

| Centro | s[n−1] | s[n−2] | u[n] | u[n−1] |
|---|---|---|---|---|
| x, c₁ (menor valor com u = +1) | −0,416 | −0,309 | +1 | +1 |
| x, c₁₁ (u = −1) | −0,273 | −0,220 | −1 | −1 |
| y, c₁ (menor valor com u = +1) | −0,774 | −0,609 | +1 | +1 |
| y, c₂₁ (troca de −1 para +1) | −0,718 | −0,782 | +1 | −1 |

### "Por que os centros estão nessa faixa?" (a versão da pergunta do colega para o seu projeto)

> "Porque é onde os dados estão: x varia de −0,42 a 0,41, y de −0,78 a 0,69, e u só vale −1 ou +1. Como os centros são linhas dos próprios dados, eles ficam automaticamente nessa faixa e na mesma unidade."

Isso foi testado: se os centros tivessem s entre 0 e 30, longe dos dados, o EMQ de y iria de 0,0076 para 0,0751, quase igual à variância do sinal (0,0838). Ou seja, o modelo não explicaria praticamente nada.

---

## 3. Como mostrar isso no código na hora

| O quê | Linha no `projeto_u2.py` | Código |
|---|---|---|
| Distância ao centro (4D) | 75 | `distancia2 = np.sum((Z - centros[j]) ** 2, axis=1)` |
| Fórmula da gaussiana | 76 | `np.exp(-distancia2 / (2 * sigma ** 2))` |
| Escolha dos centros | 156–164 | `escolher_centros`: as 4 combinações de u, ordena por s[n−1] (linha 161), escolhe igualmente espaçados (linha 162) |
| Valores de σ testados | 180 | `lista_sigma = [0.1, 0.2, 0.3, 0.4, 0.5, 0.75]` |
| Escolha de σ | 191–192 | menor σ com erro até 2% acima do mínimo |
| Centros finais e pesos | 203–207 | `centros_x = escolher_centros(Z_x)` … `equacao_normal(Phi_x, alvo_x)` |

Se ele pedir para **ver** os centros, rode isto no final (no notebook, numa célula nova):

```python
print("Centros de x  [x[n-1], x[n-2], u[n], u[n-1]]:")
print(np.round(centros_x, 3))
print("sigma de x =", sigma_rbf["x"])
```

E mostre o gráfico da Seção 4.4 ("dados e os 26 centros no plano").

---

## 4. Por que centros no lugar errado estragam o modelo

Pense em cada gaussiana como uma **lanterna** apontada para um ponto (o centro), com um facho de largura σ. O modelo só consegue "desenhar" a resposta onde alguma lanterna ilumina.

Cenários testados com os seus dados (EMQ um passo à frente):

| Cenário | EMQ de x | EMQ de y | O que acontece |
|---|---|---|---|
| **Seu modelo** (centros dos dados, cobrindo as 4 combinações de u) | 0,01195 | 0,00757 | correto |
| Centros com s entre 0 e 30 (fora da faixa dos dados) | 0,01751 | 0,07511 | quase nada é iluminado: o modelo explica quase nada |
| Sem os 6 centros de troca (só 20 centros) | 0,01201 | 0,01173 | em y, nas linhas de troca da entrada, o erro sobe de 0,006 para 0,221 |
| 26 centros igualmente espaçados no tempo | 0,01198 | 0,01118 | o sinal é periódico, os centros caem sempre nas mesmas fases e uma combinação de u fica sem centro |

**Lição:** não basta ter centros; eles precisam **cobrir a região onde os dados estão**, incluindo situações raras (as trocas da entrada).

**E na RBF no tempo (Seção 3)?** Mesma regra: a entrada é o tempo (0 a 30 s), então os centros vão de 0 a 30 s. Se a entrada fosse o número da amostra (0 a 1500), os centros teriam de ir de 0 a 1500 e σ seria ≈ 38 amostras. Provavelmente foi esse tipo de troca de escala que aconteceu com o colega.

---

## 5. Outras perguntas que ele pode fazer (e respostas)

### Sobre σ (a largura)

**"Qual σ você usou? Por quê?"**
> "σ = 0,2 para x e 0,3 para y. Testei de 0,1 a 0,75 e medi o erro na validação. Peguei o menor σ com erro até 2% acima do mínimo."

**"E se mudar o σ?"** (26 centros fixos, ajuste com todos os dados)

| σ | EMQ x | Maior peso \|w\| (x) | EMQ y | Maior peso \|w\| (y) | Condição de ΦᵀΦ (y) |
|---|---|---|---|---|---|
| 0,1 | 0,01279 | 0,8 | 0,02110 | 1,2 | ~10³ |
| 0,2 | **0,01195** | 3,8 | 0,00799 | 0,8 | ~10⁴ |
| 0,3 | 0,01192 | 34 | **0,00757** | 4,0 | ~10⁷ |
| 0,5 | 0,01195 | 497 | 0,00743 | 60 | ~10¹⁰ |
| 1,0 | 0,01194 | 52 864 | 0,00739 | 12 230 | ~10¹⁴ |
| 2,0 | 0,01190 | 7,8 milhões | 0,00738 | 1,4 milhão | ~10¹⁸ |

> "Com σ pequeno, cada gaussiana só enxerga uma região minúscula e sobram 'buracos' entre os centros: o erro sobe. Com σ grande, o erro quase não muda, mas as gaussianas ficam todas parecidas, a matriz ΦᵀΦ fica quase singular e os pesos explodem para milhões. Esses pesos são instáveis: uma pequena mudança nos dados mudaria tudo. Por isso escolhi o menor σ que já dá o erro mínimo."

**"Qual a unidade de σ?"**
> "A mesma das entradas do modelo: a unidade da saída (s) e da entrada (u), que aqui estão na mesma escala, de −1 a 1."

**"Por que não padronizou (z-score) os dados?"**
> "Porque as 4 entradas já estão numa escala parecida: s entre −0,8 e 0,7, u entre −1 e 1. Padronizar não mudaria muito e acrescentaria uma etapa."

### Sobre os centros

**"Por que tirar os centros dos próprios dados e não fazer uma grade regular?"**
> "Testei a grade. Os dados ficam concentrados perto da diagonal s[n−1] ≈ s[n−2] (amostras vizinhas são quase iguais), então uma grade põe muitos centros onde não há dados. Essas gaussianas ficam sem informação e a matriz fica quase singular. Pegando centros dos dados, todos ficam onde há informação."

**"Por que não um centro em cada ponto dos dados?"**
> "Seriam 1499 centros para 1499 equações: a curva passaria exatamente por todos os pontos, copiando o ruído (sobreajuste), e a matriz ficaria quase singular."

**"Quantas gaussianas influenciam cada previsão?"**
> "Poucas. Num instante típico de x, o centro mais próximo está a 0,07 de distância e vale 0,93; cerca de 8 gaussianas valem mais que 0,1. Em y, cerca de 3. As distantes valem praticamente zero."

**"Uma gaussiana com u = +1 influencia os dados com u = −1?"**
> "Não. A diferença de u entre +1 e −1 é 2 em cada coordenada de u, então a distância é grande e a gaussiana vale ≈ 10⁻²². Na prática o modelo funciona como **dois modelos**, um para a entrada em +1 e outro para −1, mais os centros de troca. Por isso precisei de centros para cada combinação de u."

**"Por que x e y têm o mesmo número de centros e σ diferentes?"**
> "O número de centros por estado foi testado nas duas saídas e 10 foi o melhor nas duas. O σ foi escolhido separadamente: y varia numa faixa maior (−0,78 a 0,69) que x (−0,42 a 0,41), então precisa de gaussianas um pouco mais largas."

### Sobre os pesos e a matriz

**"Qual o tamanho da matriz Φ?"**
> "1499 × 27: uma linha para cada instante de n = 2 até 1500, uma coluna de 1s e uma coluna por gaussiana. São 1499 equações para 27 incógnitas, resolvidas pela equação normal."

**"Por que alguns pesos vizinhos são grandes e de sinais opostos?"** (ex.: em x, w₂ = −1,79 e w₃ = 3,79)
> "Porque os centros c₂ e c₃ estão muito próximos e as gaussianas deles se sobrepõem. Os pesos se compensam: o que importa é a soma final, não cada peso isolado."

**"O que é o w₀?"**
> "O termo constante (coluna de 1s). Ele ajusta o nível médio da resposta. Também é o que faz a regressão medido × ajustado dar inclinação 1 e intercepto 0 exatos."

**"O modelo RBF é linear ou não linear?"**
> "Não linear nas entradas (por causa das gaussianas), mas linear nos pesos. Por isso os pesos saem da mesma equação normal do MMQ, sem iterações."

**"A sua gaussiana tem 2σ² no denominador. Por quê?"**
> "É a forma do material da disciplina: φ = exp(−‖z − c‖² / (2σ²)). Com o 2, σ tem o significado de desvio padrão. Usar só σ² mudaria a escala de σ, não o método."

### Sobre a RBF no tempo (se ele perguntar da Seção 3)

**"Quantas gaussianas e onde estão os centros na RBF no tempo?"**
> "40 para x e 50 para y, escolhidas por validação. Os centros estão no eixo do tempo, de 0 a 30 s, igualmente espaçados: a cada 0,77 s em x e a cada 0,61 s em y. σ é igual ao espaçamento, a regra da aula, para os sinos vizinhos se sobreporem sem ficarem iguais. Ela serve de preparação e de referência; o modelo do enunciado é o da Seção 4."

---

## 6. Dicas de postura na apresentação

- **Responda com o número primeiro**, depois o porquê: "26 gaussianas, 10 por estado da entrada mais 3 por tipo de troca; testei 5, 10, 15 e 20 e 10 foi o melhor".
- **Aponte a linha do código** enquanto fala (tabela da seção 3).
- **Mostre o gráfico dos centros** (Seção 4.4): ele responde "onde estão os centros" visualmente.
- Se não souber algo, diga o que você **sabe** e como **testaria**: "não testei isso, mas daria para verificar mudando o σ na linha 180 e comparando o EMQ".
- Regra principal: **os centros têm que estar onde os dados estão, na mesma escala da entrada da gaussiana.**
