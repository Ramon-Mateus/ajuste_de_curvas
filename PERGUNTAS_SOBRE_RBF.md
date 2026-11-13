# Perguntas do professor sobre as gaussianas (RBF): como responder

Este guia parte das perguntas que o professor fez ao colega na aula de dúvidas e mostra como respondê-las **com os números do seu projeto**. No fim, há outras perguntas que ele pode fazer na mesma linha.

Todos os números abaixo foram calculados com o seu código e os seus dados.

---

## 1. Essas perguntas fazem sentido para o seu projeto?

Fazem, e são as perguntas mais importantes sobre RBF. Os três parâmetros que **você escolhe** num modelo RBF são:

1. **quantas** gaussianas usar (M);
2. **onde** ficam os centros (c₁, c₂, …, c_M);
3. **qual** a largura de cada uma (σ).

Os pesos (w) o computador calcula sozinho pela equação normal. Por isso o professor pergunta justamente sobre M, centros e σ: é aí que o aluno mostra se entendeu o método ou só copiou código.

### O que o professor estava cobrando do colega

> "Por que está em 10 se os valores são a partir de 1400?"

A regra por trás dessa pergunta é:

> **Os centros têm que ficar no mesmo intervalo e na mesma unidade da variável de entrada da gaussiana.**

A gaussiana φ(v) = exp(−(v − c)² / (2σ²)) só "liga" (vale perto de 1) quando a entrada **v** está perto do centro **c**. Pelo que você descreveu, o colega colocou os centros perto de 10 enquanto a variável de entrada tinha valores na casa de 1400 (por exemplo, o número da amostra, que vai de 0 a 1500). Nesse caso, nenhum dado fica perto de nenhum centro e todas as gaussianas valem praticamente zero. O modelo vira uma **reta horizontal** (só sobra o w₀) e não aprende nada.

**No seu projeto isso não acontece:** a entrada das gaussianas é o **tempo t**, que vai de 0 a 30 s, e os centros também vão de 0 a 30 s, na mesma unidade (segundos).

---

## 2. Respostas prontas para as três perguntas

### "Quantas gaussianas você está usando?" / "Quantos centros escolheu?"

As duas perguntas são a mesma coisa: **cada gaussiana tem um centro**, então número de gaussianas = número de centros = M.

> "Uso **40 gaussianas para x(t)** e **50 para y(t)**. Escolhi por validação: separei 1 de cada 5 amostras, testei M de 20 a 200 e medi o erro nas amostras que o modelo não viu. Com poucas gaussianas (M = 20) o erro de validação era bem maior: 1,7 vez maior em x e 12 vezes maior em y. Acima de 40–50 o erro para de cair e, em x, volta a subir, porque a curva começa a copiar o ruído. Como o erro fica quase plano perto do mínimo, peguei o **menor M** com erro até 2% acima do mínimo (regra da parcimônia)."

Somando o w₀, o modelo tem **41 pesos para x** e **51 para y**.

### "Onde estão os centros?"

> "Os centros estão **no eixo do tempo, de 0 a 30 segundos, igualmente espaçados**, cobrindo todo o intervalo dos dados. Para x, com 40 centros, eles ficam a cada 0,77 s: 0; 0,77; 1,54; 2,31; … ; 29,23; 30. Para y, com 50 centros, ficam a cada 0,61 s. É a linha `centros = np.linspace(0, 30, M)`."

| | x(t) | y(t) |
|---|---|---|
| Número de gaussianas M | 40 | 50 |
| Primeiro centro | 0 s | 0 s |
| Último centro | 30 s | 30 s |
| Espaçamento entre centros | 0,769 s | 0,612 s |
| σ (largura) | 0,769 s | 0,612 s |
| Amostras entre dois centros | ≈ 38 | ≈ 31 |
| Total de pesos (w₀ + M) | 41 | 51 |

### "Por que está em X se os valores são a partir de Y?" (a versão para o seu projeto)

Se ele perguntar algo como "por que os centros vão de 0 a 30?":

> "Porque a entrada da minha gaussiana é o tempo, e o tempo dos dados vai de 0 a 30 s. Os centros precisam estar onde os dados estão. Se eu colocasse os centros só de 0 a 10, as gaussianas não cobririam de 10 a 30 s e a curva nesse trecho ficaria praticamente zero."

Isso foi testado: com os centros só entre 0 e 10 s, a curva em t = 20 s vale ≈ 0,001 (praticamente zero). O EMQ de y piora de 0,0053 para 0,051 (quase 10 vezes) e o de x de 0,0097 para 0,0147.

---

## 3. Como mostrar isso no código na hora

As linhas que respondem às três perguntas estão no `projeto_u2.py` (e na seção 3 do notebook):

| O quê | Linha no `projeto_u2.py` | Código |
|---|---|---|
| Fórmula da gaussiana | 80 | `np.exp(-(tempos - centros[j]) ** 2 / (2 * sigma ** 2))` |
| Onde ficam os centros | 84 | `centros = np.linspace(0, 30, M)` |
| Largura σ | 85 | `sigma = centros[1] - centros[0]` |
| Cálculo dos pesos | 87 | `w = np.linalg.solve(Phi.T @ Phi, Phi.T @ sinal)` |
| Valores de M testados | 94 | `lista_M = np.arange(20, 201, 10)` |
| Escolha de M | 106–107 | `limite = 1.02 * erros_val.min()` … |

Se ele pedir para **ver** os centros, rode isto depois do código principal (no notebook, numa célula nova; no script, no final):

```python
print("Centros de x:", np.round(c_x, 2))
print("sigma de x =", round(s_x, 4), "s")
print("Centros de y:", np.round(c_y, 2))
print("sigma de y =", round(s_y, 4), "s")

plt.figure(figsize=(14, 3))
plt.plot(t, x, "o", ms=2, alpha=0.3, label="x medido")
plt.plot(c_x, np.zeros_like(c_x), "k^", label="centros das gaussianas")
plt.xlabel("Tempo (s)"); plt.legend(); plt.show()
```

O gráfico mostra os 40 triângulos espalhados de 0 a 30 s, embaixo dos dados.

---

## 4. Por que centros fora do lugar estragam o modelo (para entender de verdade)

Pense em cada gaussiana como uma **lanterna** apontada para um ponto do eixo (o centro), com um facho de largura σ. O modelo só consegue "desenhar" a curva onde alguma lanterna ilumina. Onde nenhuma ilumina, sobra só o w₀ (uma constante).

Testei três cenários com os seus dados:

| Cenário | EMQ de x | EMQ de y | O que acontece |
|---|---|---|---|
| **Seu modelo:** entrada = tempo (0–30 s), centros de 0 a 30 | 0,0097 | 0,0053 | Correto: ajusta no nível do ruído |
| Entrada = tempo, centros só de 0 a 10 | 0,0147 | 0,0512 | De 10 a 30 s a curva fica praticamente zero |
| Entrada = **número da amostra** (0–1500), centros de 0 a 30 | 0,0203 | 0,0828 | Quase tudo fica sem gaussiana: o EMQ fica igual à variância do sinal (0,0206 e 0,0837), ou seja, o modelo não explica nada |
| Entrada = número da amostra, centros de **0 a 1500** | 0,0097 | 0,0053 | Volta a funcionar, idêntico ao seu |

**Lição da última linha:** não importa se a entrada é o tempo em segundos ou o número da amostra. O que importa é que **centros e σ estejam na mesma escala da entrada**. Com o número da amostra, o σ certo seria ≈ 38 amostras (para x), e não 0,77.

Provavelmente foi isso que aconteceu com o colega: os centros estavam numa escala (perto de 10) e a variável de entrada em outra (perto de 1400).

---

## 5. Outras perguntas que ele pode fazer (e respostas)

### Sobre σ (a largura)

**"Qual o σ que você usou? Por quê?"**
> "σ igual ao espaçamento entre os centros: 0,77 s para x e 0,61 s para y. É a regra prática da aula: assim cada gaussiana se sobrepõe com as vizinhas (no centro da vizinha ela ainda vale e^(−1/2) ≈ 0,61), sem que todas fiquem iguais."

**"E se mudar o σ?"** (testado com os seus dados, centros fixos)

| σ | EMQ de x | EMQ de y | Número de condição de ΦᵀΦ | Maior peso \|w\| (x) |
|---|---|---|---|---|
| 0,25 × espaçamento | 0,0130 | 0,0264 | ~10³ | 0,3 |
| 0,5 × espaçamento | 0,0104 | 0,0067 | ~10⁵ | 0,4 |
| **1 × espaçamento (usado)** | **0,0097** | **0,0053** | ~10⁶ | 1,7 |
| 2 × espaçamento | 0,0098 | 0,0053 | ~10¹⁶ | ≈ 166 000 |
| 4 × espaçamento | 0,0161 | 0,0307 | ~10¹⁷ | ≈ 190 milhões |

> "σ pequeno demais: as gaussianas não se encostam e a curva fica com 'buracos' entre os centros (o erro sobe). σ grande demais: todas as gaussianas ficam parecidas, a matriz ΦᵀΦ fica quase singular (número de condição 10¹⁶) e os pesos explodem para centenas de milhares. Repare que com 2× o EMQ é quase igual, mas os pesos ficam absurdos e instáveis: pequenas mudanças nos dados mudariam tudo. Por isso σ = espaçamento é a escolha segura."

**"Qual a unidade de σ?"**
> "A mesma da entrada: segundos. σ e os centros vivem no eixo do tempo."

### Sobre os centros

**"Por que centros igualmente espaçados?"**
> "Porque o sinal oscila do mesmo jeito ao longo de todo o tempo, então não há uma região que precise de mais detalhe que outra. Se houvesse, por exemplo, uma mudança brusca num trecho, valeria concentrar mais centros ali."

**"Por que não colocar um centro em cada ponto dos dados?"**
> "Com 1501 centros (um por amostra) e σ pequeno, a curva passa **exatamente** por todos os pontos: o EMQ de treino dá praticamente zero (≈ 10⁻²⁸). Mas aí ela copia o ruído inteiro, que é exatamente o que eu quero remover. Seria sobreajuste total."

**"Existe outro jeito de escolher os centros?"**
> "Sim: sortear pontos dos próprios dados ou usar agrupamento (k-means), que coloca os centros onde há mais dados. Aqui o espaçamento uniforme é suficiente, porque os dados estão distribuídos uniformemente no tempo (uma amostra a cada 0,02 s)."

**"Quantas gaussianas influenciam cada ponto?"**
> "Em t = 15 s, por exemplo, 6 gaussianas valem mais que 0,01 e 4 valem mais que 0,1. Cada ponto da curva é a combinação de umas 4 a 6 gaussianas próximas; as mais distantes praticamente não contribuem (a 3σ do centro a gaussiana já vale só 0,011)."

### Sobre M (o número de gaussianas)

**"Por que x e y têm M diferentes?"**
> "Porque a validação foi feita separadamente para cada sinal. y é menos ruidoso e tem picos mais definidos, então aproveita um pouco mais de detalhe (50). x é muito ruidoso: acima de 40 as gaussianas extras já começam a seguir o ruído e o erro de validação sobe."

**"E se usar 200 gaussianas?"**
> "O erro de treino cai, mas o de validação sobe (em x, de 0,0100 para 0,0109). A curva começa a ondular para seguir o ruído. Menor erro de treino não quer dizer modelo melhor."

**"Por que testou de 20 a 200, de 10 em 10?"**
> "Com menos de 20, os centros ficam a mais de 1,5 s de distância, longe demais para seguir uma oscilação de período 2 s. Acima de 200 o erro de validação já estava subindo. De 10 em 10 basta, porque a curva de validação é quase plana perto do mínimo."

### Sobre os pesos e a matriz

**"Qual o tamanho da matriz Φ?"**
> "1501 × 41 para x (uma linha por amostra; uma coluna de 1s mais uma coluna por gaussiana) e 1501 × 51 para y. Ou seja, 1501 equações para 41 incógnitas: um sistema sobredeterminado, resolvido por mínimos quadrados."

**"O que é o w₀? Por que ele é −0,93 se o sinal tem média perto de zero?"**
> "w₀ é o termo constante (a coluna de 1s). Ele não é a média do sinal sozinho porque as gaussianas se sobrepõem: a soma de todas elas no meio do intervalo dá aproximadamente uma constante (≈ 2,5). Então w₀ e os outros pesos se compensam. O que importa é a soma final, não cada peso isolado."

**"O que significa um peso negativo?"**
> "Que aquela gaussiana 'puxa a curva para baixo' perto do seu centro. A curva final é a soma de sinos para cima (pesos positivos) e para baixo (pesos negativos)."

**"O modelo RBF é linear ou não linear?"**
> "Não linear no tempo (as gaussianas são curvas), mas linear nos pesos. Por isso os pesos saem de um sistema linear (a equação normal), sem precisar de iterações."

**"A sua gaussiana tem 2σ² no denominador. Por quê?"**
> "É a forma usada no material da disciplina: φ = exp(−(t − c)² / (2σ²)). Com o 2, σ tem o mesmo significado do desvio padrão de uma distribuição normal. Algumas referências usam só σ², o que só muda a escala de σ, não o método."

### Sobre a escolha de fazer a RBF no tempo

**"Por que a entrada da gaussiana é o tempo e não y[n−1], y[n−2], u[n], u[n−1]?"**
> "Porque usei a RBF como **filtro**: ela encontra a curva suave por trás do ruído (o enunciado fala em 'identificação da dinâmica **ou** filtragem'). O modelo com y[n−1], y[n−2], u[n], u[n−1] eu fiz com o MMQ, que é a parte da identificação da dinâmica."

**"E se fosse uma RBF com esses regressores, onde ficariam os centros?"**
> "No espaço dos regressores: cada centro seria um vetor de 4 números (um valor de y[n−1], um de y[n−2], um de u[n] e um de u[n−1]). Pela mesma regra, eles teriam de ficar na faixa desses valores, por exemplo, de −0,8 a 0,7 para y e em −1 ou +1 para u. Colocá-los em 0 a 30 não faria sentido: é exatamente o tipo de erro da pergunta do colega."

---

## 6. Dicas de postura na apresentação

- **Responda com o número primeiro**, depois com o porquê: "40 gaussianas, escolhi por validação porque…".
- **Aponte a linha do código** enquanto fala (a tabela da seção 3 tem as linhas).
- Se não souber algo, diga o que você **sabe** e como **testaria**: "não testei isso, mas daria para verificar mudando o σ na linha 85 e comparando o EMQ".
- Lembre-se da regra principal: **centros e σ sempre na mesma escala da entrada da gaussiana**. Quase toda pergunta sobre centros volta para isso.
